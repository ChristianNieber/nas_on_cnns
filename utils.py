import random
import tensorflow as tf
import keras
from keras import backend
from keras.callbacks import Callback, ModelCheckpoint
from keras.utils.layer_utils import count_params
from time import time
import numpy as np
import os
import pickle

from utilities.data import load_dataset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

from statistics import *
from logger import *

# possible test: impose memory constraints
# tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=50)])

# Tuning parameters
PREDICT_BATCH_SIZE = 1024   # batch size used for model.predict()
EARLY_STOP_DELTA = 0.001    # currently unused
EARLY_STOP_PATIENCE = 3

LOG_MODEL_SUMMARY = 0		# keras summary of each evaluated model
LOG_MODEL_TRAINING = 0		# training progress: 1 for progress bar, 2 for one line per epoch
LOG_EARLY_STOPPING = False	# log early stopping
LOG_MODEL_SAVE = 1			# log for saving after each epoch


class TimedStopping(keras.callbacks.Callback):
	"""
		Stop training when maximum time has passed.
		Code from:
			https://github.com/keras-team/keras-contrib/issues/87

		Attributes
		----------
		start_time : float
			time when the training started
		seconds : float
			maximum time before stopping.

		Methods
		-------
		on_train_begin(logs)
			method called upon training beginning
		on_epoch_end(epoch, logs={})
			method called after the end of each training epoch
	"""

	def __init__(self, seconds=None):
		"""
		Parameters
		----------
		seconds : float
			maximum time before stopping.
		"""
		super(keras.callbacks.Callback, self).__init__()
		self.start_time = 0
		self.seconds = seconds
		self.timer_stop_triggered = False

	def on_train_begin(self, logs={}):
		"""
			Method called upon training beginning

			Parameters
			----------
			logs : dict
				training logs
		"""
		self.start_time = time()
		self.timer_stop_triggered = False

	def on_epoch_end(self, epoch, logs={}):
		"""
			Method called after the end of each training epoch.
			Checks if the maximum time has passed

			Parameters
			----------
			epoch : int
				current epoch

			logs : dict
				training logs
		"""
		if time() - self.start_time > self.seconds:
			self.model.stop_training = True
			self.timer_stop_triggered = True


class KFoldEvalResult:
	""" result returned by Individual.evaluate_cnn_k_folds() """
	def __init__(self):
		self.final_test_accuracy_list = []
		self.folds_eval_time = []
		self.folds_test_accuracy = []
		self.folds_train_accuracy = []
		self.folds_val_accuracy = []
		self.folds_train_loss = []
		self.folds_val_loss = []
		self.folds_million_inferences_time = []
		self.folds_fitness = []
		self.parameters = 0
		self.accuracy = None
		self.accuracy_std = None
		self.fitness = None
		self.fitness_std = None
		self.final_accuracy = None
		self.final_accuracy_std = None
		self.million_inferences_time = .0
		self.million_inferences_time_std = .0
		self.total_eval_time = .0
		self.n_mutable_variables = 0

	def append_cnn_eval_result(self, result):
		self.folds_eval_time.append(result.eval_time)
		self.final_test_accuracy_list.append(result.final_test_accuracy)
		self.folds_test_accuracy.append(result.accuracy)
		self.folds_train_accuracy.append(result.train_accuracy)
		self.folds_val_accuracy.append(result.val_accuracy)
		self.folds_train_loss.append(result.train_loss)
		self.folds_val_loss.append(result.val_loss)
		self.folds_million_inferences_time.append(result.million_inferences_time)
		self.parameters = result.parameters
		self.total_eval_time += result.eval_time

	def calculate_fitness_mean_std(self, k_fold_eval):
		self.folds_fitness = []
		for acc in self.folds_test_accuracy:
			self.folds_fitness.append(k_fold_eval.fitness_func(acc, self.parameters))
		self.accuracy = np.mean(self.folds_test_accuracy)
		self.accuracy_std = np.std(self.folds_test_accuracy)
		self.fitness = np.mean(self.folds_fitness)
		self.fitness_std = np.std(self.folds_fitness)
		self.final_accuracy = np.mean(self.final_test_accuracy_list)
		self.final_accuracy_std = np.std(self.final_test_accuracy_list)
		self.million_inferences_time = np.mean(self.folds_million_inferences_time)
		self.million_inferences_time_std = np.std(self.folds_million_inferences_time)


def fitness_function_accuracy(accuracy, parameters):
	return accuracy


def calculate_accuracy(y_true, y_pred):
	"""
		Computes the accuracy.

		Parameters
		----------
		y_true : np.array
			array of right labels
		y_pred : np.array
			array of class confidences for each instance

		Returns
		-------
		accuracy : float
			accuracy value
	"""
	y_pred_labels = np.argmax(y_pred, axis=1)
	return accuracy_score(y_true, y_pred_labels)


class Evaluator:
	"""
		Stores the dataset, maps the phenotype into a trainable model, and
		evaluates it

		Attributes
		----------
		dataset : dict
			dataset instances and partitions
		fitness_func : function
			fitness_metric (y_true, y_pred)
			y_pred are the confidences
	"""

	# subclass for evaluation cache entries in evaluation_cache
	class EvaluationCacheEntry:
		"""
			entry in the list of cached evaluations
		"""
		origin_description: str
		metrics: CnnEvalResult
		k_fold_metrics: KFoldEvalResult

		def __init__(self, origin_description, metrics, k_fold_metrics=None):
			self.origin_description = origin_description
			self.metrics = metrics
			self.k_fold_metrics = k_fold_metrics

		def set_k_fold_metrics(self, k_fold_metrics):
			self.k_fold_metrics = k_fold_metrics


	def __init__(self, dataset, fitness_func=fitness_function_accuracy, for_k_fold_validation=False, calculate_fitness_with_k_folds_accuracy=False, evaluation_cache_path='', experiment_name=''):
		"""
			Creates the Evaluator instance and loads the dataset.

			Parameters
			----------
			dataset : str
				dataset to be loaded
			fitness_func : function
				calculates fitness from accuracy and number of trainable weights
		"""
		self.dataset = load_dataset(dataset, for_k_fold_validation)
		self.fitness_func = fitness_func

		self.early_stop_delta = EARLY_STOP_DELTA
		self.early_stop_patience = EARLY_STOP_PATIENCE

		self.experiment_name = experiment_name
		self.calculate_fitness_with_k_folds = calculate_fitness_with_k_folds_accuracy
		self.evaluation_cache_path = evaluation_cache_path
		self.evaluation_cache = {}
		self.evaluation_cache_changed = False

		if self.evaluation_cache_path:
			if  os.path.isfile(self.evaluation_cache_path):
				with open(self.evaluation_cache_path, 'rb') as fh:
					self.evaluation_cache = pickle.load(fh)
				log_bold(f"loaded {len(self.evaluation_cache)} cache entries from {self.evaluation_cache_path}")
			else:
				log_bold(f"will create new evaluation cache {self.evaluation_cache_path}")

	def init_options(self, early_stop_patience, early_stop_delta):
		self.early_stop_patience = early_stop_patience
		self.early_stop_delta = early_stop_delta

	def flush_evaluation_cache(self):
		""" if evaluation cache entries have been added, write all to file """
		if self.evaluation_cache_path and self.evaluation_cache_changed:
			self.evaluation_cache_changed = False
			with open(self.evaluation_cache_path, 'wb') as fh:
				pickle.dump(self.evaluation_cache, fh, protocol=pickle.HIGHEST_PROTOCOL)


	@staticmethod
	def get_keras_layers(phenotype):
		"""
			Parses the phenotype corresponding to the layers.
			Auxiliary function of the assemble_network function.

			Parameters
			----------
			phenotype : str
				individual layers phenotype

			Returns
			-------
			layers : list
				list of tuples (layer_type : str, node properties : dict)
		"""

		raw_phenotype = phenotype.replace('\n', ' ').split(' ')

		idx = 0
		first = True
		node_type, node_val = raw_phenotype[idx].split(':')
		layers = []
		layer_type = None
		node_properties = None

		while idx < len(raw_phenotype):
			if node_type == 'layer':
				if not first:
					layers.append((layer_type, node_properties))
				else:
					first = False
				layer_type = node_val
				node_properties = {}
			else:
				node_properties[node_type] = node_val

			idx += 1
			if idx < len(raw_phenotype):
				node_type, node_val = raw_phenotype[idx].split(':')

		layers.append((layer_type, node_properties))

		return layers

	@staticmethod
	def get_learning(learning):
		"""
			Parses the phenotype corresponding to the learning
			Auxiliary function of the assemble_optimiser function

			Parameters
			----------
			learning : str
				learning phenotype of the individual

			Returns
			-------
			learning_params : dict
				learning parameters
		"""

		raw_learning = learning.split(' ')

		idx = 0
		learning_params = {}
		while idx < len(raw_learning):
			param_name, param_value = raw_learning[idx].split(':')
			learning_params[param_name] = param_value
			idx += 1

		for _key_ in sorted(list(learning_params.keys())):
			if len(learning_params[_key_]) == 1:
				try:
					learning_params[_key_] = eval(learning_params[_key_])
				except NameError:
					learning_params[_key_] = learning_params[_key_]

		return learning_params

	@staticmethod
	def assemble_network(keras_layers, input_size):
		"""
			Maps the layers phenotype into a keras model

			Parameters
			----------
			keras_layers : list
				output from get_layers
			input_size : tuple
				network input shape

			Returns
			-------
			model : keras.models.Model
				keras trainable model
		"""

		# input layer
		inputs = tf.keras.layers.Input(shape=input_size)

		# Create layers -- ADD NEW LAYERS HERE
		layers = []
		for layer_type, layer_params in keras_layers:
			# convolutional layer
			if layer_type == 'conv':
				conv_layer = tf.keras.layers.Conv2D(filters=int(layer_params['num-filters']),
													kernel_size=(int(layer_params['filter-shape']), int(layer_params['filter-shape'])),
													strides=(int(layer_params['stride']), int(layer_params['stride'])),
													padding=layer_params['padding'],
													use_bias=eval(layer_params['bias']),
													# activation=layer_params['act'],
													kernel_initializer='he_normal',
													kernel_regularizer=tf.keras.regularizers.l2(0.0005))
				layers.append(conv_layer)

			# batch-normalisation
			elif layer_type == 'batch-norm':
				# DENSER - check because channels are not first
				batch_norm_layer = tf.keras.layers.BatchNormalization()
				layers.append(batch_norm_layer)

			# average pooling layer; pool-avg is for fd back compatibility
			elif layer_type == 'pool-avg' or (layer_type == 'pooling' and layer_params['pooling-type'] == 'avg'):
				pool_avg = tf.keras.layers.AveragePooling2D(pool_size=(int(layer_params['kernel-size']), int(layer_params['kernel-size'])),
															strides=int(layer_params['stride']),
															padding=layer_params['padding'])
				layers.append(pool_avg)

			# max pooling layer, pool-max is for fd back compatibility
			elif layer_type == 'pool-max' or (layer_type == 'pooling' and layer_params['pooling-type'] == 'max'):
				pool_max = tf.keras.layers.MaxPooling2D(pool_size=(int(layer_params['kernel-size']), int(layer_params['kernel-size'])),
														strides=int(layer_params['stride']),
														padding=layer_params['padding'])
				layers.append(pool_max)

			# fully-connected layer
			elif layer_type == 'fc':
				fc = tf.keras.layers.Dense(int(layer_params['num-units']),
											use_bias=eval(layer_params['bias']),
											# activation=layer_params['act'],
											kernel_initializer='he_normal',
											kernel_regularizer=tf.keras.regularizers.l2(0.0005))
				layers.append(fc)

			elif layer_type == 'output':
				output_layer = tf.keras.layers.Dense(int(layer_params['num-units']),
														use_bias=eval(layer_params['bias']),
														activation='softmax',
														kernel_initializer='he_normal',
														kernel_regularizer=tf.keras.regularizers.l2(0.0005))
				layers.append(output_layer)

			# dropout layer
			elif layer_type == 'dropout':
				dropout = tf.keras.layers.Dropout(rate=min(0.5, float(layer_params['rate'])))
				layers.append(dropout)

			# gru layer DENSERTODO: initializers, recurrent dropout, dropout, unroll, reset_after
			elif layer_type == 'gru':
				gru = tf.keras.layers.GRU(units=int(layer_params['units']),
											activation=layer_params['act'],
											recurrent_activation=layer_params['rec_act'],
											use_bias=eval(layer_params['bias']))
				layers.append(gru)

			# lstm layer DENSERTODO: initializers, recurrent dropout, dropout, unroll, reset_after
			elif layer_type == 'lstm':
				lstm = tf.keras.layers.LSTM(units=int(layer_params['units']),
											activation=layer_params['act'],
											recurrent_activation=layer_params['rec_act'],
											use_bias=eval(layer_params['bias']))
				layers.append(lstm)

			# rnn DENSERTODO: initializers, recurrent dropout, dropout, unroll, reset_after
			elif layer_type == 'rnn':
				rnn = tf.keras.layers.SimpleRNN(units=int(layer_params['units']),
												activation=layer_params['act'],
												use_bias=eval(layer_params['bias']))
				layers.append(rnn)

			elif layer_type == 'conv1d':  # todo initializer
				conv1d = tf.keras.layers.Conv1D(filters=int(layer_params['num-filters']),
												kernel_size=int(layer_params['kernel-size']),
												strides=int(layer_params['stride']),
												padding=layer_params['padding'],
												activation=layer_params['activation'],
												use_bias=eval(layer_params['bias']))
				layers.append(conv1d)
			else:
				raise ValueError(f"Invalid {layer_type=}")

		# END ADD NEW LAYERS

		# Connection between layers
		for layer in keras_layers:
			layer[1]['input'] = list(map(int, layer[1]['input'].split(',')))

		first_fc = True
		data_layers = []
		invalid_layers = []

		for layer_idx, layer in enumerate(layers):
			layer_inputs = keras_layers[layer_idx][1]['input']
			if len(layer_inputs) == 1:
				layer_type = keras_layers[layer_idx][0]
				layer_params = keras_layers[layer_idx][1]
				input_idx = layer_inputs[0]
				# use inputs for first layer, otherwise input
				input_layer = inputs if input_idx == -1 else data_layers[input_idx]
				# add Flatten layer before first fc layer
				if layer_type == 'fc' and first_fc:
					first_fc = False
					flatten = tf.keras.layers.Flatten()(input_layer)
					new_data_layer = layer(flatten)
				else:
					new_data_layer = layer(input_layer)

				# conv and fc layers can have an optional batch normalisation layer, that should be inserted before the activation layer
				if layer_type == 'conv' or layer_type == 'fc':
					if ('batch-norm' in layer_params) and layer_params['batch-norm']:
						new_data_layer = tf.keras.layers.BatchNormalization()(new_data_layer)
					if 'act' in layer_params:
						activation_function = layer_params['act']
						if activation_function == 'relu':
							new_data_layer = tf.keras.layers.ReLU()(new_data_layer)
						elif activation_function == 'elu':
							new_data_layer = tf.keras.layers.ELU()(new_data_layer)
						elif activation_function == 'sigmoid':
							new_data_layer = tf.keras.layers.Activation(tf.keras.activations.sigmoid)(new_data_layer)
						elif activation_function == 'linear':
							pass
						else:
							log_warning(f"Invalid activation value {activation_function}")

				data_layers.append(new_data_layer)

			else:
				# Get minimum shape: when merging layers all the signals are converted to the minimum shape
				minimum_shape = input_size[0]
				for input_idx in layer_inputs:
					if input_idx != -1 and input_idx not in invalid_layers:
						if data_layers[input_idx].shape[-3:][0] < minimum_shape:
							minimum_shape = int(data_layers[input_idx].shape[-3:][0])

				# Reshape signals to the same shape
				merge_signals = []
				for input_idx in layer_inputs:
					if input_idx == -1:
						if inputs.shape[-3:][0] > minimum_shape:
							actual_shape = int(inputs.shape[-3:][0])
							merge_signals.append(tf.keras.layers.MaxPooling2D(pool_size=(actual_shape - (minimum_shape - 1), actual_shape - (minimum_shape - 1)), strides=1)(inputs))
						else:
							merge_signals.append(inputs)

					elif input_idx not in invalid_layers:
						if data_layers[input_idx].shape[-3:][0] > minimum_shape:
							actual_shape = int(data_layers[input_idx].shape[-3:][0])
							merge_signals.append(tf.keras.layers.MaxPooling2D(pool_size=(actual_shape - (minimum_shape - 1), actual_shape - (minimum_shape - 1)), strides=1)(data_layers[input_idx]))
						else:
							merge_signals.append(data_layers[input_idx])

				if len(merge_signals) == 1:
					merged_signal = merge_signals[0]
				elif len(merge_signals) > 1:
					merged_signal = tf.keras.layers.concatenate(merge_signals)
				else:
					merged_signal = data_layers[-1]

				data_layers.append(layer(merged_signal))

		model = tf.keras.models.Model(inputs=inputs, outputs=data_layers[-1])
		return model

	@staticmethod
	def calculate_model_multiplications(model):
		""" print number of multiplications per layer of a keras model"""
		for l in model.layers:
			weights = sum([i.size for i in l.get_weights()])
			outputs = int(np.prod(l.output_shape[1: -1]))
			multiplications = weights * outputs
			print(f'output shape: {l.output_shape[1:]}, weights: {weights}, multiplications: {multiplications}, layer: {l.name}')
			pass

	@staticmethod
	def get_model_summary(model):
		""" returns model summary from keras as a line list """
		stringlist = []
		model.summary(line_length=120, print_fn=lambda x: stringlist.append(x))
		stringlist = [s.rstrip() for s in stringlist if not s.isspace()]
		if stringlist[0].startswith('Model:'):
			del stringlist[0]
		if stringlist[0].startswith('______'):
			del stringlist[0]
		return '\n'.join(stringlist)

	@staticmethod
	def assemble_optimiser(learning):
		"""
			Maps the learning into a keras optimiser

			Parameters
			----------
			learning : dict
				output of get_learning

			Returns
			-------
			optimiser : keras.optimizers.Optimizer
				keras optimiser that will be later used to train the model
		"""

		if learning['learning'] == 'rmsprop':
			return tf.keras.optimizers.RMSprop(learning_rate=float(learning['lr']),
												rho=float(learning['rho']))

		elif learning['learning'] == 'gradient-descent':
			return tf.keras.optimizers.SGD(learning_rate=float(learning['lr']),
											momentum=float(learning['momentum']),
											nesterov=bool(learning['nesterov']))

		elif learning['learning'] == 'adam':
			return tf.keras.optimizers.Adam(learning_rate=float(learning['lr']),
											beta_1=float(learning['beta1']),
											beta_2=float(learning['beta2']))

	def calculate_fitness(self, ind):
		""" calculate fitness of individual """
		accuracy = None
		if self.calculate_fitness_with_k_folds and ind.k_fold_metrics is not None:
			accuracy = ind.k_fold_metrics.accuracy
		elif ind.metrics is not None:
			accuracy = ind.metrics.accuracy

		fitness = self.fitness_func(accuracy, ind.metrics.parameters) if accuracy is not None else None
		return fitness

	@staticmethod
	def cache_key(phenotype, max_training_time, max_training_epochs):
		return f"{phenotype}#{max_training_time}#{max_training_epochs}"

	def evaluate_cnn(self, phenotype, name, max_training_time, max_training_epochs, model_save_path, dataset, stat, datagen=None, datagen_test=None, for_k_folds_validation=False, load_prev_weights=False, input_size=(28, 28, 1)):
		"""
			Evaluates the keras model using the keras optimiser

			Parameters
			----------
			phenotype : str
				individual phenotype
			load_prev_weights : bool
				resume training from a previous train or not
			max_training_time : float
				maximum training time
			max_training_epochs : int
				maximum number of epochs
			model_save_path : str
				path where model and its weights are saved, or empty
			name : str
				id string (<generation>-<number>)
			dataset : dict
				train and test datasets
			datagen : keras.preprocessing.image.ImageDataGenerator
				Data augmentation method image data generator
			datagen_test : keras.preprocessing.image.ImageDataGenerator
				Image data generator without augmentation
			for_k_folds_validation : bool
				suppress caching, logging and early stopping for k-folds validation
			input_size : tuple
				dataset input shape

			Returns
			-------
			result : CnnEvalResult
				contains all result metrics
		"""
		# Mixed precision slows down LeNet training by 50%. Is this because it's too small?
		# tf.keras.mixed_precision.set_global_policy("mixed_float16")

		start_time = time()

		# only allocate as much GPU memory as needed
		gpu_devices = tf.config.experimental.list_physical_devices('GPU')
		tf.config.experimental.set_memory_growth(gpu_devices[0], True)

		model_phenotype, learning_phenotype = phenotype.split('learning:')
		learning_phenotype = 'learning:' + learning_phenotype.rstrip().lstrip()
		model_phenotype = model_phenotype.rstrip().lstrip().replace('  ', ' ')

		keras_layers = self.get_keras_layers(model_phenotype)
		keras_layers_count = len(keras_layers)
		keras_learning = self.get_learning(learning_phenotype)
		batch_size = int(keras_learning['batch_size'])

		if load_prev_weights and os.path.exists(model_save_path):
			model = keras.models.load_model(model_save_path)
			initial_epoch = 10  # !! load weights not implemented
		else:
			initial_epoch = 0
			model = self.assemble_network(keras_layers, input_size)
			opt = self.assemble_optimiser(keras_learning)
			model.compile(optimizer=opt,
							loss='sparse_categorical_crossentropy',
							metrics=['accuracy'])

		model_summary = Evaluator.get_model_summary(model)
		if LOG_MODEL_SUMMARY:
			log(model_summary)

		model_build_time = time() - start_time

		model_layers = len(model.get_config()['layers'])
		parameters = count_params(model.trainable_weights)
		# non_trainable_parameters = count_params(model.non_trainable_weights)

		# time based stopping
		timed_stopping = TimedStopping(seconds=max_training_time)

		callbacks_list = [timed_stopping]

		# early stopping
		#if not for_k_folds_validation:
		#	early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=False, verbose=LOG_EARLY_STOPPING)
		#	# early_stop = keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=self.early_stop_delta, patience=self.early_stop_patience, restore_best_weights=False, verbose=LOG_EARLY_STOPPING)
		#	callbacks_list.append(early_stop)

		training_start_time = time()

		x_train = dataset['evo_x_train']
		y_train = dataset['evo_y_train']
		x_val = dataset['evo_x_val']
		y_val = dataset['evo_y_val']
		if datagen is not None:
			score = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
										steps_per_epoch=(x_train.shape[0] // batch_size),
										epochs=max_training_epochs,
										validation_data=(datagen_test.flow(x_val, y_val, batch_size=batch_size)),
										validation_steps=(x_val.shape[0] // batch_size),
										callbacks=callbacks_list,
										initial_epoch=initial_epoch,
										verbose=LOG_MODEL_TRAINING)
		else:
			score = model.fit(x=x_train,
								y=y_train,
								batch_size=batch_size,
								epochs=max_training_epochs,
								steps_per_epoch=(x_train.shape[0] // batch_size),
								validation_data=(x_val, y_val),
								callbacks=callbacks_list,
								initial_epoch=initial_epoch,
								verbose=LOG_MODEL_TRAINING)

		training_time = time() - training_start_time
		training_epochs = len(score.epoch)
		timer_stop_triggered = timed_stopping.timer_stop_triggered
		early_stop_triggered = training_epochs < max_training_epochs and not timer_stop_triggered

		# save final model to file
		if model_save_path:
			model.save(model_save_path)

		# measure test performance
		x_test = dataset['evo_x_test']
		y_test = dataset['evo_y_test']
		test_accuracy, model_test_time, million_inferences_time  = self.test_model_with_data(model, x_test, y_test, datagen_test)

		fitness = self.fitness_func(test_accuracy, parameters)

		# measure final accuracy
		x_final_test = dataset['x_final_test']
		y_final_test = dataset['y_final_test']
		final_test_accuracy, final_test_time, million_inferences_time  = self.test_model_with_data(model, x_final_test, y_final_test, datagen_test)

		keras.backend.clear_session()

		eval_time = time() - start_time

		result = CnnEvalResult(score.history, final_test_accuracy, eval_time, training_time, final_test_time, million_inferences_time, timer_stop_triggered, early_stop_triggered, parameters, keras_layers_count, model_layers, test_accuracy, fitness, model_summary)

		stat.record_evaluation(seconds=eval_time, is_k_folds=for_k_folds_validation)

		return result


	def evaluate_cnn_with_cache(self, phenotype, name, max_training_time, max_training_epochs, model_save_path, dataset, stat, datagen=None, datagen_test=None, for_k_folds_validation=False, load_prev_weights=False, input_size=(28, 28, 1)):
		"""
			Evaluates the keras model using the keras optimiser, using caching
		"""

		cache_key = Evaluator.cache_key(phenotype, max_training_time, max_training_epochs)
		if self.evaluation_cache_path and cache_key in self.evaluation_cache:
			cache_entry = self.evaluation_cache[cache_key]
			log_debug('using cached metrics from ' + cache_entry.origin_description)
			stat.record_evaluation(seconds=cache_entry.metrics.eval_time, is_cache_hit=True)
			return cache_entry.metrics

		result = self.evaluate_cnn(phenotype, name, max_training_time, max_training_epochs, model_save_path, dataset, stat, datagen, datagen_test, for_k_folds_validation, load_prev_weights, input_size)

		if self.evaluation_cache_path:
			new_cache_entry = Evaluator.EvaluationCacheEntry(f"{self.experiment_name}:{name}", result)
			self.evaluation_cache[cache_key] = new_cache_entry
			self.evaluation_cache_changed = True

		return result

	def final_test_saved_model(self, model_path, datagen_test=None):
		"""
			Compute final testing performance of the model

			Parameters
			----------
			model_path : str
				Path to the model .h5 file

			Returns
			-------
			accuracy : float
				Model accuracy
		"""

		model = keras.models.load_model(model_path)
		accuracy, model_test_time, million_inferences_time = Evaluator.test_model_with_data(model, self.dataset['x_final_test'], self.dataset['y_final_test'], datagen_test)

		return accuracy



	def evaluate_cnn_k_folds(self, phenotype, name, metrics, n_folds, max_training_time, max_training_epochs, stat, datagen=None, datagen_test=None, input_size=(28, 28, 1)):
		""" evaluate individual for k-folds, using cache """

		cache_key = Evaluator.cache_key(phenotype, max_training_time, max_training_epochs)
		if self.evaluation_cache_path and cache_key in self.evaluation_cache:
			cache_entry = self.evaluation_cache[cache_key]
			if cache_entry.k_fold_metrics is not None:
				log_debug('k-fold cached metrics from ' + cache_entry.origin_description)
				stat.record_evaluation(seconds=cache_entry.k_fold_metrics.total_eval_time, is_cache_hit=True, is_k_folds=True)
				return cache_entry.k_fold_metrics
			else:
				log_debug('*** k-fold metrics not in cache entry ' + cache_entry.origin_description)

		x_combined = self.dataset['x_combined']
		y_combined = self.dataset['y_combined']

		split = StratifiedShuffleSplit(n_splits=n_folds, test_size=7000)

		k_folds_result = KFoldEvalResult()

		fold_number = 1
		for train_index, test_index in split.split(x_combined, y_combined):
			evo_x_train, x_val = x_combined[train_index], x_combined[test_index]
			evo_y_train, y_val = y_combined[train_index], y_combined[test_index]

			val_test_split = StratifiedShuffleSplit(n_splits=1, test_size=3500)
			for train_index2, test_index2 in val_test_split.split(x_val, y_val):
				evo_x_val, evo_x_test = x_val[train_index2], x_val[test_index2]
				evo_y_val, evo_y_test = y_val[train_index2], y_val[test_index2]

			fold_dataset = {
				'evo_x_train': evo_x_train, 'evo_y_train': evo_y_train,
				'evo_x_val': evo_x_val, 'evo_y_val': evo_y_val,
				'evo_x_test': evo_x_test, 'evo_y_test': evo_y_test,
				'x_final_test': self.dataset['x_final_test'], 'y_final_test': self.dataset['y_final_test']
			}
			result = self.evaluate_cnn(phenotype, f"Fold #{fold_number}", max_training_time, max_training_epochs, '', fold_dataset, stat, datagen, datagen_test, for_k_folds_validation=True, input_size=input_size)
			fold_number += 1
			k_folds_result.append_cnn_eval_result(result)

		if self.evaluation_cache_path and metrics is not None:
			new_cache_entry = Evaluator.EvaluationCacheEntry(f"{self.experiment_name}:{name}", metrics, k_folds_result)
			self.evaluation_cache[cache_key] = new_cache_entry
			self.evaluation_cache_changed = True

		return k_folds_result

	@staticmethod
	def test_model_with_data(model, x_test, y_test, datagen_test):
		model_test_start_time = time()
		if datagen_test is None:
			y_pred = model.predict(x_test, batch_size=PREDICT_BATCH_SIZE, verbose=0)
		else:
			y_pred = model.predict_generator(datagen_test.flow(x_test, batch_size=PREDICT_BATCH_SIZE, shuffle=False), steps=x_test.shape[0] // PREDICT_BATCH_SIZE, verbose=LOG_MODEL_TRAINING)
		accuracy = calculate_accuracy(y_test, y_pred)
		model_test_time = time() - model_test_start_time
		million_inferences_time = 1000000.0 * model_test_time / len(y_pred)
		return accuracy, model_test_time, million_inferences_time



class Individual:
	"""
		Candidate solution.

		Attributes
		----------
		network_structure : list
			ordered list of tuples formatted as follows
			[(non-terminal, min_expansions, max_expansions), ...]
		output_rule : str
			output non-terminal symbol
		macro_symbols : list
			list of non-terminals (str) with the marco rules (e.g., learning)
		modules : list
			list of Modules (genotype) of the layers
		output : dict
			output rule genotype
		phenotype : str
			phenotype of the candidate solution
		fitness : float
			fitness value of the candidate solution
		training_epochs : int
			number of performed epochs during training
		parameters : int
			number of trainable parameters of the network
		id : str
			string <generation>-<index>

		Methods
		-------
			initialise_individual(grammar, reuse)
				Randomly creates a candidate solution
			decode(grammar)
				Maps the genotype to the phenotype
			evaluate_individual()
				Performs the evaluation of a candidate solution
	"""
	is_parent: bool
	final_test_accuracy: None	# accuracy with final test set
	fitness: None				# calculated fitness
	metrics: CnnEvalResult
	k_fold_metrics: KFoldEvalResult
	training_complete: bool
	model_save_path: str

	def __init__(self, network_structure, macro_symbols, output_rule, generation, idx):
		"""
			Parameters
			----------
			network_structure : list
				ordered list of tuples formatted as follows
				[(non-terminal, min_expansions, max_expansions), ...]
			macro_symbols : list
				list of non-terminals (str) with the learning_rule
			output_rule : str
				output non-terminal symbol
			generation : int
				generation count
			idx: int
				index in generation
		"""

		self.macro_layer_step = None
		self.network_structure = network_structure
		self.output_rule = output_rule
		self.macro_symbols = macro_symbols
		self.modules = []
		self.modules_including_macro = []   # contains references to modules + macro_module
		self.macro_module = None
		self.output = None
		self.phenotype_lines = []
		self.generation = generation
		self.id = f"{generation}-{idx}"
		self.parent_id = None
		self.evolution_history = []
		self.reset_training()

	def reset_training(self):
		"""reset all values computed during training"""
		self.is_parent = False
		self.fitness = 0.0
		self.metrics = None
		self.k_fold_metrics = None
		self.training_complete = False
		self.model_save_path = None

	def __repr__(self):
		return self.short_description()

	def short_description(self):
		""" return short description of individual with fitness, k-fold accuracy and final test accuracy if calculated,
			test accuracy and number of parameters """
		result = f"{self.id} "
		if self.fitness:
			result += f"{self.fitness:.5f} "
		if self.k_fold_metrics:
			result += f"k-folds: {self.k_fold_metrics.accuracy:.5f} (SD:{self.k_fold_metrics.accuracy_std:.5f}) "
		if self.metrics:
			if self.metrics.final_test_accuracy:
				result += f"final: {self.metrics.final_test_accuracy:.5f} "
			result += f"acc: {self.metrics.accuracy:.5f} p: {self.metrics.parameters}"
		return result

	def log_long_description(self, title):
		""" output long description to log, with phenotype, model summary and evolution history """
		log('\n----------------------------------------------------------------------------------------------------------------------------------------')
		log_bold(f"{title}: {self.short_description()}\nPhenotype:")
		log('\n'.join(self.phenotype_lines) + '\n')
		log(self.metrics.model_summary)
		log_bold('\nEvolution history:')
		log('\n'.join(self.evolution_history))
		log('\n----------------------------------------------------------------------------------------------------------------------------------------')

	def json_statistics(self):
		""" return dictionary of statistics for individual to write to json file"""
		result = {
			'id': self.id,
			'is_parent' : self.is_parent,
			'final_test_accuracy': self.metrics.final_test_accuracy,
			'accuracy': self.metrics.accuracy,
			'fitness': self.fitness,
			'parameters': self.metrics.parameters,
			'training_epochs': self.metrics.training_epochs,
			'training_time': self.metrics.training_time,
			'history_train_accuracy': self.metrics.history_train_accuracy,
			'history_train_loss': self.metrics.history_train_loss,
			'history_val_accuracy': self.metrics.history_val_accuracy,
			'history_val_loss': self.metrics.history_val_loss,
			'training_complete': self.training_complete,
			'phenotype': self.phenotype_lines,
			'model_summary': self.metrics.model_summary,
			'evolution_history': self.evolution_history,
			'layers_features': self.modules[0].layers.__repr__(),
			'layers_classifier': self.modules[1].layers.__repr__(),
			'layers_learning': self.macro_module.layers.__repr__(),
		}
		if self.k_fold_metrics:
			result.update(
				{
					'k_fold_accuracy': self.k_fold_metrics.accuracy,
					'k_fold_accuracy_std': self.k_fold_metrics.accuracy_std,
					'k_fold_final_accuracy': self.k_fold_metrics.final_accuracy,
					'k_fold_final_accuracy_std': self.k_fold_metrics.final_accuracy_std,
					'k_fold_fitness_std': self.k_fold_metrics.fitness_std,
					'k_fold_million_inferences_time': self.k_fold_metrics.million_inferences_time,
					'k_fold_million_inferences_time_std': self.k_fold_metrics.million_inferences_time_std,
				}
			)
		return result

	def initialise_random(self, grammar, init_max):
		"""Randomly creates a candidate solution

			Parameters
			----------
			grammar : Grammar
				grammar instances that stores the expansion rules

			Returns
			-------
			candidate_solution : Individual
				randomly created candidate solution
		"""

		for non_terminal, min_expansions, max_expansions in self.network_structure:
			new_module = grammar.Module(non_terminal, min_expansions, max_expansions)
			new_module.initialise_module_random(grammar, init_max)

			self.modules.append(new_module)

		# Initialise output
		self.output = grammar.initialise_layer_random(self.output_rule)

		# Initialise the macro structure: learning, data augmentation, etc.
		self.macro_module = grammar.Module("learning", 1, 1)
		for rule in self.macro_symbols:
			self.macro_module.layers.append(grammar.initialise_layer_random(rule))
		self.modules_including_macro = self.modules + [self.macro_module]
		log_bold(f"randomly created individual {self.id}:")
		log(self.get_phenotype(grammar))
		return self

	def initialise_as_lenet(self, grammar):
		"""
			Create a pre-set Lenet Individual

			Parameters
			----------
			grammar : Grammar
				grammar instance that stores the expansion rules

			Returns
			-------
			candidate_solution : Individual
				randomly created candidate solution
		"""

		new_module = grammar.Module('features', 0, 10)
		new_module.initialise_module_as_lenet()
		self.modules.append(new_module)

		new_module = grammar.Module('classification', 1, 5)
		new_module.initialise_module_as_lenet()
		self.modules.append(new_module)

		# Initialise output
		self.output = grammar.initialise_layer_random(self.output_rule)

		# Initialise the macro structure: learning, data augmentation, etc.
		self.macro_module = grammar.Module("learning", 1, 1)
		self.macro_module.layers = layers = grammar.Module.default_learning_rule_adam()
		self.modules_including_macro = self.modules + [self.macro_module]
		return self

	def initialise_as_perceptron(self, grammar):
		"""
			Create a pre-set Perceptron Individual

			Parameters
			----------
			grammar : Grammar
				grammar instance that stores the expansion rules

			Returns
			-------
			candidate_solution : Individual
				randomly created candidate solution
		"""

		new_module = grammar.Module('features', 0, 10)
		new_module.initialise_module_as_perceptron()
		self.modules.append(new_module)

		new_module = grammar.Module('classification', 1, 5)
		new_module.initialise_module_as_perceptron()
		self.modules.append(new_module)

		# Initialise output
		self.output = grammar.initialise_layer_random(self.output_rule)

		# Initialise the macro structure: learning, data augmentation, etc.
		self.macro_module = grammar.Module("learning", 1, 1)
		self.macro_module.layers = layers = grammar.Module.default_learning_rule_adam()
		self.modules_including_macro = self.modules + [self.macro_module]
		return self

	def log_mutation_summary(self, description):
		""" log a mutation text"""
		self.evolution_history.append(description)

	def log_mutation(self, description):
		"""log a mutation. Prefixes with parent's id. """
		self.evolution_history.append(f"{self.parent_id}: {description}")
		log_mutation(f"mutate {self.parent_id}: {description}")

	def log_mutation_add_to_line(self, description):
		"""log a mutation, appending to last line"""
		if len(self.evolution_history) == 0:
			self.evolution_history.append("")
		self.evolution_history[-1] += "    " + description
		log_mutation(f"    {description}")

	def get_phenotype(self, grammar):
		"""
			Maps the genotype to the phenotype

			Parameters
			----------
			grammar : Grammar
				grammar instances that stores the expansion rules

			Returns
			-------
			phenotype : str
				phenotype of the individual to be used in the mapping to the keras model.
		"""

		phenotype = ''
		layer_counter = 0
		for module in self.modules:
			for layer_idx, layer in enumerate(module.layers):
				phenotype += '\n' + grammar.decode_layer(module.module_name, layer) + ' input:' + str(layer_counter - 1)
				layer_counter += 1

		phenotype += '\n' + grammar.decode_layer(self.output_rule, self.output) + ' input:' + str(layer_counter - 1)

		for rule_idx, learning_rule in enumerate(self.macro_symbols):
			phenotype += '\n' + grammar.decode_layer(learning_rule, self.macro_module.layers[rule_idx])

		phenotype = phenotype.lstrip('\n')
		self.phenotype_lines = phenotype.split('\n')
		return phenotype

	def evaluate_individual(self, grammar, cnn_eval, stat, datagen, datagen_test, save_path, max_training_time, max_training_epochs):
		"""
			Performs the evaluation of a candidate solution

			Parameters
			----------
			grammar : Grammar
				grammar instance that stores the expansion rules
			cnn_eval : Evaluator
				Evaluator instance used to train the networks
			datagen : keras.preprocessing.image.ImageDataGenerator
				Data augmentation method image data generator
			datagen_test : keras.preprocessing.image.ImageDataGenerator
				Image data generator without augmentation
			save_path : str
				path where statistics and weights are saved

			Returns
			-------
			fitness : float
		"""

		if not self.training_complete:
			random_state = random.getstate()
			numpy_state = np.random.get_state()

			phenotype = self.get_phenotype(grammar)

			model_save_path = save_path + 'individual-' + self.id + '.h5' if save_path else None

			metrics = None
			log_training_nolf(f"{self.id} layers: {len(self.modules_including_macro[0].layers)}+{len(self.modules_including_macro[1].layers)} ")

			try:
				metrics = cnn_eval.evaluate_cnn_with_cache(phenotype, self.id, max_training_time, max_training_epochs, model_save_path, cnn_eval.dataset, stat, datagen, datagen_test)
			except tf.errors.ResourceExhaustedError as e:
				log_warning(f"{self.id} : ResourceExhaustedError {e}")
				keras.backend.clear_session()
				stat.record_evaluation(is_invalid=True)
			except (TypeError, ValueError) as e:
				log_warning(f"{self.id} : caught exception {e}")
				keras.backend.clear_session()
				stat.record_evaluation(is_invalid=True)

			if metrics is not None:
				self.metrics = metrics
				self.model_save_path = model_save_path
				self.fitness = cnn_eval.calculate_fitness(self)
				log_training(self.metrics.summary())
				if not self.metrics.training_epochs:
					log_warning(f"*** {self.id}: no training epoch completed ***")
			else:
				self.fitness = None

			self.training_complete = True

			random.setstate(random_state)
			np.random.set_state(numpy_state)

		return self.fitness

	def evaluate_individual_k_folds(self, grammar, cnn_eval, nfolds, stat, datagen, datagen_test, max_training_time, max_training_epochs):
		"""
			Performs the evaluation of a candidate solution

			Parameters
			----------
			grammar : Grammar
				grammar instance that stores the expansion rules
			cnn_eval : Evaluator
				Evaluator instance used to train the networks
			nfolds : int
				number of folds
			datagen : keras.preprocessing.image.ImageDataGenerator
				Data augmentation method image data generator
			datagen_test : keras.preprocessing.image.ImageDataGenerator
				Image data generator without augmentation
			max_training_time : int
				Maximum training time per model
			max_training_epochs : int
				Maximal epochs to train

			Returns
			-------
			fitness : float
		"""

		random_state = random.getstate()
		numpy_state = np.random.get_state()

		phenotype = self.get_phenotype(grammar)

		try:
			self.k_fold_metrics = cnn_eval.evaluate_cnn_k_folds(phenotype, self.id, self.metrics, nfolds, max_training_time, max_training_epochs, stat, datagen, datagen_test)
			self.k_fold_metrics.calculate_fitness_mean_std(cnn_eval)
			old_fitness = self.fitness
			self.fitness = cnn_eval.calculate_fitness(self)
			log_bold(f"--> {self.id} with {nfolds} folds: acc: {self.metrics.accuracy:0.5f} -> {self.k_fold_metrics.accuracy:0.5f} (SD:{self.k_fold_metrics.accuracy_std:0.5f}), final acc: {self.k_fold_metrics.final_accuracy:0.5f} (SD:{self.k_fold_metrics.final_accuracy_std:0.5f}), fitness: {old_fitness:0.5f} -> {self.fitness:0.5f}")
		except tf.errors.ResourceExhaustedError as e:
			log_warning(f"{self.id} k-folds evaluation: ResourceExhaustedError {e}")
			keras.backend.clear_session()
		except (TypeError, ValueError) as e:
			log_warning(f"{self.id} k-folds evaluation: caught exception {e}")
			keras.backend.clear_session()

		random.setstate(random_state)
		np.random.set_state(numpy_state)

	def record_statistics(self, ind_stats: RunStatistics.IndividualStatistics):
		ind_stats.id.append(self.id)
		if self.metrics is not None:
			ind_stats.final_test_accuracy.append(self.metrics.final_test_accuracy)
			ind_stats.accuracy.append(self.metrics.accuracy)
			ind_stats.parameters.append(self.metrics.parameters)
			ind_stats.evaluation_time.append(self.metrics.eval_time)
			ind_stats.training_time.append(self.metrics.training_time)
			ind_stats.training_epochs.append(self.metrics.training_epochs)
			ind_stats.fitness.append(self.fitness)
			if len(self.metrics.history_train_accuracy):
				ind_stats.train_accuracy.append(self.metrics.history_train_accuracy[-1])
				ind_stats.train_loss.append(self.metrics.history_train_loss[-1])
				ind_stats.val_accuracy.append(self.metrics.history_val_accuracy[-1])
				ind_stats.val_loss.append(self.metrics.history_val_loss[-1])
		if self.k_fold_metrics is not None:
			ind_stats.k_fold_accuracy.append(self.k_fold_metrics.accuracy)
			ind_stats.k_fold_accuracy_std.append(self.k_fold_metrics.accuracy_std)
			ind_stats.k_fold_final_accuracy.append(self.k_fold_metrics.final_accuracy)
			ind_stats.k_fold_final_accuracy_std.append(self.k_fold_metrics.final_accuracy_std)
			ind_stats.k_fold_fitness.append(self.k_fold_metrics.fitness)
			ind_stats.k_fold_fitness_std.append(self.k_fold_metrics.fitness_std)
			ind_stats.k_fold_million_inferences_time.append(self.k_fold_metrics.million_inferences_time)
			ind_stats.k_fold_million_inferences_time_std.append(self.k_fold_metrics.million_inferences_time_std)

	def record_stepwidth_statistics(self, stepwidth_stats):
		stepwidth_stats.clear()
		for module in self.modules_including_macro:
			if len(module.step_history):
				stepwidth_stats.append((module.module_name, 'structure', module.step_history))
