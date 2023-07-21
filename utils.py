import random
import tensorflow as tf
import keras
from keras import backend
from keras.callbacks import Callback, ModelCheckpoint
from keras.utils.layer_utils import count_params
from time import time
import numpy as np
import os
from utilities.data import load_dataset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

from grammar import Module, default_learning_rule_adam
from logger import *

# TODO: future -- impose memory constraints
# tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=50)])

# Tuning parameters
PREDICT_BATCH_SIZE = 1024  # batch size used for model.predict()
EARLY_STOP_DELTA = 0.001
EARLY_STOP_PATIENCE = 3

DEBUG = True
LOG_MODEL_SUMMARY = False  # keras summary of each evaluated model
LOG_MODEL_TRAINING = 0  # training progress: 1 for progress bar, 2 for one line per epoch
LOG_MODEL_SAVE = True  # log for saving after each epoch
LOG_EARLY_STOPPING = True  # log early stopping
SAVE_MODEL_AFTER_EACH_EPOCH = False  # monitor and save model after each epoch


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


def fitness_function_accuracy(accuracy, trainable_parameters):
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

		Methods
		-------
		get_layers(phenotype)
			parses the phenotype corresponding to the layers
			auxiliary function of the assemble_network function
		get_learning(learning)
			parses the phenotype corresponding to the learning
			auxiliary function of the assemble_optimiser function
		assemble_network(keras_layers, input_size)
			maps the layers phenotype into a keras model
		assemble_optimiser(learning)
			maps the learning into a keras optimiser
		evaluate_cnn(phenotype, load_prev_weights, weights_save_path, parent_weights_path,
							train_time, training_epochs, datagen=None, input_size=(32, 32, 3))
			evaluates the keras model using the keras optimiser
		testing_performance(self, model_path)
			compute testing performance of the model
	"""

	def __init__(self, dataset, for_k_fold_validation=False, fitness_func=fitness_function_accuracy):
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

	def init_options(self, early_stop_patience, early_stop_delta):
		self.early_stop_patience = early_stop_patience
		self.early_stop_delta = early_stop_delta

	@staticmethod
	def get_layers(phenotype):
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
													#                                                   activation=layer_params['act'],
													kernel_initializer='he_normal',
													kernel_regularizer=tf.keras.regularizers.l2(0.0005))
				layers.append(conv_layer)

			# batch-normalisation
			elif layer_type == 'batch-norm':
				# TODO - check because channels are not first
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

			# gru layer #TODO: initializers, recurrent dropout, dropout, unroll, reset_after
			elif layer_type == 'gru':
				gru = tf.keras.layers.GRU(units=int(layer_params['units']),
										  activation=layer_params['act'],
										  recurrent_activation=layer_params['rec_act'],
										  use_bias=eval(layer_params['bias']))
				layers.append(gru)

			# lstm layer #TODO: initializers, recurrent dropout, dropout, unroll, reset_after
			elif layer_type == 'lstm':
				lstm = tf.keras.layers.LSTM(units=int(layer_params['units']),
											activation=layer_params['act'],
											recurrent_activation=layer_params['rec_act'],
											use_bias=eval(layer_params['bias']))
				layers.append(lstm)

			# rnn #TODO: initializers, recurrent dropout, dropout, unroll, reset_after
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
					if ('batch-normalization' in layer_params) and layer_params['batch-normalization']:
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

		if LOG_MODEL_SUMMARY:
			model.summary(line_length=120)

		return model

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

	def evaluate_cnn_k_folds(self, phenotype, n_folds, max_training_time, max_training_epochs, datagen=None, datagen_test=None, input_size=(28, 28, 1)):  # pragma: no cover
		random_state = random.getstate()
		numpy_state = np.random.get_state()

		x_combined = self.dataset['x_combined']
		y_combined = self.dataset['y_combined']

		# log(f"evaluating {id} with {n_folds} folds:")
		split = StratifiedShuffleSplit(n_splits=n_folds, test_size=17000)
		test_accuracy_list = []
		accuracy_list = []
		val_accuracy_list = []
		loss_list = []
		fitness_list = []

		fold_number = 1
		for train_index, test_index in split.split(x_combined, y_combined):
			evo_x_train, x_val = x_combined[train_index], x_combined[test_index]
			evo_y_train, y_val = y_combined[train_index], y_combined[test_index]

			val_test_split = StratifiedShuffleSplit(n_splits=1, test_size=17000 - 3500)
			for train_index2, test_index2 in val_test_split.split(x_val, y_val):
				evo_x_val, evo_x_test = x_val[train_index2], x_val[test_index2]
				evo_y_val, evo_y_test = y_val[train_index2], y_val[test_index2]

			fold_dataset = {
				'evo_x_train': evo_x_train, 'evo_y_train': evo_y_train,
				'evo_x_val': evo_x_val, 'evo_y_val': evo_y_val,
				'evo_x_test': evo_x_test, 'evo_y_test': evo_y_test,
			}
			metrics = self.evaluate_cnn(phenotype, max_training_time, max_training_epochs, '', f"Fold #{fold_number}", fold_dataset, datagen, datagen_test, suppress_logging=True, suppress_early_stopping=True, input_size=input_size)
			fold_number += 1
			test_accuracy = metrics['test_accuracy']
			accuracy = metrics['accuracy'][-1]
			val_accuracy = metrics['val_accuracy'][-1]
			loss = metrics['loss'][-1]
			fitness = metrics['fitness']
			test_accuracy_list.append(test_accuracy)
			accuracy_list.append(accuracy)
			val_accuracy_list.append(val_accuracy)
			loss_list.append(loss)
			fitness_list.append(fitness)

		k_fold_metrics = {
			'test_accuracy_list': test_accuracy_list,
			'accuracy_list': accuracy_list,
			'val_accuracy_list': val_accuracy_list,
			'loss_list': loss_list,
			'fitness_list': fitness_list,
		}
		random.setstate(random_state)
		np.random.set_state(numpy_state)
		return k_fold_metrics

	def evaluate_cnn(self, phenotype, max_training_time, max_training_epochs, model_save_path, id, dataset, datagen=None, datagen_test=None, suppress_logging=False, suppress_early_stopping=False, load_prev_weights=False, input_size=(28, 28, 1)):
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
				path where model and its weights are saved
			id : str
				id string (<generation>-<number>)
			dataset : dict
				train and test datasets
			datagen : keras.preprocessing.image.ImageDataGenerator
				Data augmentation method image data generator
			datagen_test : keras.preprocessing.image.ImageDataGenerator
				Image data generator without augmentation
			input_size : tuple
				dataset input shape

			Returns
			-------
			score_history : dict
				training data: loss and accuracy
		"""

		# Mixed precision slows down LeNet training by 50%. Is this because it's too small?
		# tf.keras.mixed_precision.set_global_policy("mixed_float16")

		# only allocate as much GPU memory as needed
		gpu_devices = tf.config.experimental.list_physical_devices('GPU')
		tf.config.experimental.set_memory_growth(gpu_devices[0], True)

		model_build_start_time = time()
		model_phenotype, learning_phenotype = phenotype.split('learning:')
		learning_phenotype = 'learning:' + learning_phenotype.rstrip().lstrip()
		model_phenotype = model_phenotype.rstrip().lstrip().replace('  ', ' ')

		keras_layers = self.get_layers(model_phenotype)
		keras_learning = self.get_learning(learning_phenotype)
		batch_size = int(keras_learning['batch_size'])

		if load_prev_weights and os.path.exists(model_save_path):
			model = keras.models.load_model(model_save_path)
			initial_epoch = 10  # TODO
		else:
			initial_epoch = 0
			model = self.assemble_network(keras_layers, input_size)
			opt = self.assemble_optimiser(keras_learning)
			model.compile(optimizer=opt,
							loss='sparse_categorical_crossentropy',
							metrics=['accuracy'])
		model_build_time = time() - model_build_start_time

		model_layers = len(model.get_config()['layers'])
		trainable_parameters = count_params(model.trainable_weights)
		non_trainable_parameters = count_params(model.non_trainable_weights)

		# time based stopping
		time_stop = TimedStopping(seconds=max_training_time)

		# early stopping
		early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=LOG_EARLY_STOPPING)

		callbacks_list = [early_stop, time_stop]

		# new early stopping
		#if not suppress_early_stopping:
		#	early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=self.early_stop_delta, patience=self.early_stop_patience, restore_best_weights=False, verbose=LOG_EARLY_STOPPING)
		#	callbacks_list.append(early_stop)

		if not suppress_logging:
			log_training_nolf(f"{id} layers:{len(keras_layers):2d}/{model_layers:2d} params:{trainable_parameters:6d}/{non_trainable_parameters}")

		training_start_time = time()
		# save individual with the lowest validation loss - useful for when training is halted because of time
		if SAVE_MODEL_AFTER_EACH_EPOCH:
			monitor = ModelCheckpoint(model_save_path, monitor='val_loss', verbose=LOG_MODEL_SAVE, save_best_only=True)
			callbacks_list.append(monitor)

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
		timer_stop_triggered = time_stop.timer_stop_triggered
		early_stop_triggered = training_epochs < max_training_epochs and not timer_stop_triggered

		# save final model to file
		if model_save_path:
			model.save(model_save_path)

		# measure test performance
		model_test_start_time = time()
		x_test = dataset['evo_x_test']
		y_test = dataset['evo_y_test']
		if datagen_test is None:
			y_pred_test = model.predict(x_test, batch_size=PREDICT_BATCH_SIZE, verbose=0)
		else:
			y_pred_test = model.predict_generator(datagen_test.flow(x_test, batch_size=PREDICT_BATCH_SIZE, shuffle=False), steps=self.dataset['evo_x_test'].shape[0] // 100, verbose=LOG_MODEL_TRAINING)
		test_accuracy = calculate_accuracy(y_test, y_pred_test)
		model_test_time = time() - model_test_start_time
		million_inferences_time = 1000000.0 * model_test_time / len(y_pred_test)
		fitness = self.fitness_func(test_accuracy, trainable_parameters)

		history = score.history
		if training_epochs:
			accuracy = history['accuracy'][-1]
			val_accuracy = history['val_accuracy'][-1]
			loss = history['loss'][-1]
			if not suppress_logging:
				log_training(
					f" ep:{training_epochs:2d} inf: {million_inferences_time:0.2f} test: {test_accuracy:0.5f}, acc: {accuracy:0.5f} val: {val_accuracy:0.5f} loss: {loss:0.5f} t: {training_time:0.2f}s (b{model_build_time:0.2f},t{model_test_time:0.2f}) fitness: {fitness} {'T' if timer_stop_triggered else ''}{'E' if early_stop_triggered else ''}")
		else:
			log_warning(f" *** no training epoch completed ***")

		# return values
		history['training_epochs'] = training_epochs
		history['training_time'] = training_time
		history['million_inferences_time'] = million_inferences_time
		history['trainable_parameters'] = trainable_parameters
		history['test_accuracy'] = test_accuracy
		history['fitness'] = fitness
		history['time_stop'] = timer_stop_triggered
		history['early_stop'] = early_stop_triggered

		keras.backend.clear_session()

		return history

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

		random_state = random.getstate()
		numpy_state = np.random.get_state()

		model = keras.models.load_model(model_path)
		accuracy = test_model_with_dataset(model, self.dataset['x_test'], self.dataset['y_test'], datagen_test)

		random.setstate(random_state)
		np.random.set_state(numpy_state)
		return accuracy


def test_model_with_dataset(model, x_test, y_test, datagen_test=None):
	model_test_start_time = time()
	if datagen_test is None:
		y_pred = model.predict(x_test, verbose=LOG_MODEL_TRAINING)
	else:
		y_pred = model.predict_generator(datagen_test.flow(x_test, y_test, shuffle=False), verbose=LOG_MODEL_TRAINING)
	accuracy = calculate_accuracy(y_test, y_pred)
	final_test_time = time() - model_test_start_time
	final_million_inferences_time = 1000000.0 * final_test_time / len(y_pred)
	return accuracy




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
		learning_rule : list
			list of non-terminals (str) with the marco rules (e.g., learning)
		modules : list
			list of Modules (genotype) of the layers
		output : dict
			output rule genotype
		macro : list
			list of Modules (genotype) for the macro rules
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
			initialise_individual(grammar, levels_back, reuse)
				Randomly creates a candidate solution
			decode(grammar)
				Maps the genotype to the phenotype
			evaluate_individual()
				Performs the evaluation of a candidate solution
	"""

	def __init__(self, network_structure, learning_rule, output_rule, gen, idx):
		"""
			Parameters
			----------
			network_structure : list
				ordered list of tuples formatted as follows
				[(non-terminal, min_expansions, max_expansions), ...]
			learning_rule : list
				list of non-terminals (str) with the learning_rule
			output_rule : str
				output non-terminal symbol
			gen : int
				generation count
			idx: int
				index in generation
		"""

		self.network_structure = network_structure
		self.output_rule = output_rule
		self.learning_rule = learning_rule
		self.modules = []
		self.output = None
		self.macro = []
		self.phenotype = []
		self.id = f"{gen}-{idx}"
		self.parent = None
		self.evolution_history = []
		self.reset_training()

	def reset_training(self):
		"""reset all values computed during training"""
		self.final_test_accuracy = None
		self.test_accuracy = None
		self.parameters = None
		self.training_time = 0
		self.training_epochs = 0
		self.million_inferences_time = None
		self.fitness = None
		self.training_complete = False
		self.metrics_accuracy = None
		self.metrics_loss = None
		self.metrics_val_accuracy = None
		self.metrics_val_loss = None
		self.weights_save_path = None
		self.model_save_path = None
		self.k_fold_test_accuracy_average = None
		self.k_fold_test_accuracy_std = None
		self.k_fold_test_accuracy_min = None
		self.k_fold_test_accuracy_max = None
		self.k_fold_metrics = None

	def short_description(self):
		return f"{self.id} {self.fitness:.5f} ({f'k-folds: {self.k_fold_test_accuracy_average:.5f} sd: {self.k_fold_test_accuracy_std:.5f} ' if self.k_fold_test_accuracy_average else ''}{f'final: {self.final_test_accuracy:.5f} ' if self.final_test_accuracy else ''}acc: {self.test_accuracy:.5f} p: {self.parameters})"

	def json_statistics(self):
		""" return dictionary of statistics for individual to write to json file"""
		return {
			'id': self.id,
			'final_test_accuracy': self.final_test_accuracy,
			'test_accuracy': self.test_accuracy,
			'fitness': self.fitness,
			'trainable_parameters': self.parameters,
			'training_epochs': self.training_epochs,
			'training_time': self.training_time,
			'million_inferences_time': self.million_inferences_time,
			'training_complete': self.training_complete,
			'phenotype': self.phenotype,
			'metrics_accuracy': self.metrics_accuracy,
			'metrics_loss': self.metrics_loss,
			'metrics_val_accuracy': self.metrics_val_accuracy,
			'metrics_val_loss': self.metrics_val_loss,
			'k_fold_test_accuracy_average': self.k_fold_test_accuracy_average,
			'k_fold_test_accuracy_std': self.k_fold_test_accuracy_std,
			'k_fold_test_accuracy_min': self.k_fold_test_accuracy_min,
			'k_fold_test_accuracy_max': self.k_fold_test_accuracy_max,
			'k_fold_metrics': self.k_fold_metrics,
			'evolution_history': self.evolution_history
		}

	def initialise_individual_random(self, grammar, levels_back, reuse, init_max):
		"""Randomly creates a candidate solution

			Parameters
			----------
			grammar : Grammar
				grammar instances that stores the expansion rules
			levels_back : dict
				number of previous layers a given layer can receive as input
			reuse : float
				likelihood of reusing an existing layer

			Returns
			-------
			candidate_solution : Individual
				randomly created candidate solution
		"""

		for non_terminal, min_expansions, max_expansions in self.network_structure:
			new_module = Module(non_terminal, min_expansions, max_expansions, levels_back[non_terminal])
			new_module.initialise_module(grammar, reuse, init_max)

			self.modules.append(new_module)

		# Initialise output
		self.output = grammar.initialise_layer(self.output_rule)

		# Initialise the macro structure: learning, data augmentation, etc.
		for rule in self.learning_rule:
			self.macro.append(grammar.initialise_layer(rule))
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

		new_module = Module('features', 0, 10, 1)
		new_module.initialise_module_as_lenet(grammar)
		self.modules.append(new_module)

		new_module = Module('classification', 1, 5, 1)
		new_module.initialise_module_as_lenet(grammar)
		self.modules.append(new_module)

		# Initialise output
		self.output = grammar.initialise_layer(self.output_rule)

#		self.output = [('output', [('layer:output', ''), ('num-units', 10), ('bias', 'True')])]

		# Initialise the macro structure: learning, data augmentation, etc.
		self.macro = default_learning_rule_adam()
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

		new_module = Module('features', 0, 10, 1)
		new_module.initialise_module_as_perceptron(grammar)
		self.modules.append(new_module)

		new_module = Module('classification', 1, 5, 1)
		new_module.initialise_module_as_perceptron(grammar)
		self.modules.append(new_module)

		# Initialise output
		self.output = grammar.initialise_layer(self.output_rule)

		# Initialise the macro structure: learning, data augmentation, etc.
		self.macro = default_learning_rule_adam()
		return self

	def log_mutation(self, description):
		"""log a mutation"""
		self.evolution_history.append(f"{self.id} <- {self.parent}: {description}")
		log_mutation(f"mutate {self.id} <- {self.parent}: {description}")

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
			offset = layer_counter
			for layer_idx, layer in enumerate(module.layers):
				layer_counter += 1
				phenotype += '\n' + grammar.decode_layer(module.module, layer) + ' input:' + ",".join(map(str, np.array(module.connections[layer_idx]) + offset))

		phenotype += '\n' + grammar.decode_layer(self.output_rule, self.output) + ' input:' + str(layer_counter - 1)

		for rule_idx, learning_rule in enumerate(self.learning_rule):
			phenotype += '\n' + grammar.decode_layer(learning_rule, self.macro[rule_idx])

		phenotype = phenotype.lstrip('\n')
		self.phenotype = phenotype.split('\n')
		return phenotype

	def evaluate_individual(self, grammar, cnn_eval, datagen, datagen_test, save_path, max_training_time, max_training_epochs):
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
			phenotype = self.get_phenotype(grammar)

			model_save_path = save_path + 'individual-' + self.id + '.h5'

			metrics = None
			try:
				metrics = cnn_eval.evaluate_cnn(phenotype, max_training_time, max_training_epochs, model_save_path, self.id, cnn_eval.dataset, datagen, datagen_test)
			except tf.errors.ResourceExhaustedError as e:
				log_warning(f"{self.id} : ResourceExhaustedError {e}")
				keras.backend.clear_session()
			except (TypeError, ValueError) as e:
				log_warning(f"{self.id} : caught exception {e}")
				keras.backend.clear_session()

			if metrics is not None:
				self.model_save_path = model_save_path
				self.metrics_accuracy = metrics['accuracy']
				self.metrics_loss = metrics['loss']
				self.metrics_val_accuracy = metrics['val_accuracy']
				self.metrics_val_loss = metrics['val_loss']
				self.parameters = metrics['trainable_parameters']
				self.training_epochs += metrics['training_epochs']
				self.training_time += metrics['training_time']
				self.million_inferences_time = metrics['million_inferences_time']
				self.fitness = metrics['fitness']
				if 'test_accuracy' in metrics:
					self.test_accuracy = metrics['test_accuracy']
				else:
					self.test_accuracy = -1
			else:
				self.fitness = None
				self.test_accuracy = None
				self.parameters = -1
				self.million_inferences_time = -1

			self.training_complete = True
		return self.fitness

	def evaluate_with_k_fold_validation(self, grammar, k_fold_eval, nfolds, datagen, datagen_test, max_training_time, max_training_epochs):
		"""
			Performs the evaluation of a candidate solution

			Parameters
			----------
			grammar : Grammar
				grammar instance that stores the expansion rules
			k_fold_eval : Evaluator
				Evaluator instance used to train the networks
			datagen : keras.preprocessing.image.ImageDataGenerator
				Data augmentation method image data generator
			datagen_test : keras.preprocessing.image.ImageDataGenerator
				Image data generator without augmentation

			Returns
			-------
			fitness : float
		"""

		phenotype = self.get_phenotype(grammar)

		try:
			metrics = k_fold_eval.evaluate_cnn_k_folds(phenotype, nfolds, max_training_time, max_training_epochs, datagen, datagen_test)
			test_accuracy_list = metrics['test_accuracy_list']
			self.k_fold_test_accuracy_average = np.average(test_accuracy_list)
			self.k_fold_test_accuracy_std = np.std(test_accuracy_list)
			self.k_fold_test_accuracy_min = np.min(test_accuracy_list)
			self.k_fold_test_accuracy_max = np.max(test_accuracy_list)
			self.k_fold_metrics = metrics
			log_bold(f"--> {self.id} with {nfolds} folds: avg acc: {self.k_fold_test_accuracy_average:0.5f} sd: {self.k_fold_test_accuracy_std:0.5f} range: {self.k_fold_test_accuracy_max - self.k_fold_test_accuracy_min:0.5f}")
		except tf.errors.ResourceExhaustedError as e:
			keras.backend.clear_session()
			return None
		except TypeError as e2:
			keras.backend.clear_session()
			return None

	def calculate_final_test_accuracy(self, cnn_eval):
		self.final_test_accuracy = cnn_eval.final_test_saved_model(self.model_save_path)
		return self.final_test_accuracy
