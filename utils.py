import random
import tensorflow as tf
import keras
from time import time
import numpy as np
import os
import pickle
from enum import Enum

from utilities.data import Dataset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
import concurrent.futures
import threading

from runstatistics import RunStatistics, CnnEvalResult
from logger import *

N_GPUS = None                 # number of GPUs to simulate for parallel evaluation

# Tuning parameters
PREDICT_BATCH_SIZE = 1024   # batch size used for model.predict(), unused
EARLY_STOP_DELTA = 0.001    # currently unused
EARLY_STOP_PATIENCE = 3

LOG_MODEL_TRAINING = 0		# training progress: 1 for progress bar, 2 for one line per epoch
LOG_EARLY_STOPPING = False	# log early stopping
LOG_MODEL_SAVE = 1			# log for saving after each epoch

class Type(Enum):
	NONTERMINAL = 0
	TERMINAL = 1
	FLOAT = 2
	INT = 3
	CAT = 4

gpu_list = []

def init_gpu(n_gpus=N_GPUS):
	global N_GPUS
	global gpu_list

	if not N_GPUS:
		# tf.keras.mixed_precision.set_global_policy("mixed_float16") # using mixed precision made training about 30% slower on average ?!?
		# tf.config.optimizer.set_jit(True)

		if n_gpus <= 0:
			n_gpus = 1
		N_GPUS = n_gpus
		gpus = tf.config.experimental.list_physical_devices('GPU')
		# only allocate as much GPU memory as needed
		if n_gpus == 1:
			tf.config.experimental.set_memory_growth(gpus[0], True)
		else:
			tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=8000/N_GPUS) for i in range(0, N_GPUS)])
		gpu_list = tf.config.list_logical_devices('GPU')
		log_bold(f"{len(gpus)} physical GPUs, {len(gpu_list)} logical GPUs")

def store_random_state():
	""" store the random state relevant for mutations """
	return (random.getstate(), np.random.get_state())

def restore_random_state(random_state : tuple):
	random.setstate(random_state[0])
	np.random.set_state(random_state[1])

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

	def __init__(self, dataset_name, fitness_func=fitness_function_accuracy, max_training_time=10, max_training_epochs=10, max_parameters = 0, input_size=(28, 28, 1), for_k_fold_validation=False, calculate_fitness_with_k_folds_accuracy=False, test_init_seeds=False,
					evaluation_cache_path='', experiment_name='', data_generator=None, data_generator_test=None, use_float=False, n_gpus=None, batch_size=None):
		"""
			Creates the Evaluator instance and loads the dataset.

			Parameters
			----------
			dataset_name : str
				dataset to be loaded
			data_generator : keras.preprocessing.image.ImageDataGenerator
				Data augmentation method image data generator
			data_generator_test : keras.preprocessing.image.ImageDataGenerator
				Image data generator without augmentation
			fitness_func : function
				calculates fitness from accuracy and number of trainable weights
			batch_size : int
				if given, will override batch size specified in individuals
		"""
		if n_gpus is not None:
			init_gpu(n_gpus=n_gpus)

		self.dataset = Dataset(dataset_name, use_float=use_float, for_k_fold_validation=for_k_fold_validation)
		self.fitness_func = fitness_func
		self.max_training_time = max_training_time
		self.max_training_epochs = max_training_epochs
		self.max_parameters = max_parameters

		self.early_stop_delta = EARLY_STOP_DELTA
		self.early_stop_patience = EARLY_STOP_PATIENCE

		self.calculate_fitness_with_k_folds = calculate_fitness_with_k_folds_accuracy
		self.test_init_seeds = test_init_seeds
		self.evaluation_cache_path = evaluation_cache_path
		self.evaluation_cache = {}
		self.evaluation_cache_changed = False

		self.experiment_name = experiment_name
		self.data_generator = data_generator
		self.data_generator_test = data_generator_test
		self.input_size = input_size
		self.batch_size = batch_size

		if self.evaluation_cache_path:
			if os.path.isfile(self.evaluation_cache_path):
				with open(self.evaluation_cache_path, 'rb') as fh:
					self.evaluation_cache = pickle.load(fh)
				log_bold(f"loaded {len(self.evaluation_cache)} cache entries from {self.evaluation_cache_path}")
			else:
				log_bold(f"will create new evaluation cache {self.evaluation_cache_path}")


	def init_options(self, early_stop_patience, early_stop_delta):
		self.early_stop_patience = early_stop_patience
		self.early_stop_delta = early_stop_delta

	@staticmethod
	def get_n_gpus():
		""" returns the number of logical GPUs, or the number of models to evaluate in parallel """
		return N_GPUS

	def flush_evaluation_cache(self):
		""" if evaluation cache entries have been added, write all to file """
		if self.evaluation_cache_path and self.evaluation_cache_changed and self.test_init_seeds == 0:
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

		input, output = Evaluator.assemble_network_input_output(keras_layers, input_size)

		model = tf.keras.models.Model(inputs=input, outputs=output)
		return model

	@staticmethod
	def assemble_network_input_output(keras_layers, input_size):
		# input layer
		input = tf.keras.layers.Input(shape=input_size)
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

			elif layer_type == 'conv1d':  # missing initializer
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
				# use input for first layer, otherwise input
				input_layer = input if input_idx == -1 else data_layers[input_idx]
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
						if input.shape[-3:][0] > minimum_shape:
							actual_shape = int(input.shape[-3:][0])
							merge_signals.append(tf.keras.layers.MaxPooling2D(pool_size=(actual_shape - (minimum_shape - 1), actual_shape - (minimum_shape - 1)), strides=1)(input))
						else:
							merge_signals.append(input)

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
		return input, data_layers[-1]

	@staticmethod
	def calculate_model_multiplications(model):
		""" print number of multiplications per layer of a keras model"""
		for layer in model.layers:
			weights = sum([i.size for i in layer.get_weights()])
			outputs = int(np.prod(layer.output_shape[1: -1]))
			multiplications = weights * outputs
			print(f'output shape: {layer.output_shape[1:]}, weights: {weights}, multiplications: {multiplications}, layer: {layer.name}')
			pass

	@staticmethod
	def get_model_summary(model):
		""" returns model summary from keras as a line list """
		stringlist = []
		model.summary(line_length=100, print_fn=lambda x: stringlist.append(x))
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

	def cache_key(self, phenotype):
		""" generate key for metrics lookup """
		return f"{phenotype}#{self.max_training_time}#{self.max_training_epochs}"

	def cache_lookup(self, phenotype, stat):
		""" look up phenotype in cache. Returns metrics if found or None """
		cache_key = self.cache_key(phenotype)
		if self.evaluation_cache_path and cache_key in self.evaluation_cache:
			cache_entry = self.evaluation_cache[cache_key]
			if self.is_model_valid(cache_entry.metrics.parameters):
				log_debug('using cached metrics from ' + cache_entry.origin_description)
				if stat:
					stat.record_evaluation(seconds=cache_entry.metrics.eval_time, is_cache_hit=True)
				return cache_entry.metrics

		return None

	def cache_update(self, phenotype, metrics, origin_id):
		""" add/update cache entry after evaluation. origin_description is for information only. """
		if self.evaluation_cache_path:
			new_cache_entry = Evaluator.EvaluationCacheEntry(f"{self.experiment_name}:{origin_id}", metrics)
			self.evaluation_cache[self.cache_key(phenotype)] = new_cache_entry
			self.evaluation_cache_changed = True

	def construct_keras_layers(self, phenotype):
		""" construct list of layers to pass into keras, and learning parameters from phenotype string """
		model_phenotype, learning_phenotype = phenotype.split('learning:')
		learning_phenotype = 'learning:' + learning_phenotype.rstrip().lstrip()
		model_phenotype = model_phenotype.rstrip().lstrip().replace('  ', ' ')
		keras_layers = self.get_keras_layers(model_phenotype)
		keras_learning = self.get_learning(learning_phenotype)
		return keras_layers, keras_learning

	@staticmethod
	def get_model_parameters(model):
		""" calculate number of trainable parameters from keras model """
		# parameters = count_params(model.trainable_weights)                    ! Deprecated !
		parameters = sum([np.prod(keras.backend.get_value(w).shape) for w in model.trainable_weights])
		# non_trainable_parameters = sum([np.prod(keras.backend.get_value(w).shape) for w in model.non_trainable_weights])  # not used
		return parameters

	def is_model_valid(self, parameters):
		if self.max_parameters > 0 and parameters > self.max_parameters:
				log_warning(f"New model is invalid because {parameters=} > {self.max_parameters}")
				return False
		return True

	def validate_cnn(self, phenotype):
		""" validate the phenotype by generating keras layers anc compiling the model in keras. Will throw an exception for an invalid model. """
		start_time = time()

		keras_layers, keras_learning = self.construct_keras_layers(phenotype)

		model = self.assemble_network(keras_layers, self.input_size)
		opt = self.assemble_optimiser(keras_learning)
		model.compile(optimizer=opt,
						loss='sparse_categorical_crossentropy',
						metrics=['accuracy'])
		is_valid = self.is_model_valid(Evaluator.get_model_parameters(model))
		del model
		log(f"Validated in {time() - start_time:.2f}")
		return is_valid


	def evaluate_cnn(self, phenotype: str, dataset: Dataset, stat, for_k_folds_validation=False):
		"""
			Evaluates the phenotype with keras

			Parameters
			----------
			phenotype : str
				individual phenotypes (one or more)
			dataset : Dataset
				train and test datasets
			stat : RunStatistics
				for recording statistics
			for_k_folds_validation : bool
				suppress caching, logging and early stopping for k-folds validation

			Returns
			-------
			result : CnnEvalResult
				contains all result metrics
				or True/False in validate_only mode
			The function throws exceptions if the keras model is invalid, or some resources are exhausted
		"""
		# Using mixed precision slows down LeNet training by 50%. Is this because this model is too small?
		# tf.keras.mixed_precision.set_global_policy("mixed_float16")

		start_time = time()

		keras_layers, keras_learning = self.construct_keras_layers(phenotype)

		model = self.assemble_network(keras_layers, self.input_size)
		opt = self.assemble_optimiser(keras_learning)
		model.compile(optimizer=opt,
						loss='sparse_categorical_crossentropy',
						metrics=['accuracy'])

		keras_layers_count = len(keras_layers)
		batch_size = self.batch_size if self.batch_size else int(keras_learning['batch_size'])

		model_summary = Evaluator.get_model_summary(model)
		model_layers = len(model.get_config()['layers'])
		parameters = Evaluator.get_model_parameters(model)

		if not self.is_model_valid(parameters):
			del model
			return None

		# time based stopping
		timed_stopping = TimedStopping(seconds=self.max_training_time)

		callbacks_list = [timed_stopping]

		# early stopping not used
		# if not for_k_folds_validation:
		#   early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=False, verbose=LOG_EARLY_STOPPING)
		#   # or early_stop = keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=self.early_stop_delta, patience=self.early_stop_patience, restore_best_weights=False, verbose=LOG_EARLY_STOPPING)
		#   callbacks_list.append(early_stop)

		training_start_time = time()

		if self.data_generator is not None:
			score = model.fit_generator(self.data_generator.flow(dataset.X_train, dataset.y_train, batch_size=batch_size),
										steps_per_epoch=(dataset.X_train.shape[0] // batch_size),
										epochs=self.max_training_epochs,
										validation_data=(self.data_generator_test.flow(dataset.X_val, dataset.y_val, batch_size=batch_size)),
										validation_steps=(dataset.X_val.shape[0] // batch_size),
										callbacks=callbacks_list,
										initial_epoch=0,
										verbose=LOG_MODEL_TRAINING)
		elif hasattr(dataset, 'train_dataset'):
			score = model.fit(dataset.train_dataset,
								batch_size=batch_size,
								epochs=self.max_training_epochs,
								steps_per_epoch=(dataset.train_dataset_size // batch_size),
								validation_data=dataset.val_dataset,
								callbacks=callbacks_list,
								initial_epoch=0,
								verbose=LOG_MODEL_TRAINING)
		else:
			score = model.fit(dataset.X_train, dataset.y_train,
								batch_size=batch_size,
								epochs=self.max_training_epochs,
								steps_per_epoch=(dataset.train_dataset_size // batch_size),
								validation_data=(dataset.X_val, dataset.y_val),
								validation_steps=(dataset.X_val.shape[0] // batch_size),
								callbacks=callbacks_list,
								initial_epoch=0,
								verbose=LOG_MODEL_TRAINING)


		training_time = time() - training_start_time
		training_epochs = len(score.epoch)
		timer_stop_triggered = timed_stopping.timer_stop_triggered
		early_stop_triggered = training_epochs < self.max_training_epochs and not timer_stop_triggered

		# measure test accuracy
		test_accuracy, model_test_time, million_inferences_time = self.test_model_with_data(model, dataset.X_test, dataset.y_test)

		fitness = self.fitness_func(test_accuracy, parameters)

		# measure final test accuracy
		final_test_accuracy, final_test_time, million_inferences_time = self.test_model_with_data(model, dataset.X_final_test, dataset.y_final_test)

		del model
		# keras.backend.clear_session()

		eval_time = time() - start_time

		result = CnnEvalResult(score.history, final_test_accuracy, eval_time, training_time, final_test_time, million_inferences_time, timer_stop_triggered, early_stop_triggered,
								parameters, keras_layers_count, model_layers, test_accuracy, fitness, model_summary)

		stat.record_evaluation(seconds=eval_time, is_k_folds=for_k_folds_validation)

		return result


	def evaluate_cnn_init_seeds(self, phenotype, name, metrics, n_seeds, dataset, stat):
		""" evaluate individual for different initialisation seeds """

		k_folds_result = KFoldEvalResult()

		for seed in range(0, n_seeds):
			result = self.evaluate_cnn(phenotype, '', dataset, stat, for_k_folds_validation=True)
			k_folds_result.append_cnn_eval_result(result)

		return k_folds_result

	def evaluate_cnn_k_folds(self, phenotype, name, metrics, n_folds, stat):
		""" evaluate individual for k-folds, using cache """

		cache_key = self.cache_key(phenotype)
		if self.evaluation_cache_path and cache_key in self.evaluation_cache:
			cache_entry = self.evaluation_cache[cache_key]
			if cache_entry.k_fold_metrics is not None:
				log_debug('k-fold cached metrics from ' + cache_entry.origin_description)
				stat.record_evaluation(seconds=cache_entry.k_fold_metrics.total_eval_time, is_cache_hit=True, is_k_folds=True)
				return cache_entry.k_fold_metrics
			else:
				log_debug('*** k-fold metrics not in cache entry ' + cache_entry.origin_description)

		x_combined = self.dataset.combined_dataset
		y_combined = self.dataset.combined_dataset

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
			result = self.evaluate_cnn(phenotype, fold_dataset, stat, for_k_folds_validation=True)
			fold_number += 1
			k_folds_result.append_cnn_eval_result(result)

		if self.evaluation_cache_path and metrics is not None:
			new_cache_entry = Evaluator.EvaluationCacheEntry(f"{self.experiment_name}:{name}", metrics, k_folds_result)
			self.evaluation_cache[cache_key] = new_cache_entry
			self.evaluation_cache_changed = True

		return k_folds_result

	def final_test_saved_model(self, model_path):
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
		accuracy, model_test_time, million_inferences_time = Evaluator.test_model_with_data(model, self.dataset['x_final_test'], self.dataset['y_final_test'], self.data_generator_test)

		return accuracy

	@staticmethod
	def test_model_with_data(model, x_test, y_test, datagen_test=None):
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
		phenotype_lines : list of str
			phenotype of the candidate solution
		fitness : float
			fitness value of the candidate solution
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
	final_test_accuracy: None       # accuracy with final test set
	fitness: None                   # calculated fitness
	metrics: CnnEvalResult
	k_fold_metrics: KFoldEvalResult

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
		# strategy parameters
		self.step_width = 0
		self.previous_step = 0
		# mutation statistics
		self.statistic_nlayers = 0
		self.statistic_variables = 0
		self.statistic_floats = 0
		self.statistic_ints = 0
		self.statistic_cats = 0
		self.statistic_variable_mutations = 0
		self.statistic_layer_mutations = 0
		self.reset_training()

	def reset_training(self):
		"""reset all values computed during training"""
		self.is_parent = False
		self.fitness = None
		self.metrics = None
		self.k_fold_metrics = None

	def __repr__(self):
		return self.description()

	def description(self):
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
		if hasattr(self, 'step_width') and self.step_width:
			result += f" σ: {self.step_width:.4f}"
		return result

	def id_and_layer_description(self):
		return f"{self.id} layers: {len(self.modules_including_macro[0].layers)} + {len(self.modules_including_macro[1].layers)} "

	def log_long_description(self, title, with_history=False):
		""" output long description to log, with phenotype, model summary and evolution history """
		log('\n----------------------------------------------------------------------------------------------------------------------------------------')
		log_bold(f"{title}: {self.description()}\nPhenotype:")

		log(self.metrics.model_summary)
		if with_history:
			log_bold('\nEvolution history:')
			log('\n'.join(self.evolution_history))
			log('\n----------------------------------------------------------------------------------------------------------------------------------------')

	def json_statistics(self):
		""" return dictionary of statistics for individual to write to json file"""
		result = {
			'id': self.id,
			'is_parent': self.is_parent,
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
			grammar : StepperGrammar
				grammar instances that stores the expansion rules
			init_max : dict
				number of layers per module for random initialisation

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
		# log_bold(f"randomly created individual {self.id}:")
		# log(self.get_phenotype(grammar))
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
		self.macro_module.layers = grammar.Module.default_learning_rule_gradient_descent()
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
		self.macro_module.layers = grammar.Module.default_learning_rule_adam()
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


	def validate_individual(self, grammar, cnn_eval, stat, use_cache=True):

		phenotype = self.get_phenotype(grammar)

		# look up in cache first - entries in cache are valid
		if use_cache:
			metrics = cnn_eval.cache_lookup(phenotype, stat)
			if metrics:
				self.set_metrics(metrics, cnn_eval, 'v')
				return True

		# store random state
		random_state = store_random_state()

		is_valid = False
		try:
			is_valid = cnn_eval.validate_cnn(phenotype)

		except tf.errors.ResourceExhaustedError as e:
			log_warning(f"Validating {self.id} : ResourceExhaustedError {e}")
			keras.backend.clear_session()
			stat.record_evaluation(is_invalid=True)
		except (TypeError, ValueError) as e:
			log_warning(f"Validating {self.id} : caught exception {e}")
			keras.backend.clear_session()
			stat.record_evaluation(is_invalid=True)

		restore_random_state(random_state)

		return is_valid

	def evaluate_individual(self, grammar, cnn_eval, stat, use_cache = True):
		"""
			Performs the evaluation of a candidate solution

			Parameters
			----------
			grammar : Grammar
				grammar instance that stores the expansion rules
			stat : RunStatistics
				for recording statistics
			cnn_eval : Evaluator
				Evaluator instance used to train the networks

			Returns
			-------
			fitness : float
		"""

		# generate phenotype
		phenotype = self.get_phenotype(grammar)

		# look up in cache
		metrics = cnn_eval.cache_lookup(phenotype, stat) if use_cache else None
		if metrics:
			self.set_metrics(metrics, cnn_eval, 'c')
		else:
			random_state = store_random_state()

			# evaluate and catch exceptions
# 			try:
			metrics = cnn_eval.evaluate_cnn(phenotype, cnn_eval.dataset, stat)
			if metrics:
				cnn_eval.cache_update(phenotype, metrics, self.id)
				self.set_metrics(metrics, cnn_eval)

#			except tf.errors.ResourceExhaustedError as e:
#				log_warning(f"{self.id} : ResourceExhaustedError {e}")
#				keras.backend.clear_session()
#				stat.record_evaluation(is_invalid=True)
#			except (TypeError, ValueError) as e:
#				log_warning(f"{self.id} : caught exception {e}")
#				keras.backend.clear_session()
#				stat.record_evaluation(is_invalid=True)

			restore_random_state(random_state)

		return self.metrics is not None

		# phenotype_list = [ind.get_phenotype(grammar) for ind in individual_list]
		# results = [x for x in executor.map(cnn_eval.evaluate_cnn, phenotype_list)]

	@staticmethod
	def evaluate_in_thread(ind, grammar, cnn_eval, stat):
		thread_name =  threading.current_thread().name
		thread_number = int(thread_name[thread_name.rindex("_") + 1:])
		with tf.device(gpu_list[thread_number]):
			log_bold(f"{ind.id_and_layer_description()} starts on {gpu_list[thread_number].name} in thread {thread_name}")
			result = ind.evaluate_individual(grammar, cnn_eval, stat, use_cache=False)
		return result

	@staticmethod
	def evaluate_multiple_individuals(individual_list, grammar, cnn_eval, stat):
		""" evaluate multiple individuals with multithreading """
		ok = True
		if N_GPUS > 1 and len(individual_list):
			with concurrent.futures.ThreadPoolExecutor(max_workers=N_GPUS, thread_name_prefix='Evaluator') as executor:
				future_to_evaluate_individual = {executor.submit(Individual.evaluate_in_thread, ind, grammar, cnn_eval, stat) : ind for ind in individual_list}
				for future in concurrent.futures.as_completed(future_to_evaluate_individual):
					result = future_to_evaluate_individual[future]
					if not result:
						log_warning(f"Invalid multiple evaluation in list starting with {individual_list[0].id}")
						ok = False
		else:
			for ind in individual_list:
				result = ind.evaluate_individual(grammar, cnn_eval, stat)
				if not result:
					log_warning(f"Invalid multiple evaluation in list starting with {individual_list[0].id}")
					ok = False
		return ok

	def set_metrics(self, metrics, cnn_eval, log_suffix=None):
		""" set metrics to individual after evaluation or cache lookup, and calculate fitness """
		self.metrics = metrics
		self.fitness = cnn_eval.calculate_fitness(self)
		log_training(self.id_and_layer_description() + self.metrics.summary(f" ({log_suffix})" if log_suffix else ''))
		if not self.metrics.training_epochs:
			log_warning(f"*** {self.id}: no training epoch completed ***")

	def evaluate_individual_k_folds(self, grammar, cnn_eval, nfolds, stat):
		"""
			Performs the evaluation of a candidate solution

			Parameters
			----------
			grammar : StepperGrammar
				grammar instance that stores the expansion rules
			cnn_eval : Evaluator
				Evaluator instance used to train the networks
			nfolds : int
				number of folds
			stat : RunStatistics
				keeps statistics

			Returns
			-------
			fitness : float
		"""

		random_state = store_random_state()

		phenotype = self.get_phenotype(grammar)

		self.k_fold_metrics = cnn_eval.evaluate_cnn_k_folds(phenotype, self.id, self.metrics, nfolds, stat)
		self.k_fold_metrics.calculate_fitness_mean_std(cnn_eval)
		old_fitness = self.fitness
		self.fitness = cnn_eval.calculate_fitness(self)

		stat.k_fold_accuracy_stds.append(self.k_fold_metrics.accuracy_std)
		stat.k_fold_final_accuracy_stds.append(self.k_fold_metrics.final_accuracy_std)
		stat.k_fold_fitness_stds.append(self.k_fold_metrics.fitness_std)

		log_bold(f"--> {self.id} with {nfolds} folds: acc: {self.metrics.accuracy:0.5f} -> {self.k_fold_metrics.accuracy:0.5f} (SD:{self.k_fold_metrics.accuracy_std:0.5f}), final acc: {self.k_fold_metrics.final_accuracy:0.5f} (SD:{self.k_fold_metrics.final_accuracy_std:0.5f}), fitness: {old_fitness:0.5f} -> {self.fitness:0.5f}")

		restore_random_state(random_state)

	def evaluate_individual_init_seeds(self, grammar, cnn_eval, n_seeds, stat):
		"""
			Performs the evaluation of a candidate solution

			Parameters
			----------
			grammar : Grammar
				grammar instance that stores the expansion rules
			cnn_eval : Evaluator
				Evaluator instance used to train the networks
			n_seeds : int
				number of seeds to test
			stat : RunStatistics
				records statistics

			Returns
			-------
			fitness : float
		"""

		random_state = store_random_state()

		phenotype = self.get_phenotype(grammar)

		self.k_fold_metrics = cnn_eval.evaluate_cnn_init_seeds(phenotype, self.id, self.metrics, n_seeds, cnn_eval.dataset, stat)
		self.k_fold_metrics.calculate_fitness_mean_std(cnn_eval)
		old_fitness = self.fitness
		self.fitness = cnn_eval.calculate_fitness(self)

		stat.k_fold_accuracy_stds.append(self.k_fold_metrics.accuracy_std)
		stat.k_fold_final_accuracy_stds.append(self.k_fold_metrics.final_accuracy_std)
		stat.k_fold_fitness_stds.append(self.k_fold_metrics.fitness_std)

		log_bold(f"--> {self.id} with {n_seeds} seeds: acc: {self.metrics.accuracy:0.5f} -> {self.k_fold_metrics.accuracy:0.5f} (SD:{self.k_fold_metrics.accuracy_std:0.5f}), final acc: {self.k_fold_metrics.final_accuracy:0.5f} (SD:{self.k_fold_metrics.final_accuracy_std:0.5f}), fitness: {old_fitness:0.5f} -> {self.fitness:0.5f}")

		restore_random_state(random_state)

	def compute_mutated_variables_statistics(self, mutable_vars):
		""" record layers and variable statistics for individual """
		self.statistic_nlayers = sum(len(module.layers) for module in self.modules_including_macro)
		self.statistic_variables = len(mutable_vars)
		self.statistic_floats = sum(1 for mvar in mutable_vars if mvar.type == Type.FLOAT)
		self.statistic_ints = sum(1 for mvar in mutable_vars if mvar.type == Type.INT)
		self.statistic_cats = sum(1 for mvar in mutable_vars if mvar.type == Type.CAT)
		self.statistic_variable_mutations = sum(1 for mvar in mutable_vars if mvar.new_value is not None)

	def record_statistics(self, ind_stats: RunStatistics.IndividualStatistics):
		if hasattr(self, 'statistic_nlayers') and hasattr(ind_stats, 'statistic_nlayers'):
			ind_stats.statistic_nlayers.append(self.statistic_nlayers)
			ind_stats.statistic_variables.append(self.statistic_variables)
			ind_stats.statistic_floats.append(self.statistic_floats)
			ind_stats.statistic_ints.append(self.statistic_ints)
			ind_stats.statistic_cats.append(self.statistic_cats)
			ind_stats.statistic_variable_mutations.append(self.statistic_variable_mutations)
			ind_stats.statistic_layer_mutations.append(self.statistic_layer_mutations)
		if hasattr(self, 'step_width') and hasattr(ind_stats, 'step_width'):
			ind_stats.step_width.append(self.step_width)
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
