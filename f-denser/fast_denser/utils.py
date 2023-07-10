# Copyright 2019 Filipe Assuncao
# Restructured 2023 Christian Nieber

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import tensorflow as tf
import keras
from keras import backend
from keras.callbacks import Callback, ModelCheckpoint
from keras.utils.layer_utils import count_params
from time import time
import numpy as np
import os
from fast_denser.utilities.data import load_dataset

# TODO: future -- impose memory constraints
# tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=50)])

# Tuning parameters
PREDICT_BATCH_SIZE = 1024  # batch size used for model.predict()

DEBUG = True
LOG_MODEL_SUMMARY = False				# keras summary of each evaluated model
LOG_MODEL_TRAINING = 0					# training progress: 1 for progress bar, 2 for one line per epoch
LOG_MODEL_SAVE = True					# log for saving after each epoch
LOG_TIMED_STOPPING = False				# log stopping by timer
LOG_EARLY_STOPPING = True				# log early stopping
SAVE_MODEL_AFTER_EACH_EPOCH = False		# monitor and save model after each epoch


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
		verbose : bool
			verbosity mode.

		Methods
		-------
		on_train_begin(logs)
			method called upon training beginning
		on_epoch_end(epoch, logs={})
			method called after the end of each training epoch
	"""

	def __init__(self, seconds=None, verbose=0):
		"""
		Parameters
		----------
		seconds : float
			maximum time before stopping.
		verbose : bool
			verbosity mode
		"""
		super(keras.callbacks.Callback, self).__init__()
		self.start_time = 0
		self.seconds = seconds
		self.verbose = verbose

	def on_train_begin(self, logs={}):
		"""
			Method called upon training beginning

			Parameters
			----------
			logs : dict
				training logs
		"""
		self.start_time = time()

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
			if self.verbose:
				print('Stopping after %s seconds.' % self.seconds)


class Evaluator:
	"""
		Stores the dataset, maps the phenotype into a trainable model, and
		evaluates it

		Attributes
		----------
		dataset : dict
			dataset instances and partitions
		fitness_metric : function
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

	def __init__(self, dataset, fitness_metric):
		"""
			Creates the Evaluator instance and loads the dataset.

			Parameters
			----------
			dataset : str
				dataset to be loaded
		"""

		#        def setUp(self):
		self.dataset = load_dataset(dataset)
		self.fitness_metric = fitness_metric

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

		raw_phenotype = phenotype.split(' ')

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

			# average pooling layer
			elif layer_type == 'pool-avg':
				pool_avg = tf.keras.layers.AveragePooling2D(pool_size=(int(layer_params['kernel-size']), int(layer_params['kernel-size'])),
															strides=int(layer_params['stride']),
															padding=layer_params['padding'])
				layers.append(pool_avg)

			# max pooling layer
			elif layer_type == 'pool-max':
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
													 activation=layer_params['act'],
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

			# END ADD NEW LAYERS

		# Connection between layers
		for layer in keras_layers:
			layer[1]['input'] = list(map(int, layer[1]['input'].split(',')))

		first_fc = True
		data_layers = []
		invalid_layers = []

		for layer_idx, layer in enumerate(layers):
			# try:
			layer_inputs = keras_layers[layer_idx][1]['input']
			if len(layer_inputs) == 1:
				layer_type = keras_layers[layer_idx][0]
				layer_params = keras_layers[layer_idx][1]
				input_idx = layer_inputs[0]
				# use input layer as first input
				if input_idx == -1:
					new_data_layer = layer(inputs)
				# add Flatten layer before first fc layer
				elif layer_type == 'fc' and first_fc:
					first_fc = False
					flatten = tf.keras.layers.Flatten()(data_layers[input_idx])
					new_data_layer = layer(flatten)
				# all other layers
				else:
					new_data_layer = layer(data_layers[input_idx])

				# conv and fc layers can have an optional batch normalisation layer, that should be inserted before the activation layer
				if layer_type == 'conv' or layer_type == 'fc':
					if ('batch-normalization' in layer_params) and layer_params['batch-normalization']:
						new_data_layer = tf.keras.layers.BatchNormalization()(new_data_layer)
					activation_function = layer_params['act']
					new_data_layer = tf.keras.layers.ReLU()(new_data_layer)

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
			# except ValueError as e:
			#    data_layers.append(data_layers[-1])
			#    invalid_layers.append(layer_idx)
			#    if DEBUG:
			#        print(keras_layers[layer_idx][0])
			#        print(e)

		model = tf.keras.models.Model(inputs=inputs, outputs=data_layers[-1])

		if LOG_MODEL_SUMMARY:
			model.summary(line_length=120)

		return model

	def assemble_optimiser(self, learning):
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
											   rho=float(learning['rho']),
											   decay=float(learning['decay']))

		elif learning['learning'] == 'gradient-descent':
			return tf.keras.optimizers.SGD(learning_rate=float(learning['lr']),
										   momentum=float(learning['momentum']),
										   decay=float(learning['decay']),
										   nesterov=bool(learning['nesterov']))

		elif learning['learning'] == 'adam':
			return tf.keras.optimizers.Adam(learning_rate=float(learning['lr']),
											beta_1=float(learning['beta1']),
											beta_2=float(learning['beta2']))

	def evaluate_cnn(self, phenotype, load_prev_weights, max_training_time, max_training_epochs, save_path, individual_name, datagen=None, datagen_test=None, input_size=(28, 28, 1)):  # pragma: no cover
		"""
			Evaluates the keras model using the keras optimiser

			Parameters
			----------
			phenotype : str
				individual phenotype
			load_prev_weights : bool
				resume training from a previous train or not
			weights_save_path : str
				path where to save the model weights after training
			parent_weights_path : str
				path to the weights of the previous training
			max_training_time : float
				maximum training time
			max_training_epochs : int
				maximum number of epochs
			save_path : str
				path where weights are saved
			individual_name : str
				name (<generation>-<number>)
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
		gpu_devices = tf.config.experimental.list_physical_devices('GPU')
		tf.config.experimental.set_memory_growth(gpu_devices[0], True)

		model_build_start_time = time()
		model_phenotype, learning_phenotype = phenotype.split('learning:')
		learning_phenotype = 'learning:' + learning_phenotype.rstrip().lstrip()
		model_phenotype = model_phenotype.rstrip().lstrip().replace('  ', ' ')

		keras_layers = self.get_layers(model_phenotype)
		keras_learning = self.get_learning(learning_phenotype)
		batch_size = int(keras_learning['batch_size'])

		weights_save_path = save_path + 'individual-' + individual_name + '.h5'
		if load_prev_weights and os.path.exists(weights_save_path):
			model = keras.models.load_model(weights_save_path)
			initial_epoch = 10  #TODO
		else:
			initial_epoch = 0
			model = self.assemble_network(keras_layers, input_size)
			opt = self.assemble_optimiser(keras_learning)
			model.compile(optimizer=opt,
						  loss='sparse_categorical_crossentropy',
						  metrics=['accuracy'])
		model_build_time = time() - model_build_start_time

		model_layers = len(model.get_config()['layers'])
		trainable_params_count = count_params(model.trainable_weights)
		non_trainable_parameters = count_params(model.non_trainable_weights)
		trainable_parameters = trainable_params_count + non_trainable_parameters

		print(f"{individual_name} layers:{len(keras_layers):2d}/{model_layers:2d} params:{trainable_parameters:6d}/{non_trainable_parameters}", end="")

		training_start_time = time()

		# early stopping
		early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=int(keras_learning['early_stop']), restore_best_weights=True, verbose=LOG_EARLY_STOPPING)

		# time based stopping
		time_stop = TimedStopping(seconds=max_training_time, verbose=LOG_TIMED_STOPPING)

		callbacks_list = [early_stop, time_stop]

		# save individual with the lowest validation loss
		# useful for when training is halted because of time
		if SAVE_MODEL_AFTER_EACH_EPOCH:
			monitor = ModelCheckpoint(weights_save_path, monitor='val_loss', verbose=LOG_MODEL_SAVE, save_best_only=True)
			callbacks_list.append(monitor)

		if datagen is not None:
			score = model.fit_generator(datagen.flow(self.dataset['evo_x_train'],
													 self.dataset['evo_y_train'],
													 batch_size=batch_size),
										steps_per_epoch=(self.dataset['evo_x_train'].shape[0] // batch_size),
										epochs=max_training_epochs,
										validation_data=(datagen_test.flow(self.dataset['evo_x_val'], self.dataset['evo_y_val'], batch_size=batch_size)),
										validation_steps=(self.dataset['evo_x_val'].shape[0] // batch_size),
										callbacks=callbacks_list,
										initial_epoch=initial_epoch,
										verbose=LOG_MODEL_TRAINING)
		else:
			score = model.fit(x=self.dataset['evo_x_train'],
							  y=self.dataset['evo_y_train'],
							  batch_size=batch_size,
							  epochs=max_training_epochs,
							  steps_per_epoch=(self.dataset['evo_x_train'].shape[0] // batch_size),
							  validation_data=(self.dataset['evo_x_val'], self.dataset['evo_y_val']),
							  callbacks=callbacks_list,
							  initial_epoch=initial_epoch,
							  verbose=LOG_MODEL_TRAINING)

		training_time = time() - training_start_time

		# save final model to file
		model_save_start_time = time()
		model.save(weights_save_path)
		model_save_time = time() - model_save_start_time

		# measure test performance
		model_test_start_time = time()
		if datagen_test is None:
			y_pred_test = model.predict(self.dataset['evo_x_test'], batch_size=PREDICT_BATCH_SIZE, verbose=0)
		else:
			y_pred_test = model.predict_generator(datagen_test.flow(self.dataset['evo_x_test'], batch_size=PREDICT_BATCH_SIZE, shuffle=False), steps=self.dataset['evo_x_test'].shape[0] // 100, verbose=LOG_MODEL_TRAINING)
		test_accuracy = self.fitness_metric(self.dataset['evo_y_test'], y_pred_test)
		model_test_time = time() - model_test_start_time
		million_inferences_time = 1000000.0 * model_test_time / len(y_pred_test)

		training_epochs = len(score.epoch)
		if training_epochs:
			accuracy = score.history['accuracy'][-1]
			val_accuracy = score.history['val_accuracy'][-1]
			loss = score.history['loss'][-1]
			print(f" ep:{training_epochs:2d} inf: {million_inferences_time:0.2f} test: {test_accuracy:0.5f}, acc: {accuracy:0.5f} val: {val_accuracy:0.5f} loss: {loss:0.5f} t: {training_time:0.2f}s (b{model_build_time:0.2f},t{model_test_time:0.2f},s{model_save_time:0.2f})")
		else:
			print(f" *** no training epoch completed ***")

		score.history['training_epochs'] = training_epochs
		score.history['training_time'] = training_time
		score.history['million_inferences_time'] = million_inferences_time
		score.history['trainable_parameters'] = trainable_parameters
		score.history['test_accuracy'] = test_accuracy

		keras.backend.clear_session()

		return score.history

	def test_with_final_test_dataset(self, model_path, datagen_test):  # pragma: no cover
		"""
			Compute testing performance of the model

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
		model_test_start_time = time()
		if datagen_test is None:
			y_pred = model.predict(self.dataset['x_test'], verbose=LOG_MODEL_TRAINING)
		else:
			y_pred = model.predict_generator(datagen_test.flow(self.dataset['x_test'], shuffle=False), verbose=LOG_MODEL_TRAINING)
		accuracy = self.fitness_metric(self.dataset['y_test'], y_pred)
		final_test_time = time() - model_test_start_time
		final_million_inferences_time = 1000000.0 * final_test_time / len(y_pred)
		return accuracy


class Module:
	"""
		Each of the units of the outer-level genotype

		Attributes
		----------
		module : str
			non-terminal symbol
		min_expansions : int
			minimum expansions of the block
		max_expansions : int
			maximum expansions of the block
		levels_back : dict
			number of previous layers a given layer can receive as input
		layers : list
			list of layers of the module
		connections : dict
			list of connections of each layer

		Methods
		-------
			initialise(grammar, reuse)
				Randomly creates a module
	"""

	def __init__(self, module, min_expansions, max_expansions, levels_back):
		"""
			Parameters
			----------
			module : str
				non-terminal symbol
			min_expansions : int
				minimum expansions of the block
					max_expansions : int
				maximum expansions of the block
			levels_back : int
				number of previous layers a given layer can receive as input
		"""

		self.module = module
		self.levels_back = levels_back
		self.layers = []
		self.min_expansions = min_expansions
		self.max_expansions = max_expansions

	def initialise(self, grammar, reuse, init_max):
		"""
			Randomly creates a module

			Parameters
			----------
			grammar : Grammar
				grammar instance that stores the expansion rules

			reuse : float
				likelihood of reusing an existing layer
		"""

		num_expansions = random.choice(init_max[self.module])

		# Initialise layers
		for idx in range(num_expansions):
			if idx > 0 and random.random() <= reuse:
				r_idx = random.randint(0, idx - 1)
				self.layers.append(self.layers[r_idx])
			else:
				self.layers.append(grammar.initialise(self.module))

		# Initialise connections: feed-forward and allowing skip-connections
		self.connections = {}
		for layer_idx in range(num_expansions):
			if layer_idx == 0:
				# the -1 layer is the input
				self.connections[layer_idx] = [-1, ]
			else:
				connection_possibilities = list(range(max(0, layer_idx - self.levels_back), layer_idx - 1))
				if len(connection_possibilities) < self.levels_back - 1:
					connection_possibilities.append(-1)

				sample_size = random.randint(0, len(connection_possibilities))

				self.connections[layer_idx] = [layer_idx - 1]
				if sample_size > 0:
					self.connections[layer_idx] += random.sample(connection_possibilities, sample_size)

	def initialise_module_as_lenet(self, grammar):
		"""
			Creates a pre-defined LeNet module

			Parameters
			----------
			grammar : Grammar
				grammar instance that stores the expansion rules
		"""

		feature_layers_lenet = [
			{'features': [{'ge': 0, 'ga': {}}], 'convolution': [{'ge': 0, 'ga': {'num-filters': ('int', 2.0, 64.0, 6), 'filter-shape': ('int', 2.0, 5.0, 5), 'stride': ('int', 1.0, 3.0, 1)}}],
			 'activation-function': [{'ge': 1, 'ga': {}}], 'padding': [{'ge': 0, 'ga': {}}], 'bias': [{'ge': 0, 'ga': {}}], 'batch-normalization': [{'ge': 0, 'ga': {}}]},
			{'features': [{'ge': 1, 'ga': {}}], 'padding': [{'ge': 1, 'ga': {}}], 'pool-type': [{'ge': 1, 'ga': {}}], 'pooling': [{'ga': {'kernel-size': ('int', 2.0, 5.0, 2), 'stride': ('int', 1.0, 3.0, 2)}, 'ge': 0}]},
			{'features': [{'ge': 0, 'ga': {}}], 'convolution': [{'ge': 0, 'ga': {'num-filters': ('int', 2.0, 64.0, 16), 'filter-shape': ('int', 2.0, 5.0, 5), 'stride': ('int', 1.0, 3.0, 1)}}],
			 'activation-function': [{'ge': 1, 'ga': {}}], 'padding': [{'ge': 1, 'ga': {}}], 'bias': [{'ge': 0, 'ga': {}}], 'batch-normalization': [{'ge': 0, 'ga': {}}]},
			{'features': [{'ge': 1, 'ga': {}}], 'padding': [{'ge': 1, 'ga': {}}], 'pool-type': [{'ge': 1, 'ga': {}}], 'pooling': [{'ga': {'kernel-size': ('int', 2.0, 5.0, 2), 'stride': ('int', 1.0, 3.0, 2)}, 'ge': 0}]},
		]

		classification_layers_lenet = [
			{'classification': [{'ga': {'num-units': ('int', 64.0, 2048.0, 120)}, 'ge': 0}], 'activation-function': [{'ge': 1, 'ga': {}}], 'bias': [{'ge': 0, 'ga': {}}], 'batch-normalization': [{'ge': 0, 'ga': {}}]},
			{'classification': [{'ga': {'num-units': ('int', 64.0, 2048.0, 84)}, 'ge': 0}], 'activation-function': [{'ge': 1, 'ga': {}}], 'bias': [{'ge': 0, 'ga': {}}], 'batch-normalization': [{'ge': 0, 'ga': {}}]},
		]

		if self.module == 'features':
			self.layers = feature_layers_lenet
		elif self.module == 'classification':
			self.layers = classification_layers_lenet

		# Initialise connections: feed-forward and allowing skip-connections
		self.connections = {}
		for layer_idx in range(len(self.layers)):
			if layer_idx == 0:
				# the -1 layer is the input
				self.connections[layer_idx] = [-1, ]
			else:
				self.connections[layer_idx] = [layer_idx - 1]


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
		trainable_parameters : int
			number of trainable parameters of the network
		train_time : float
			maximum training time
		name : str
			name as <generation>-<index>

		Methods
		-------
			initialise(grammar, levels_back, reuse)
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
		self.phenotype = None
		self.name = f"{gen}-{idx}"
		self.parent = None
		self.reset_training()

	def reset_training(self):
		"""reset all values computed during training"""
		self.test_accuracy = None
		self.trainable_parameters = None
		self.training_time = 0
		self.training_epochs = 0
		self.final_test_accuracy = None
		self.million_inferences_time = None
		self.fitness = None
		self.training_complete = False
		self.metrics_accuracy = None
		self.metrics_loss = None
		self.metrics_val_accuracy = None
		self.metrics_val_loss = None
		self.weights_save_path = None

	def json_statistics(self):
		""" return dictionary of statistics for individual to write to json file"""
		return {
			'name': self.name,
			'test_accuracy': self.test_accuracy,
			'fitness': self.fitness,
			'trainable_parameters': self.trainable_parameters,
			'training_epochs': self.training_epochs,
			'training_time': self.training_time,
			'million_inferences_time': self.million_inferences_time,
			'training_complete': self.training_complete,
			'phenotype': self.phenotype,
			'metrics_accuracy': self.metrics_accuracy,
			'metrics_loss': self.metrics_loss,
			'metrics_val_accuracy': self.metrics_val_accuracy,
			'metrics_val_loss': self.metrics_val_loss,
		}

	def initialise(self, grammar, levels_back, reuse, init_max):
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
			new_module.initialise(grammar, reuse, init_max)

			self.modules.append(new_module)

		# Initialise output
		self.output = grammar.initialise(self.output_rule)

		# Initialise the macro structure: learning, data augmentation, etc.
		for rule in self.learning_rule:
			self.macro.append(grammar.initialise(rule))
		return self

	def initialise_as_lenet(self, grammar):
		"""
			Create a pre-set Lenet Individual

			Parameters
			----------
			grammar : Grammar
				grammar instance that stores the expansion rules
			reuse : float
				likelihood of reusing an existing layer

			Returns
			-------
			candidate_solution : Individual
				randomly created candidate solution
		"""

		new_module = Module('features', 1, 1, 1)
		new_module.initialise_module_as_lenet(grammar)
		self.modules.append(new_module)

		new_module = Module('classification', 1, 1, 1)
		new_module.initialise_module_as_lenet(grammar)
		self.modules.append(new_module)

		# Initialise output
		self.output = grammar.initialise(self.output_rule)

		# Initialise the macro structure: learning, data augmentation, etc.
		self.macro = [{'learning': [{'ge': 0, 'ga': {'batch_size': ('int', 50.0, 4096.0, 1024)}}],
						'adam': [{'ge': 0, 'ga': {'lr': ('float', 0.0001, 0.1, 0.0005), 'beta1': ('float', 0.5, 1.0, 0.9), 'beta2': ('float', 0.5, 1.0, 0.999)}}],
						'early-stop': [{'ge': 0, 'ga': {'early_stop': ('int', 5.0, 20.0, 8)}}]}]
		return self

	def log_mutation(self, str):
		"""log a mutation"""
		print(f"mutate {self.name}({self.parent}): {str}")

	def decode(self, grammar):
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
		offset = 0
		layer_counter = 0
		for module in self.modules:
			offset = layer_counter
			for layer_idx, layer_genotype in enumerate(module.layers):
				layer_counter += 1
				phenotype += ' ' + grammar.decode(module.module, layer_genotype) + ' input:' + ",".join(map(str, np.array(module.connections[layer_idx]) + offset))

		phenotype += ' ' + grammar.decode(self.output_rule, self.output) + ' input:' + str(layer_counter - 1)

		for rule_idx, macro_rule in enumerate(self.learning_rule):
			phenotype += ' ' + grammar.decode(macro_rule, self.macro[rule_idx])

		self.phenotype = phenotype.rstrip().lstrip()
		return self.phenotype

	def evaluate_individual(self, grammar, cnn_eval, datagen, datagen_test, save_path, max_training_time, max_training_epochs):  # pragma: no cover
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
			phenotype = self.decode(grammar)

			load_prev_weights = False

			metrics = None
			try:
				metrics = cnn_eval.evaluate_cnn(phenotype, load_prev_weights, max_training_time, max_training_epochs, save_path, self.name, datagen, datagen_test)
			except tf.errors.ResourceExhaustedError as e:
				keras.backend.clear_session()
				return None
			except TypeError as e2:
				keras.backend.clear_session()
				return None

			if metrics is not None:
				self.metrics_accuracy = metrics['accuracy']
				self.metrics_loss = metrics['loss']
				self.metrics_val_accuracy = metrics['val_accuracy']
				self.metrics_val_loss = metrics['val_loss']
				if 'test_accuracy' in metrics:
					self.fitness = metrics['test_accuracy']
					self.test_accuracy = metrics['test_accuracy']
				self.trainable_parameters = metrics['trainable_parameters']
				self.training_epochs += metrics['training_epochs']
				self.training_time += metrics['training_time']
				self.million_inferences_time = metrics['million_inferences_time']
			else:
				self.fitness = -1
				self.test_accuracy = -1
				self.trainable_parameters = -1
				self.million_inferences_time = -1

			self.training_complete = True
		return self.fitness
