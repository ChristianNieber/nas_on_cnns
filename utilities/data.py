# Copyright 2019 Filipe Assuncao

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from utilities.svhn import load_svhn
from utilities.cifar import load_cifar
from utilities.tiny_imagenet import load_tiny_imagenet
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from multiprocessing import Pool
import tensorflow as tf
import keras
import contextlib
import sys

# dataset paths - change if the path is different
SVHN = 'utilities/data/svhn'
TINY_IMAGENET = 'utilities/data/tiny-imagenet-200'

class Dataset:
	def __init__(self, dataset_name=None, for_k_fold_validation=False, shape=(32, 32)):
		"""
			Load a specific dataset

			Parameters
			----------
			dataset_name : str
				dataset to load
			for_k_fold_validation : bool
				prepare X_combined/y_combined for k-fold validation
			shape : tuple(int, int)
				shape of the instances
		"""
		if dataset_name:
			reshape_data = True

			if dataset_name == 'mnist':

				(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
				n_classes = 10

				reshape_data = False

			elif dataset_name == 'fashion-mnist':
				(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
				n_classes = 10

				x_train = 255 - x_train
				x_test = 255 - x_test

				x_train = self.resize_data((x_train, shape))
				x_test = self.resize_data((x_test, shape))

			# 255, unbalanced
			elif dataset_name == 'svhn':
				x_train, y_train, x_test, y_test = load_svhn(SVHN)
				n_classes = 10

			# 255, 50000, 10000
			elif dataset_name == 'cifar10':
				x_train, y_train, x_test, y_test = load_cifar(10)
				n_classes = 10

			# 255, 50000, 10000
			elif dataset_name == 'cifar100-fine':
				x_train, y_train, x_test, y_test = load_cifar(100, 'fine')
				n_classes = 100

			elif dataset_name == 'cifar100-coarse':
				x_train, y_train, x_test, y_test = load_cifar(100, 'coarse')
				n_classes = 20

			elif dataset_name == 'tiny-imagenet':
				x_train, y_train, x_test, y_test = load_tiny_imagenet(TINY_IMAGENET, shape)
				n_classes = 200

			else:
				print('Error: the dataset is not valid')
				sys.exit(-1)

			self.prepare_data(x_train, y_train, x_test, y_test, reshape_data, n_classes, for_k_fold_validation)


	def prepare_data(self, x_train, y_train, x_test, y_test, reshape_data, n_classes=10, for_k_fold_validation=False):
		"""
			Split the data into independent sets

			Parameters
			----------
			x_train : np.array
				training instances
			y_train : np.array
				training labels
			x_test : np.array
				testing instances
			x_test : np.array
				testing labels

			Returns
			-------
			dataset : dict
				instances of the dataset:
					For evolution:
						- evo_x_train and evo_y_train : training x, and y instances
						- evo_x_val and evo_y_val : validation x, and y instances
													used for early stopping
						- evo_x_test and evo_y_test : testing x, and y instances
													  used for fitness assessment
					After evolution:
						- x_test and y_test : for measuring the effectiveness of the model on unseen data
		"""

		# x_train = x_train.astype('float32') / 255.0    # converting from uint8 to float slows training down
		# x_test = x_test.astype('float32') / 255.0

		if reshape_data:
			x_train = x_train.reshape((-1, 32, 32, 3))
			x_test = x_test.reshape((-1, 32, 32, 3))


		self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(x_train, y_train, test_size=7776, shuffle=True, stratify=y_train)
		self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(self.X_val, self.y_val, test_size=3888, shuffle=True, stratify=self.y_val)

		self.X_final_test = x_test
		self.y_final_test = y_test

		# evo_y_train = keras.utils.to_categorical(evo_y_train, n_classes)
		# evo_y_val = keras.utils.to_categorical(evo_y_val, n_classes

		# dataset = {
		# 	'evo_x_train': evo_x_train, 'evo_y_train': evo_y_train,
		# 	'evo_x_val': evo_x_val, 'evo_y_val': evo_y_val,
		# 	'evo_x_test': evo_x_test, 'evo_y_test': evo_y_test,
		# 	'x_final_test': x_test, 'y_final_test': y_test
		# }
		if for_k_fold_validation:
			# self.X_combined = np.r_[x_train, x_test]
			# self.y_combined = np.r_[y_train, y_test]
			self.X_combined = x_train
			self.y_combined = y_train


	@staticmethod
	def resize_data(args):
		"""
			Resize the dataset 28 x 28 datasets to 32x32

			Parameters
			----------
			args : tuple(np.array, (int, int))
				instances, and shape of the reshaped signal

			Returns
			-------
			content : np.array
				reshaped instances
		"""

		import tensorflow as tf

		content, shape = args
		content = content.reshape(-1, 28, 28, 1)

		if shape != (28, 28):
			content = tf.image.resize(content, shape, tf.image.ResizeMethod.NEAREST_NEIGHBOR)

		content = tf.image.grayscale_to_rgb(tf.constant(content))

		return content.numpy()


