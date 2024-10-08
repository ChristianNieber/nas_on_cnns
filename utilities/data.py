import numpy as np
from utilities.tiny_imagenet import load_tiny_imagenet
from sklearn.model_selection import train_test_split
import tensorflow as tf
import sys
import scipy
from logger import *

# dataset paths - change if the path is different
SVHN_PATH = 'utilities/data'
TINY_IMAGENET_PATH = 'utilities/data'
USE_TF_DATASET = False
TF_DATASET_BATCH_SIZE = 1536


def load_cifar(n_classes=10, label_type='fine'):
	"""
	Load the cifar dataset

	Parameters
	----------
	n_classes : int
		number of problem classes

	label_type : str
		label type of the cifar100 dataset


	Returns
	-------
	x_train : np.array
		training instances
	y_train : np.array
		training labels
	x_test : np.array
		testing instances
	x_test : np.array
		testing labels
	"""

	if n_classes == 10:
		(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
	elif n_classes == 100:
		(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode=label_type)

	return x_train, y_train, x_test, y_test


def load_mat(path):
	"""
		Load SVHN mat files

		Parameters
		----------
		path : str
			path to the dataset files

		Returns
		-------
		x : np.array
			instances
		y : np.array
			labels
	"""

	data = scipy.io.loadmat(path)
	x = data['X']
	y = data['y']-1

	x = np.rollaxis(x, 3, 0)
	y = y.reshape(-1)

	return x, y


def load_svhn(dataset_path):
	"""
		Load the SVHN dataset

		Parameters
		----------
		dataset_path : str
			path to the dataset files

		Returns
		-------
		x_train : np.array
			training instances
		y_train : np.array
			training labels
		x_test : np.array
			testing instances
		x_test : np.array
			testing labels
	"""

	try:
		x_train, y_train = load_mat('%s/train_32x32.mat' % dataset_path)
		x_test, y_test = load_mat('%s/test_32x32.mat' % dataset_path)
	except FileNotFoundError:
		print("Error: you need to download the SVHN files first.")
		sys.exit(-1)

	return x_train, y_train, x_test, y_test


class Dataset:
	def __init__(self, dataset_name=None, use_float=False, use_augmentation=False, for_k_fold_validation=False, shape=(32, 32)):
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
			self.n_classes = 10
			self.input_size = (28, 28, 1)

			if dataset_name == 'mnist':
				(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
				if use_augmentation:
					x_train = x_train.reshape((-1, 28, 28, 1))
					x_test = x_test.reshape((-1, 28, 28, 1))
				reshape_data = False

			elif dataset_name == 'fashion-mnist':
				(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
				if use_augmentation:
					x_train = x_train.reshape((-1, 28, 28, 1))
					x_test = x_test.reshape((-1, 28, 28, 1))
				reshape_data = False

			# 255, unbalanced
			elif dataset_name == 'svhn':
				x_train, y_train, x_test, y_test = load_svhn(SVHN_PATH)
				self.input_size = (32, 32, 3)

			# 255, 50000, 10000
			elif dataset_name == 'cifar10':
				x_train, y_train, x_test, y_test = load_cifar(10)
				self.input_size = (32, 32, 3)

			# 255, 50000, 10000
			elif dataset_name == 'cifar100-fine':
				x_train, y_train, x_test, y_test = load_cifar(100, 'fine')
				self.input_size = (32, 32, 3)
				self.n_classes = 100

			elif dataset_name == 'cifar100-coarse':
				x_train, y_train, x_test, y_test = load_cifar(100, 'coarse')
				self.input_size = (32, 32, 3)
				self.n_classes = 20

			elif dataset_name == 'tiny-imagenet':
				x_train, y_train, x_test, y_test = load_tiny_imagenet(TINY_IMAGENET_PATH, shape)
				self.input_size = (32, 32, 3)
				self.n_classes = 200

			else:
				print('Error: the dataset is not valid')
				sys.exit(-1)

			if use_float:
				x_train = x_train.astype(np.float16) / 255.0    # converting from uint8 to float slows training down
				x_test = x_test.astype(np.float16) / 255.0

			log_bold(f"Loaded dataset {dataset_name}: {len(y_train)} training, {len(y_test)} test, shape {self.input_size}, {self.n_classes} classes")
			self.prepare_data(x_train, y_train, x_test, y_test, reshape_data, for_k_fold_validation)

	@classmethod
	def from_data(cls, X_train, y_train, X_test, y_test, X_val, y_val, X_final_test, y_final_test):
		""" init Dataset object from data """
		cls.X_train = X_train
		cls.y_train = y_train
		cls.X_test = X_test
		cls.y_test = y_test
		cls.X_val = X_val
		cls.y_val = y_val
		cls.X_final_test = X_final_test
		cls.y_final_test = y_final_test
		cls.train_dataset_size = X_train.shape[0]
		cls.val_dataset_size = X_val.shape[0]
		return cls

	def prepare_data(self, original_X_train, original_y_train, X_final_test, y_final_test, reshape_data, for_k_fold_validation=False):
		"""
			Split the data into independent sets

			Parameters
			----------
			original_X_train : np.array
				training instances
			original_y_train : np.array
				training labels
			X_final_test : np.array
				testing instances
			y_final_test : np.array
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

		if reshape_data:
			original_X_train = original_X_train.reshape((-1, 32, 32, 3))
			X_final_test = X_final_test.reshape((-1, 32, 32, 3))

		# for batch_size=512, better use test_size=6752, test_size=3680
		X_train, X_val, y_train, y_val = train_test_split(original_X_train, original_y_train, test_size=7000, shuffle=True, stratify=original_y_train)
		X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=3500, shuffle=True, stratify=y_val)

		self.train_dataset_size = X_train.shape[0]
		self.val_dataset_size = X_val.shape[0]

		if USE_TF_DATASET:
			self.train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(TF_DATASET_BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
			self.val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(TF_DATASET_BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
		else:
			self.X_train = X_train
			self.y_train = y_train
			self.X_val = X_val
			self.y_val = y_val

		self.X_test = X_test
		self.y_test = y_test
		self.X_final_test = X_final_test
		self.y_final_test = y_final_test

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
			self.X_combined = original_X_train
			self.y_combined = original_y_train

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
		content, shape = args
		content = content.reshape(-1, 28, 28, 1)

		if shape != (28, 28):
			content = tf.image.resize(content, shape, tf.image.ResizeMethod.NEAREST_NEIGHBOR)

		content = tf.image.grayscale_to_rgb(tf.constant(content))

		return content.numpy()
