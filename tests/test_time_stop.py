import unittest
import warnings
import time


class Model:
	def __init__(self):
		self.stop_training = False


class Test(unittest.TestCase):
	def setUp(self):
		warnings.simplefilter('ignore', category=DeprecationWarning)

	def test_time_stop(self):
		import utils as utils

		time_stop = utils.TimedStopping(seconds=2, verbose=True)
		time_stop.model = Model()

		time_stop.on_train_begin()

		time.sleep(1)

		self.assertEqual(time_stop.model.stop_training, False, "Error stop training")

		time.sleep(1)

		time_stop.on_epoch_end(epoch=1)

		self.assertEqual(time_stop.model.stop_training, True, "Error stop training")


if __name__ == '__main__':
	unittest.main()