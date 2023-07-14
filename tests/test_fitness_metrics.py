import unittest
import warnings


class Test(unittest.TestCase):
	def setUp(self):
		warnings.simplefilter('ignore', category=DeprecationWarning)

	def test_calculate_accuracy(self):
		from utils import calculate_accuracy

		self.assertEqual(calculate_accuracy([0, 1], [[0, 1], [0.5, 1]]), 0.5, "Error: accuracy is wrong")


if __name__ == '__main__':
	unittest.main()
