from time import time
import json
import numpy as np
import random
from enum import IntEnum
from logger import *

class Metric(IntEnum):
	NONE = -1
	ERROR_RATE = 0
	PARAMETERS = 1
	FITNESS = 2
	STEP_SIZE = 3
	NLAYERS = 4
	NVARIABLES = 5
	FINAL_TEST_ERROR_RATE = 6
	TRAINING_ERROR_RATE = 7
	FINAL_TEST_FITNESS = 8


# The fitness metric used in experiments
def fitness_metric_with_size_penalty(accuracy, parameters):
	return 2.5625 - (((1.0 - accuracy)/0.02) ** 2 + parameters / 31079.0)


def average_standard_deviation(numlist):
	""" Take the average of a list of standard deviations """
	return np.sqrt(np.sum([i ** 2 for i in numlist]) / len(numlist))


def decimal_hours(seconds):
	""" from time given in (fractional) seconds, return string in <hours>h:MM:SS format """
	return f"{seconds/3600:.2f} h"


class RunStatistics:
	""" keeps statistics over all generations """

	class IndividualStatistics:
		def __init__(self):
			self.id = []
			self.accuracy = []
			self.parameters = []
			self.fitness = []

			self.evaluation_time = []
			self.training_time = []
			self.training_epochs = []
			self.million_inferences_time = []

			self.final_test_accuracy = []
			self.train_accuracy = []
			self.train_loss = []
			self.val_accuracy = []
			self.val_loss = []

			self.k_fold_accuracy = []
			self.k_fold_accuracy_std = []
			self.k_fold_fitness = []
			self.k_fold_fitness_std = []
			self.k_fold_final_accuracy = []
			self.k_fold_final_accuracy_std = []
			self.k_fold_million_inferences_time = []

			self.statistic_nlayers = []
			self.statistic_variables = []
			self.statistic_floats = []
			self.statistic_ints = []
			self.statistic_cats = []
			self.statistic_variable_mutations = []
			self.statistic_layer_mutations = []

			self.step_width = []

		def metric(self, m : Metric):
			""" Read by index 0-accuracy / 1-parameters / 2-fitness value """
			if m == Metric.ERROR_RATE:
				return 100.0 - np.array(self.accuracy) * 100.0
			elif m == Metric.PARAMETERS:
				return np.array(self.parameters)
			elif m == Metric.FITNESS:
				return np.array(self.fitness)
			elif m == Metric.STEP_SIZE:
				return np.array(self.step_width)
			elif m == Metric.NLAYERS:
				return np.array(self.statistic_nlayers)
			elif m == Metric.NVARIABLES:
				return np.array(self.statistic_variables)
			elif m == Metric.FINAL_TEST_ERROR_RATE:
				return 100.0 - np.array(self.final_test_accuracy) * 100.0
			elif m == Metric.TRAINING_ERROR_RATE:
				return 100.0 - np.array(self.train_accuracy) * 100.0
			elif m == Metric.FINAL_TEST_FITNESS:
				return np.array([fitness_metric_with_size_penalty(acc, params) for acc, params in zip(self.final_test_accuracy, self.parameters)])

		def metric_k_fold(self, m : Metric):
			""" Read by index 0-accuracy / 1-parameters / 2-fitness value """
			if m == Metric.ERROR_RATE:
				return 100.0 - np.array(self.k_fold_accuracy) * 100.0
			elif m == Metric.PARAMETERS:
				return []
			elif m == Metric.FITNESS:
				return np.array(self.k_fold_fitness)

		def metric_k_fold_std(self, m : Metric):
			""" Read by index 0-accuracy / 1-parameters / 2-fitness value """
			if m == Metric.ERROR_RATE:
				return np.array(self.k_fold_accuracy_std) * 100.0
			elif m == Metric.PARAMETERS:
				return []
			elif m == Metric.FITNESS:
				return np.array(self.k_fold_fitness_std)

	def __init__(self, random_seed=-1, run_nas_strategy='', run_number=-1, run_dataset=''):
		self.random_seed = random_seed

		# Data only recorded for displaying in statistics later
		self.run_nas_strategy = run_nas_strategy
		self.run_number = run_number
		self.run_dataset = run_dataset

		# best individual (the best overall for comma strategy)
		self.best = self.IndividualStatistics()
		# best in generation
		self.best_in_gen = self.IndividualStatistics()
		# k folds standard deviations
		self.k_fold_accuracy_stds = []
		self.k_fold_final_accuracy_stds = []
		self.k_fold_fitness_stds = [0]

		# best of generation
		self.generation_best_accuracy = []
		self.generation_best_parameters = []
		self.generation_best_fitness = []
		# generation
		self.generation_accuracy = []
		self.generation_final_test_accuracy = []
		self.generation_parameters = []
		self.generation_fitness = []
		self.generation_eval_time = []
		self.eval_time_of_generation = []
		# run state
		self.run_generation = -1
		self.run_time = 0
		self.eval_time = 0
		self.eval_time_this_run = 0
		self.eval_time_k_folds = 0
		self.eval_time_k_folds_this_run = 0
		self.evaluations_total = 0              # evaluations excluding invalid evaluations
		self.evaluations_k_folds = 0            # evaluations for folds
		self.evaluations_cache_hits = 0
		self.evaluations_invalid = 0            # usually means keras error during model construction
		self.evaluations_constraints_violated = 0
		self.session_start_time = time()
		self.session_previous_runtime = 0
		self.random_state = random.getstate()
		self.random_state_numpy = np.random.get_state()

	def metric_generation(self, m : Metric):
		""" Read by index 0-accuracy / 1-parameters / 2-fitness value """
		if m == Metric.ERROR_RATE:
			return 100.0 - np.array(self.generation_accuracy) * 100.0
		elif m == Metric.PARAMETERS:
			return self.generation_parameters
		elif m == Metric.FITNESS:
			return np.array(self.generation_fitness)
		else:
			return None

	@staticmethod
	def metric_name(m : Metric):
		""" Read by index 0-accuracy / 1-parameters / 2-fitness value """
		metric_names = ["Error Rate",
						"Number of Parameters",
						"Fitness",
						"Step Size",
						"Number of Layers",
						"Number of Variables",
						"Final Test Error Rate",
						"Training Error Rate",
						"Final Test Fitness",
						]
		return metric_names[int(m)]

	@staticmethod
	def metric_name_lowercase(m : Metric):
		""" Read by index 0-accuracy / 1-parameters / 2-fitness value """
		metric_names_lowercase = ["Error Rate",
						"Number of parameters",
						"Fitness",
						"Step size",
						"Number of layers",
						"Number of variables",
						"Final test error rate",
						"Training error rate",
						"Final test fitness",
						]
		return metric_names_lowercase[int(m)]

	@staticmethod
	def metric_color(m : Metric):
		""" Read by index 0-accuracy / 1-parameters / 2-fitness value """
		if m == Metric.ERROR_RATE:
			return "red"
		elif m == Metric.PARAMETERS:
			return "darkviolet"
		elif m == Metric.FITNESS:
			return "blue"
		else:
			return "blue"

	@staticmethod
	def metric_ylimits(m : Metric):
		if m == Metric.ERROR_RATE:
			return 0.0, 8.0
		elif m == Metric.PARAMETERS:
			return 0, 100000
		elif m == Metric.FITNESS:
			return -8, 2.3
		elif m == Metric.STEP_SIZE:
			return 0, 0.6
		elif m == Metric.NLAYERS:
			return 0, 10
		elif m == Metric.NVARIABLES:
			return 0, 40
		elif m == Metric.NONE:
			return 0.0, 5.0

	@staticmethod
	def metric_ticks(m : Metric):
		if int(m) < 0:
			return 0.5
		elif m == Metric.ERROR_RATE:
			return 1.0
		elif m == Metric.PARAMETERS:
			return 10000
		elif m == Metric.FITNESS:
			return 1.0
		elif m == Metric.STEP_SIZE:
			return 0.05
		elif m == Metric.NVARIABLES:
			return 2
		else:
			return 1

	def init_session(self):
		self.session_start_time = time()
		self.session_previous_runtime = self.run_time

	def record_generation(self, generation_list):
		self.run_generation += 1
		best_in_generation_idx = np.argmax([ind.fitness for ind in generation_list])
		best_in_generation = generation_list[best_in_generation_idx]
		self.generation_best_fitness.append(best_in_generation.fitness)
		self.generation_best_accuracy.append(best_in_generation.metrics.accuracy)
		self.generation_best_parameters.append(best_in_generation.metrics.parameters)
		self.generation_accuracy.append([ind.metrics.accuracy for ind in generation_list])
		self.generation_final_test_accuracy.append([ind.metrics.final_test_accuracy for ind in generation_list])
		self.generation_fitness.append([ind.fitness for ind in generation_list])
		self.generation_parameters.append([ind.metrics.parameters for ind in generation_list])
		self.generation_eval_time.append([ind.metrics.eval_time for ind in generation_list])
		self.run_time = self.session_previous_runtime + time() - self.session_start_time

	def record_evaluation(self, seconds=.0, is_cache_hit=False, is_k_folds=False, is_invalid=False, constraints_violated=False):
		if constraints_violated:
			self.evaluations_constraints_violated += 1
		elif is_invalid:
			self.evaluations_invalid += 1
		else:
			self.evaluations_total += 1
			self.eval_time += seconds
			if is_cache_hit:
				self.evaluations_cache_hits += 1
			else:
				self.eval_time_this_run += seconds
			if is_k_folds:
				self.evaluations_k_folds += 1
				self.eval_time_k_folds += seconds
				if not is_cache_hit:
					self.eval_time_k_folds_this_run += seconds

	def to_json(self):
		""" makes object json serializable """
		return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

	def save_to_json_file(self, save_path):
		""" save statistics to json - currently unused """
		json_dump = self.to_json()
		with open(save_path + 'statistics.json', 'w') as f_json:
			f_json.write(json_dump)

	def log_run_summary(self):
		log(f"{self.evaluations_total} evaluations, {self.evaluations_cache_hits} cache hits, {self.evaluations_invalid} invalid,  avg evaluation time: {self.eval_time / self.evaluations_total:.2f} s")
		log(f"runtime {decimal_hours(self.run_time)}, evaluation time {decimal_hours(self.eval_time)} (this run {decimal_hours(self.eval_time_this_run)})")

	def log_statistics_summary(self, start_generation=0):
		""" log a summary of statistics listing all generations """
		self.log_run_summary()
		total_eval_time = 0
		ngenerations = len(self.eval_time_of_generation)
		nindividuals = len(self.generation_accuracy[0])
		for generation in range(start_generation, ngenerations):
			eval_time = self.eval_time_of_generation[generation]
			total_eval_time += eval_time
			log_bold(f"Generation {generation}: {eval_time :.2f} sec")
			for idx in range(nindividuals):
				log(f"{generation}-{idx}:  p: {self.generation_parameters[generation][idx]:6d} acc: {self.generation_accuracy[generation][idx]:.5f} fitness: {self.generation_fitness[generation][idx]:.5f} t: {self.generation_eval_time[generation][idx]:.2f}")

		log_bold(f"Total eval time: {total_eval_time:.2f} sec")


def format_accuracy(acc):
	return f"{1.0-acc:6.2%}"


def format_fitness(value):
	if value > 0:
		return f"{value:+6.2f}"
	elif value > -1000.0:
		return f"{value:+6.1f}"
	else:
		return f"{value:+6.0f}"


class CnnEvalResult:
	""" results returned by Evaluator.evaluate_cnn """

	def __init__(self, history, final_test_accuracy, eval_time, training_time, final_test_time, million_inferences_time, timer_stop_triggered, early_stop_triggered, parameters, keras_layers, model_layers, accuracy, fitness, batch_size, model_summary):
		self.final_test_accuracy = final_test_accuracy
		self.eval_time = eval_time
		self.training_time = training_time
		self.final_test_time = final_test_time
		self.million_inferences_time = million_inferences_time
		self.timer_stop_triggered = timer_stop_triggered
		self.early_stop_triggered = early_stop_triggered
		self.parameters = parameters
		self.keras_layers = keras_layers
		self.model_layers = model_layers
		self.accuracy = accuracy
		self.fitness = fitness
		self.batch_size = batch_size
		self.model_summary = model_summary
		self.train_accuracy = 0
		self.train_loss = 0
		self.val_accuracy = 0
		self.val_loss = 0
		self.history_train_accuracy = []
		self.history_train_loss = []
		self.history_val_accuracy = []
		self.history_val_loss = []
		self.training_epochs = 0
		if history:
			self.history_train_accuracy = np.array(history['accuracy'], dtype=np.float32)
			self.history_train_loss = np.array(history['loss'], dtype=np.float32)
			self.history_val_accuracy = np.array(history['val_accuracy'], dtype=np.float32)
			self.history_val_loss = np.array(history['val_loss'], dtype=np.float32)
			self.training_epochs = len(self.history_train_accuracy)

		if self.training_epochs:
			self.train_accuracy = self.history_train_accuracy[-1]
			self.train_loss = self.history_train_loss[-1]
			self.val_accuracy = self.history_val_accuracy[-1]
			self.val_loss = self.history_val_loss[-1]

	def __descr__(self):
		return self.summary()

	def summary(self, suffix=''):
		return f"{format_fitness(self.fitness)} {(self.parameters/1000.0):6.1f}k err:{format_accuracy(self.accuracy)} val:{format_accuracy(self.val_accuracy)} final:{format_accuracy(self.final_test_accuracy)} ep: {self.training_epochs:2d} t: {self.training_time:0.1f}{f' b: {self.batch_size}' if hasattr(self, 'batch_size') else ''}{' (T)' if self.timer_stop_triggered else ''}{' (E)' if self.early_stop_triggered else ''} {suffix}"

	@staticmethod
	def dummy_eval_result():
		return CnnEvalResult(None, 0, 0, 0, 0, 0, False, False, 0, 0, 0, 0, 0, 0, '')
