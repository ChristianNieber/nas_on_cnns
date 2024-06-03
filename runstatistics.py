from time import time, strftime, gmtime
import json
import numpy as np
import random
from logger import *


# The fitness metric used in experiments
def fitness_metric_with_size_penalty(accuracy, parameters):
	return 2.5625 - (((1.0 - accuracy)/0.02) ** 2 + parameters / 31079.0)


def average_standard_deviation(numlist):
	""" Take the average of a list of standard deviations """
	return np.sqrt(np.sum([i ** 2 for i in numlist]) / len(numlist))


def hms(seconds):
	""" from time given in (fractional) seconds, return string in HH:MM:SS format """
	return strftime("%H:%M:%S", gmtime(seconds))


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
			self.k_fold_million_inferences_time_std = []

			self.statistic_nlayers = []
			self.statistic_variables = []
			self.statistic_floats = []
			self.statistic_ints = []
			self.statistic_cats = []
			self.statistic_variable_mutations = []
			self.statistic_layer_mutations = []

			self.step_width = []

		def metric(self, index):
			""" Read by index 0-accuracy / 1-parameters / 2-fitness value """
			if index == 0:
				return 100.0 - np.array(self.accuracy) * 100.0
			elif index == 1:
				return np.array(self.parameters)
			elif index == 2:
				return np.array(self.fitness)
			elif index == 3:
				return np.array(self.step_width)
			elif index == 4:
				return np.array(self.statistic_nlayers)
			elif index == 5:
				return np.array(self.statistic_variables)
			elif index == 6:
				return 100.0 - np.array(self.final_test_accuracy) * 100.0
			elif index == 7:
				return 100.0 - np.array(self.train_accuracy) * 100.0
			elif index == 8:
				fitness_list = [fitness_metric_with_size_penalty(acc, params) for acc, params in zip(self.final_test_accuracy, self.parameters)]
				return np.array(fitness_list)

		def metric_k_fold(self, index):
			""" Read by index 0-accuracy / 1-parameters / 2-fitness value """
			if index == 0:
				return 100.0 - np.array(self.k_fold_accuracy) * 100.0
			elif index == 1:
				return []
			elif index == 2:
				return np.array(self.k_fold_fitness)

		def metric_k_fold_std(self, index):
			""" Read by index 0-accuracy / 1-parameters / 2-fitness value """
			if index == 0:
				return np.array(self.k_fold_accuracy_std) * 100.0
			elif index == 1:
				return []
			elif index == 2:
				return np.array(self.k_fold_fitness_std)

	def __init__(self, random_seed=-1):
		self.random_seed = random_seed

		# best individual (the best overall for comma strategy)
		self.best = self.IndividualStatistics()
		# best in generation
		self.best_in_gen = self.IndividualStatistics()
		# step widths
		self.stepwidth_stats = []
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
		self.evaluations_total = 0
		self.evaluations_k_folds = 0
		self.evaluations_cache_hits = 0
		self.evaluations_invalid = 0
		self.session_start_time = time()
		self.session_previous_runtime = 0
		self.random_state = random.getstate()
		self.random_state_numpy = np.random.get_state()

	def metric_generation(self, index):
		""" Read by index 0-accuracy / 1-parameters / 2-fitness value """
		if index == 0:
			return 100.0 - np.array(self.generation_accuracy) * 100.0
		elif index == 1:
			return self.generation_parameters
		elif index == 2:
			return np.array(self.generation_fitness)
		else:
			return None

	@staticmethod
	def metric_name(index):
		""" Read by index 0-accuracy / 1-parameters / 2-fitness value """
		if index == 0:
			return "Error Rate"
		elif index == 1:
			return "Number of Parameters"
		elif index == 2:
			return "Fitness"
		elif index == 3:
			return "Step Size"
		elif index == 4:
			return "Number of Layers"
		elif index == 5:
			return "Number of Variables"
		elif index == 6:
			return "Final Test Error Rate"
		elif index == 7:
			return "Training Error Rate"
		elif index == 8:
			return "Final Test Fitness"

	@staticmethod
	def metric_name_lowercase(index):
		""" Read by index 0-accuracy / 1-parameters / 2-fitness value """
		if index == 0:
			return "Error rate"
		elif index == 1:
			return "Number of parameters"
		elif index == 2:
			return "Fitness"
		elif index == 3:
			return "Step size"
		elif index == 4:
			return "Number of layers"
		elif index == 5:
			return "Number of variables"
		elif index == 6:
			return "Final test error rate"
		elif index == 7:
			return "Training error rate"
		elif index == 8:
			return "Final test fitness"

	@staticmethod
	def metric_color(index):
		""" Read by index 0-accuracy / 1-parameters / 2-fitness value """
		if index == 0:
			return "red"
		elif index == 1:
			return "darkviolet"
		elif index == 2:
			return "blue"
		else:
			return "blue"

	@staticmethod
	def metric_ylimits(index):
		if index == 0:
			return (0.0, 8.0)
		elif index == 1:
			return (0, 100000)
		elif index == 2:
			return (-8, 2.3)
		elif index == 3:
			return (0, 0.6)
		elif index == 4:
			return (0, 10)
		elif index == 5:
			return (0, 40)
		elif index == -1:
			return (0.0, 5.0)

	@staticmethod
	def metric_ticks(index):
		if index < 0:
			return 0.5
		elif index == 0:
			return 1.0
		elif index == 1:
			return 10000
		elif index == 2:
			return 1.0
		elif index == 3:
			return 0.05
		elif index == 5:
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

	def record_evaluation(self, seconds=.0, is_cache_hit=False, is_k_folds=False, is_invalid=False):
		if is_invalid:
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

	def log_run_summary(stat):
		log(f"{stat.evaluations_total} evaluations, {stat.evaluations_cache_hits} cache hits, {stat.evaluations_invalid} invalid")
		log(f"runtime {hms(stat.run_time)}, evaluation time {hms(stat.eval_time)} (this run {hms(stat.eval_time_this_run)})")

	def log_statistics_summary(self, start_generation=0):

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



class CnnEvalResult:
	""" results returned by Evaluator.evaluate_cnn """

	def __init__(self, history, final_test_accuracy, eval_time, training_time, final_test_time, million_inferences_time, timer_stop_triggered, early_stop_triggered, parameters, keras_layers, model_layers, accuracy, fitness, model_summary):
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
			self.history_train_accuracy = history['accuracy']
			self.history_train_loss = history['loss']
			self.history_val_accuracy = history['val_accuracy']
			self.history_val_loss = history['val_loss']
			self.training_epochs = len(self.history_train_accuracy)

		if self.training_epochs:
			self.train_accuracy = self.history_train_accuracy[-1]
			self.train_loss = self.history_train_loss[-1]
			self.val_accuracy = self.history_val_accuracy[-1]
			self.val_loss = self.history_val_loss[-1]

	def __descr__(self):
		return self.summary()

	def summary(self, suffix=''):
		return f"p: {self.parameters:6d} acc: {self.accuracy:0.5f} val: {self.val_accuracy:0.5f} final: {self.final_test_accuracy:0.5f} fitness: {self.fitness:0.5f} {'T' if self.timer_stop_triggered else ''}{'E' if self.early_stop_triggered else ''} epochs: {self.training_epochs:2d} t: {self.training_time:0.2f}s{suffix}"

	@staticmethod
	def dummy_eval_result():
		return CnnEvalResult(None, 0, 0, 0, 0, 0, False, False, 0, 0, 0, 0, 0, '')
