from time import time
import json
import numpy as np


class RunStatistics:
	""" keeps statistics over all generations """

	class IndividualStatistics:
		def __init__(self):
			self.id = []
			self.final_test_accuracy = []
			self.accuracy = []
			self.parameters = []
			self.evaluation_time = []
			self.training_time = []
			self.training_epochs = []
			self.million_inferences_time = []
			self.fitness = []
			self.train_accuracy = []
			self.train_loss = []
			self.val_accuracy = []
			self.val_loss = []
			self.k_fold_accuracy = []
			self.k_fold_accuracy_std = []
			self.k_fold_final_accuracy = []
			self.k_fold_final_accuracy_std = []
			self.k_fold_fitness = []
			self.k_fold_fitness_std = []
			self.k_fold_million_inferences_time = []
			self.k_fold_million_inferences_time_std = []

	def __init__(self):
		# best individual (the best overall for comma strategy)
		self.best = self.IndividualStatistics()
		# best in generation
		self.best_in_gen = self.IndividualStatistics()
		# step widths
		self.stepwidth_stats = []

		# best of generation
		self.generation_best_accuracy = []
		self.generation_best_fitness = []
		self.generation_best_parameters = []
		# generation
		self.generation_accuracy = []
		self.generation_fitness = []
		self.generation_parameters = []
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
		self.generation_fitness.append([ind.fitness for ind in generation_list])
		self.generation_parameters.append([ind.metrics.parameters for ind in generation_list])
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
		""" save statistics """
		json_dump = self.to_json()
		with open(save_path + 'statistics.json', 'w') as f_json:
			f_json.write(json_dump)


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

	def summary(self):
		return f"p: {self.parameters:6d} acc: {self.accuracy:0.5f} val: {self.val_accuracy:0.5f} final: {self.final_test_accuracy:0.5f} fitness: {self.fitness:0.5f} {'T' if self.timer_stop_triggered else ''}{'E' if self.early_stop_triggered else ''} epochs: {self.training_epochs:2d} t: {self.training_time:0.2f}s"

	@staticmethod
	def dummy_eval_result():
		return CnnEvalResult(None, 0, 0, 0, 0, 0, False, False, 0, 0, 0, 0, 0, '')
