# Utility functions for analysing best architectures

import pickle
import glob

# import numpy as np
import pandas as pd
from os import path

from runstatistics import *
from plot_statistics import DEFAULT_EXPERIMENT_PATH, lighten_color
from logger import *
from utils import Evaluator, Individual
from strategy_stepper import StepperGrammar
from engine import evaluate_generation

from scipy.stats import gmean
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

NUM_INDIVIDUALS = 20
LOG_FILE = 'analyse.log'
EXPERIMENTS_PATH = "D:/experiments.NAS_PAPER/"
PICTURE_SAVE_PATH = EXPERIMENTS_PATH + "graphs2/"
EXPERIMENT_LIST = ['Stepper-Adaptive', 'Stepper-Decay', 'Random Search']    # evaluate_cnn currently cannot handle 'F-DENSER' architectures
GRAMMAR_FILE = 'config/lenet.grammar'
# parameters for training
DATASET = 'mnist'
K_FOLDS = 10
TEST_INIT_SEEDS = 0
RANDOM_SEED = 100
MAX_TRAINING_TIME = 100
MAX_TRAINING_EPOCHS = 10


def individual_name_prefix(experiment_name):
	""" extract name prefix A, D, F, R, X from experiment name Stepper-Adaptive, Stepper-Decay, F-DENSER, Random Search, Unknown """
	if experiment_name.startswith('Stepper-'):
		experiment_name = experiment_name[8:]
	return experiment_name[0]


def load_best_individuals(experiment_name, experiments_path=DEFAULT_EXPERIMENT_PATH, over_all_generations=False):
	"""
		load the best individuals of all runs in an experiment
	"""
	if not experiments_path.endswith('/'):
		experiments_path += '/'
	file_list = sorted(glob.glob(experiments_path + experiment_name + ('/r??_best_individuals.pkl' if over_all_generations else '/r??_population.pkl')))
	population = []
	for i, file in enumerate(file_list):
		with open(file, 'rb') as f:
			run_population = pickle.load(f)
			if not over_all_generations:
				run_population = run_population[:1]
			for ind in run_population:
				ind.id = f"{individual_name_prefix(experiment_name)}#{i:02}/{ind.id}"
			population += run_population
# 	if not population:
# 		raise FileNotFoundError(f"No best individuals found in {experiments_path}")
	return population


def init_eval_environment(log_file="analyse.log", dataset=DATASET, use_test_cache=False, max_training_epochs=MAX_TRAINING_EPOCHS, for_k_fold_validation=0, use_float=False, use_augmentation=False, n_gpus=0, batch_size=None):
	if '/' not in log_file and '\\' not in log_file:
		log_file = DEFAULT_EXPERIMENT_PATH + log_file
	init_logger(log_file)
	stat = RunStatistics(RANDOM_SEED)
	grammar = StepperGrammar(GRAMMAR_FILE)
	evaluation_cache_path = DEFAULT_EXPERIMENT_PATH + "cache_test.pkl" if use_test_cache else None
	cnn_eval = Evaluator(dataset, fitness_metric_with_size_penalty, MAX_TRAINING_TIME, max_training_epochs, for_k_fold_validation=for_k_fold_validation,
							evaluation_cache_path=evaluation_cache_path, experiment_name='Analyse Results', use_float=use_float, use_augmentation=use_augmentation, n_gpus=n_gpus, override_batch_size=batch_size)
	return stat, grammar, cnn_eval


def find_best_of_experiment(experiments_path=EXPERIMENTS_PATH, over_all_generations=False):
	best_list = []
	for experiment in EXPERIMENT_LIST:
		best_individuals = load_best_individuals(experiment, experiments_path=experiments_path, over_all_generations=over_all_generations)
		best_list += best_individuals
	return best_list


def select_best_from_population(population, num_individuals=NUM_INDIVIDUALS):
	return sorted(population, key=lambda ind: ind.fitness, reverse=True)[0:num_individuals]


def analyse_learning_statistics_with_lists(population):
	variable_list = ['batch_size', 'lr', 'momentum']
	results_dict = {'batch_size': [], 'lr': [], 'momentum': []}
	stat, grammar, cnn_eval = init_eval_environment()
	for ind in population:
		phenotype = ind.get_phenotype(grammar)
		keras_layers, keras_learning = cnn_eval.construct_keras_layers(phenotype)
		for varname in variable_list:
			value = float(keras_learning[varname])
			results_dict[varname].append(value)
	log(f"statistics over {len(population)} best:")
	for varname in variable_list:
		log(f"{varname}: avg={np.mean(results_dict[varname]):.6f}")


def analyse_learning_statistics(population):
	variable_list = ['batch_size', 'lr', 'momentum']
	stat, grammar, cnn_eval = init_eval_environment()
	dict_list = []
	for i, ind in enumerate(population):
		phenotype = ind.get_phenotype(grammar)
		keras_layers, keras_learning = cnn_eval.construct_keras_layers(phenotype)
		dict_list.append(keras_learning)
	df = pd.DataFrame(dict_list)
	for var in variable_list:
		df[var] = pd.to_numeric(df[var])
	print(df.describe())
	return df


def learning_stats(num_individuals=NUM_INDIVIDUALS, list_population=False, log_file=LOG_FILE):
	init_logger(DEFAULT_EXPERIMENT_PATH + log_file)
	logger_configuration(logger_log_training=True)

	population = select_best_from_population(find_best_of_experiment(), num_individuals)
	if list_population:
		for ind in population:
			log(ind.description())
			log_bold(ind.phenotype_lines[-1])
		# log('\n'.join(ind.phenotype_lines) + '\n')
	return analyse_learning_statistics(population)


class BenchmarkStatistics:
	def __init__(self, title):
		self.title = title
		self.batch_size = []
		self.mean_accuracy = []
		self.std_accuracy = []
		self.mean_final_test_accuracy = []
		self.std_final_test_accuracy = []
		self.mean_accuracy_diff = []
		self.std_accuracy_diff = []
		self.mean_final_accuracy_diff = []
		self.std_final_accuracy_diff = []
		self.time_per_individual = []

	def record_benchmark_statistics(self, accuracy, accuracy_diff, final_test_accuracy, final_accuracy_diff, generation_size, ngenerations, total_eval_time, total_original_eval_time, batch_size):
		mean_accuracy = np.mean(accuracy)
		std_accuracy = np.std(accuracy)
		mean_final_test_accuracy = np.mean(final_test_accuracy)
		std_final_test_accuracy = np.std(final_test_accuracy)
		mean_accuracy_diff = np.mean(accuracy_diff)
		std_accuracy_diff = np.std(accuracy_diff)
		mean_final_accuracy_diff = np.mean(final_accuracy_diff)
		std_final_accuracy_diff = np.std(final_accuracy_diff)
		time_per_individual = total_eval_time / (generation_size * ngenerations)

		self.batch_size.append(batch_size)
		self.mean_accuracy.append(mean_accuracy)
		self.std_accuracy.append(std_accuracy)
		self.mean_final_test_accuracy.append(mean_final_test_accuracy)
		self.std_final_test_accuracy.append(std_final_test_accuracy)
		self.mean_accuracy_diff.append(mean_accuracy_diff)
		self.std_accuracy_diff.append(std_accuracy_diff)
		self.mean_final_accuracy_diff.append(mean_final_accuracy_diff)
		self.std_final_accuracy_diff.append(std_final_accuracy_diff)
		self.time_per_individual.append(time_per_individual)

		description = f"{MAX_TRAINING_EPOCHS} epochs t={total_eval_time:.2f}  Test error: {format_accuracy(mean_accuracy)} ± {std_accuracy:4.2%}  final error: {format_accuracy(mean_final_test_accuracy)} ± {std_final_test_accuracy:4.2%}\n" + \
			f"                     Test diff {mean_accuracy_diff:4.2%} ± {std_accuracy_diff:4.2%}  Final diff {mean_final_accuracy_diff:4.2%} ± {std_final_accuracy_diff:4.2%}"
		log_bold(f"Total eval time: {total_eval_time:.2f} s (original {total_original_eval_time:.2f}, factor {total_eval_time / total_original_eval_time:.3f}), per individual: {time_per_individual :.2f} s")
		log_bold(description)
		return description


def evaluate_best(stat, grammar, cnn_eval, benchmark_stat: BenchmarkStatistics, experiment_name='evaluate_best', generation_size=10, num_individuals=20, plot_history=False):
	population = select_best_from_population(find_best_of_experiment(), num_individuals)
	population_generations = [population[x:x+generation_size] for x in range(0, len(population), generation_size)]
	ngenerations = len(population_generations)

	original_accuracy = np.array([[ind.metrics.accuracy for ind in population_generations[generation]] for generation in range(ngenerations)])
	original_final_test_accuracy = np.array([[ind.metrics.final_test_accuracy for ind in population_generations[generation]] for generation in range(ngenerations)])
	original_fitness = np.array([[ind.metrics.fitness for ind in population_generations[generation]] for generation in range(ngenerations)])
	original_eval_time = np.array([[ind.metrics.eval_time for ind in population_generations[generation]] for generation in range(ngenerations)])

	for generation in range(0, ngenerations):
		generation_start_time = time()

		individuals_list = population_generations[generation]
		evaluate_generation(individuals_list, grammar, cnn_eval, stat, force_reevaluation=True)

		generation_time = time() - generation_start_time
		log_bold(f"Generation evaluation time: {generation_time:.2f}")
		stat.eval_time_of_generation.append(generation_time)

		cnn_eval.flush_evaluation_cache()
		stat.record_generation(individuals_list)

	log_bold(f"\n{experiment_name}\n=============")
	# stat.log_run_summary()

	total_eval_time = 0
	accuracy_diff = []
	final_accuracy_diff = []
	nindividuals = len(stat.generation_accuracy[0])
	for generation in range(0, ngenerations):
		eval_time = stat.eval_time_of_generation[generation]
		total_eval_time += eval_time
		# log_bold(f"Generation {generation}: {eval_time :.2f} sec")
		for idx in range(nindividuals):
			accuracy_diff.append(original_accuracy[generation][idx] - stat.generation_accuracy[generation][idx])
			final_accuracy_diff.append(original_final_test_accuracy[generation][idx] - stat.generation_final_test_accuracy[generation][idx])
			# log(f"{generation}-{idx}:  p: {stat.generation_parameters[generation][idx]:6d} acc: {stat.generation_accuracy[generation][idx]:.5f} ({original_accuracy[generation][idx]:.5f}) final: {stat.generation_final_test_accuracy[generation][idx]:.5f} ({original_final_test_accuracy[generation][idx]:.5f}) fitness: {stat.generation_fitness[generation][idx]:.5f} ({original_fitness[generation][idx]:.5f}) t: {stat.generation_eval_time[generation][idx]:.2f} ({original_eval_time[generation][idx]:.2f})")

	total_original_eval_time = np.sum(original_eval_time)
	accuracy = np.array([[ind.metrics.accuracy for ind in population_generations[generation]] for generation in range(ngenerations)]).flatten()
	final_test_accuracy = np.array([[ind.metrics.final_test_accuracy for ind in population_generations[generation]] for generation in range(ngenerations)]).flatten()
	description = benchmark_stat.record_benchmark_statistics(accuracy, accuracy_diff, final_test_accuracy, final_accuracy_diff, generation_size, ngenerations, total_eval_time, total_original_eval_time, cnn_eval.override_batch_size)

	# df = pd.DataFrame({'orig_accuracy' : original_accuracy.flatten(), 'orig_final' : original_final_test_accuracy.flatten(), 'accuracy' : accuracy.flatten(), 'final_accuracy' : final_test_accuracy.flatten() })
	# print(df.describe())

	if plot_history:
		plot_training_history(population, f"\n{experiment_name}\n{description}", experiment_name)


def benchmark_batch_sizes():
	N_GPUS = 1
	USE_FLOAT = False
	# BATCH_SIZE=1024
	title = f"Benchmark epochs={MAX_TRAINING_EPOCHS} GPUs={N_GPUS} {'float16' if USE_FLOAT else 'uint8'}"
	stat, grammar, cnn_eval = init_eval_environment(log_file=title + ".log", use_test_cache=False, use_float=USE_FLOAT, n_gpus=N_GPUS)

	# warm up GPU with best individual
	best_individual = select_best_from_population(find_best_of_experiment(), 1)[0]
	best_individual.evaluate_individual(grammar, cnn_eval, stat, use_cache=False)

	benchmark_stat = BenchmarkStatistics(title)
	for batch_size in [0, 256, 512, 768, 1024]:
		stat = RunStatistics(RANDOM_SEED)
		cnn_eval.override_batch_size = batch_size
		evaluate_best(stat, grammar, cnn_eval, benchmark_stat,
						title + f" batch size={batch_size}",
						generation_size=10, num_individuals=20,
						plot_history=False)
	with open(EXPERIMENTS_PATH + 'benchmark_statistics.pkl', 'wb') as handle_statistics:
		pickle.dump(benchmark_stat, handle_statistics, protocol=pickle.HIGHEST_PROTOCOL)


def list_best(experiments_path=EXPERIMENTS_PATH, over_all_generations=False):
	population = select_best_from_population(find_best_of_experiment(experiments_path=experiments_path, over_all_generations=over_all_generations), -1)
	log(f"best individuals in {experiments_path}:")
	for ind in population:
		log(ind.description())
		# log('\n'.join(ind.phenotype_lines) + '\n')


def plot_training_history(population: list[Individual], title: str, experiment_name: str = ''):
	nindividuals = len(population)
	ncols = 6
	nrows = (nindividuals + 3) // ncols
	# plot accuracy and validations per epoch
	fig, ax = plt.subplots(nrows * 2, ncols, figsize=(4 * ncols, 5 * nrows + 1))
	for i, ind in enumerate(population):
		row = (i // ncols) * 2
		col = i % ncols
		ax1 = ax[row][col]
		ax2 = ax[row + 1][col]
		ax1.plot(ind.metrics.history_train_accuracy, label="Train accuracy")
		ax1.plot(ind.metrics.history_val_accuracy, label="Validation accuracy")
		ax1.axes.get_xaxis().set_visible(False)
		ax2.plot(ind.metrics.history_train_loss, label="Train loss", color='cyan')
		ax2.plot(ind.metrics.history_val_loss, label="Validation loss", color='magenta')
		if i == 0:
			ax1.legend(fontsize=10)
			ax2.legend(fontsize=10)
		ax1.set_title(f'VLoss= {format_accuracy(ind.metrics.history_val_loss[-1])} VErr= {format_accuracy(ind.metrics.history_val_accuracy[-1])} TErr= {format_accuracy(ind.metrics.accuracy)}', fontsize=12)
		ax1.set_ylim(0.8, 1.0)
		ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.02))
		ax2.set_ylim(0, 0.5)
		ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
		ax1.set_xlim(0, len(ind.metrics.history_train_loss)-1)
		ax2.set_xlim(0, len(ind.metrics.history_train_loss)-1)
		ax1.grid(True)
		ax2.grid(True)
	fig.suptitle(title + '\n', fontsize=28)
	fig.tight_layout()
	if experiment_name:
		plt.savefig(DEFAULT_EXPERIMENT_PATH + experiment_name + '.svg', format='svg', dpi=1200)
	plt.show()


def plot_accuracy_and_std(ax1, xscale, accuracy_list, accuracy_std_list, color, label, invert=False):
	mean_error = np.array(accuracy_list) * 100.0
	if invert:
		mean_error = 100.0 - mean_error
	std = np.array(accuracy_std_list) * 100.0
	ax1.plot(xscale, mean_error, '-', color=color, label=label)
	ax1.fill_between(xscale, mean_error - std, mean_error + std, color=lighten_color(color), alpha=0.5)


def plot_benchmark_results():
	with open(EXPERIMENTS_PATH + 'benchmark_statistics.pkl', 'rb') as handle_statistics:
		benchmark_stat = pickle.load(handle_statistics)
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
	title = benchmark_stat.title
	plt.suptitle(title + '\n ', fontsize=15)
	ax1.set_title('Error rate')
	xscale = benchmark_stat.batch_size
	plot_accuracy_and_std(ax1, xscale, benchmark_stat.mean_accuracy, benchmark_stat.std_accuracy, 'red', 'Error Rate', invert=True)
	plot_accuracy_and_std(ax1, xscale, benchmark_stat.mean_final_test_accuracy, benchmark_stat.std_final_test_accuracy, 'magenta', 'final error rate', invert=True)
	plot_accuracy_and_std(ax1, xscale, benchmark_stat.mean_accuracy_diff, benchmark_stat.std_accuracy_diff, 'blue', 'Error rate diff')
	plot_accuracy_and_std(ax1, xscale, benchmark_stat.mean_final_accuracy_diff, benchmark_stat.std_final_accuracy_diff, 'cyan', 'Final error rate diff')
	ax1.set_xlabel('Batch size')
	ax1.set_ylabel('Error rate')
	ax1.set_ylim(0, 4)
	ax1.legend()
	ax2.set_title('Average Seconds per Evaluation')
	ax2.plot(xscale, benchmark_stat.time_per_individual, '-')
	ax2.set_xlabel('Batch size')
	ax2.set_ylabel('Seconds per Evaluation')
	ax2.set_ylim(0, 14)
	plt.savefig(PICTURE_SAVE_PATH + title + '.svg', format='svg', dpi=1200)
	plt.show()


def reevaluation_description(num_folds=0, num_random_seeds=10, num_epochs=0, use_float=False, batch_size=0, use_augmentation=False):
	return f"{f'folds{num_folds}' if num_folds else f'seeds{num_random_seeds}'}_epochs{num_epochs}{'_float' if use_float else ''}{f'_b{batch_size}' if batch_size else ''}{'_aug' if use_augmentation else ''}"


def reevaluate_best(experiments_path=EXPERIMENTS_PATH, dataset=DATASET, num_individuals=10, over_all_generations=False, num_folds=0, num_random_seeds=10, num_epochs=0, use_float=False, batch_size=0, use_augmentation=False):
	if not experiments_path.endswith('/'):
		experiments_path += '/'
	description = reevaluation_description(num_folds, num_random_seeds, num_epochs, use_float, batch_size, use_augmentation)
	stat, grammar, cnn_eval = init_eval_environment(log_file=experiments_path + description + ".log", dataset=dataset, max_training_epochs=num_epochs, for_k_fold_validation=num_folds, use_test_cache=False, use_float=use_float, use_augmentation=use_augmentation)

	reevaluate_file_name = experiments_path + ('reevaluated_generations.pkl' if over_all_generations else 'reevaluated_individuals.pkl')
	if path.isfile(reevaluate_file_name):
		with open(reevaluate_file_name, 'rb') as handle_pop:
			population = pickle.load(handle_pop)
		log_bold(f"[{description} starts: loaded {len (population)} individuals from {reevaluate_file_name}]")
	else:
		population = select_best_from_population(find_best_of_experiment(experiments_path, over_all_generations=over_all_generations), -1)
		log_bold(f"[{description} starts: found {len (population)} best individuals in {experiments_path}]")

	n_reevaluated = 0
	last = min(len(population), num_individuals)
	for i in range(0, last):
		ind = population[i]
		if not hasattr(ind, 'reevaluation_metrics'):
			ind.reevaluation_metrics = {}
		if description not in ind.reevaluation_metrics:
			if ind.k_fold_metrics and description == 'seeds10_epochs0':
				ind.reevaluation_metrics[description] = ind.k_fold_metrics
			else:
				eval_start_time = time()
				k_fold_metrics = ind.evaluate_individual_k_folds(grammar, cnn_eval, stat, epochs=num_epochs if num_epochs else ind.metrics.training_epochs, num_folds=num_folds, num_random_seeds=num_random_seeds)
				eval_time = time() - eval_start_time
				log_blue(f"{description} {i+1}/{last} K-folds time: {eval_time/max(num_folds, num_random_seeds):.2f} original: {ind.metrics.eval_time:.2f}")
				ind.reevaluation_metrics[description] = k_fold_metrics
				n_reevaluated += 1
			with open(reevaluate_file_name, 'wb') as handle_pop:
				pickle.dump(population, handle_pop, protocol=pickle.HIGHEST_PROTOCOL)
		else:
			log_blue(f"{i+1}/{last} already computed:")
			ind.log_k_folds_result(num_folds, num_random_seeds, ind.reevaluation_metrics[description])


def reevaluation_get_accuracies(population, description: str, sort_by_parameters=False, sort_by_batch_size=False):
	sorted_population = [ind for ind in population if hasattr(ind, 'reevaluation_metrics') and description in ind.reevaluation_metrics]
	if sort_by_batch_size:
		sorted_population = sorted(sorted_population, key=lambda ind: ind.metrics.batch_size)
	elif sort_by_parameters:
		sorted_population = sorted(sorted_population, key=lambda ind: ind.metrics.parameters)

	parameters = []
	batch_size = []
	original_accuracy = []
	original_final_accuracy = []
	accuracy = []
	accuracy_std = []
	accuracy_diff = []
	final_accuracy = []
	final_accuracy_std = []
	final_accuracy_diff = []
	avg_eval_time = 0
	for ind in sorted_population:
		original_accuracy.append(ind.metrics.accuracy)
		original_final_accuracy.append(ind.metrics.final_test_accuracy)
		parameters.append(ind.metrics.parameters)
		batch_size.append(ind.metrics.batch_size if hasattr(ind.metrics, 'batch_size') else -1)     # batch_size not present in old píckles
		k_fold_metrics = ind.reevaluation_metrics[description]
		avg_eval_time += k_fold_metrics.total_eval_time / len(k_fold_metrics.folds_eval_time)
		accuracy.append(k_fold_metrics.accuracy)
		accuracy_std.append(k_fold_metrics.accuracy_std)
		accuracy_diff.append(ind.metrics.accuracy - k_fold_metrics.accuracy)
		final_accuracy.append(k_fold_metrics.final_accuracy)
		final_accuracy_std.append(k_fold_metrics.final_accuracy_std)
		final_accuracy_diff.append(ind.metrics.final_test_accuracy - k_fold_metrics.final_accuracy)
	log(f"{description:24}:{len(parameters):3}  {format_accuracy(np.mean(accuracy))} ± {gmean(accuracy_std):5.2%} (original {format_accuracy(np.mean(original_accuracy))}, diff {np.mean(accuracy_diff):5.2%}), "
		+ f"final {format_accuracy(np.mean(final_accuracy))} ± {gmean(final_accuracy_std):5.2%} (original {format_accuracy(np.mean(original_final_accuracy))}, diff {np.mean(final_accuracy_diff):5.2%})  avg {avg_eval_time/len(parameters):.2f} s")
	return parameters, batch_size, original_accuracy, original_final_accuracy, accuracy, accuracy_std, accuracy_diff, final_accuracy, final_accuracy_std, final_accuracy_diff


def plot_reevaluation_variant(ax, population, description, by_parameters=False, by_batch_size=False):
	parameters, batch_size, original_accuracy, original_final_accuracy, accuracy, accuracy_std, accuracy_diff, final_accuracy, final_accuracy_std, final_accuracy_diff =\
		reevaluation_get_accuracies(population, description, sort_by_parameters=by_parameters, sort_by_batch_size=by_batch_size)

	xscale = batch_size if by_batch_size else (parameters if by_parameters else range(0, len(accuracy)))

	ax.set_title(description + ('by Batch Size' if by_batch_size else (' by Parameters' if by_parameters else ' by Rank')))
	plot_accuracy_and_std(ax, xscale, accuracy, accuracy_std, 'red', 'error Rate', invert=True)
	# plot_accuracy_and_std(ax, xscale, final_accuracy, final_accuracy_std, 'magenta', 'final error rate', invert=True)
	original_accuracy = 100.0 - np.array(original_accuracy) * 100.0
	# original_final_accuracy = 100.0 - np.array(original_final_accuracy) * 100.0
	accuracy_diff = np.array(accuracy_diff) * 100.0
	final_accuracy_diff = np.array(final_accuracy_diff) * 100.0
	ax.plot(xscale, original_accuracy, label='Orig. error', color='blue')
	# ax.plot(xscale, original_final_accuracy, label='Orig. final error', color='cyan')
	ax.plot(xscale, accuracy_diff, label='Diff')
	ax.plot(xscale, final_accuracy_diff, label='Final diff')
	ax.grid()
	ax.set_ylim(-0.5, 20)


def plot_reevaluation_results(experiments_path=EXPERIMENTS_PATH, over_all_generations=False):
	if not experiments_path.endswith('/'):
		experiments_path += '/'
	reevaluate_file_name = experiments_path + ('reevaluated_generations.pkl' if over_all_generations else 'reevaluated_individuals.pkl')
	with open(reevaluate_file_name, 'rb') as handle_pop:
		population = pickle.load(handle_pop)[0:20]
	log_bold(f"[Plot reevaluation: loaded {len(population)} individuals from {reevaluate_file_name}]")

	by_parameters = False
	fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8) = plt.subplots(1, 8, figsize=(40, 3.5))
	plot_reevaluation_variant(ax1, population, reevaluation_description(), by_parameters)
	ax1.legend(fontsize=8)
	plot_reevaluation_variant(ax2, population, reevaluation_description(num_folds=10), by_parameters)
	plot_reevaluation_variant(ax3, population, reevaluation_description(num_epochs=10), by_parameters)
	plot_reevaluation_variant(ax4, population, reevaluation_description(use_float=True), by_parameters)
	plot_reevaluation_variant(ax5, population, reevaluation_description(batch_size=512), by_parameters)
	plot_reevaluation_variant(ax6, population, reevaluation_description(batch_size=1024), by_parameters)
	plot_reevaluation_variant(ax7, population, reevaluation_description(batch_size=1536), by_parameters)
	plot_reevaluation_variant(ax8, population, reevaluation_description(num_epochs=30), by_parameters)
	# plot_reevaluation_variant(ax3, population, reevaluation_description(batch_size=384), by_parameters)
	# plot_reevaluation_variant(ax5, population, reevaluation_description(batch_size=768), by_parameters)

	# plot_reevaluation_variant(ax11, population, reevaluation_description(), by_parameters)
	# plot_reevaluation_variant(ax12, population, reevaluation_description(use_float=True), by_parameters)
	# plot_reevaluation_variant(ax13, population, reevaluation_description(num_epochs=30), by_parameters)
	# plot_reevaluation_variant(ax14, population, reevaluation_description(num_epochs=30, use_float=True), by_parameters)
	# plot_reevaluation_variant(ax15, population, reevaluation_description(use_float=False), by_parameters)
	# plot_reevaluation_variant(ax16, population, reevaluation_description(use_float=True), by_parameters)
	plt.show()


if __name__ == "__main__":
	n_individuals = 20
	EXPERIMENTS_PATH = 'D:/experiments.fashion3'
	DATASET = 'fashion-mnist'
	all_generations = True
	plot_reevaluation_results(EXPERIMENTS_PATH, over_all_generations=True)
	reevaluate_best(EXPERIMENTS_PATH, DATASET, num_individuals=n_individuals, num_epochs=30, over_all_generations=all_generations)
	# reevaluate_best(EXPERIMENTS_PATH, DATASET, num_individuals=n_individuals, over_all_generations=over_all_generations)
	# reevaluate_best(EXPERIMENTS_PATH, DATASET, num_individuals=n_individuals, num_folds=10, num_random_seeds=0, over_all_generations=all_generations)
	# reevaluate_best(EXPERIMENTS_PATH, DATASET, num_individuals=n_individuals, num_epochs=10, over_all_generations=all_generations)
	# reevaluate_best(EXPERIMENTS_PATH, DATASET, num_individuals=n_individuals, use_float=True, over_all_generations=all_generations)
	# reevaluate_best(EXPERIMENTS_PATH, DATASET, num_individuals=n_individuals, batch_size=512, over_all_generations=all_generations)
	# reevaluate_best(EXPERIMENTS_PATH, DATASET, num_individuals=n_individuals, batch_size=1024, over_all_generations=all_generations)
	# reevaluate_best(EXPERIMENTS_PATH, DATASET, num_individuals=n_individuals, batch_size=1536, over_all_generations=all_generations)
	# reevaluate_best(EXPERIMENTS_PATH, DATASET, num_individuals=n_individuals, batch_size=256)
	# reevaluate_best(EXPERIMENTS_PATH, DATASET, num_individuals=n_individuals, batch_size=384)
	# reevaluate_best(EXPERIMENTS_PATH, DATASET, num_individuals=n_individuals, batch_size=768)

	# reevaluate_best(EXPERIMENTS_PATH, DATASET, num_individuals=n_individuals, num_folds=10, use_float=True)
	# reevaluate_best(EXPERIMENTS_PATH, DATASET, num_individuals=n_individuals, use_float=True)
	# reevaluate_best(EXPERIMENTS_PATH, DATASET, num_individuals=n_individuals, batch_size=256, use_augmentation=True)
	# list_best("D:/experiments.fashion3", over_all_generations=True)
	# benchmark_batch_sizes()
	# plot_benchmark_results()
