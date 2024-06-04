# Utility functions for analysing best architectures

import pickle
import glob
import pandas as pd

from runstatistics import *
from plot_statistics import DEFAULT_EXPERIMENT_PATH
from logger import *
from utils import Evaluator, Individual
from strategy_stepper import StepperGrammar
from engine import evaluate_generation
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

NUM_INDIVIDUALS = 20
LOG_FILE = 'analyse.log'
EXPERIMENT_LIST = ['Stepper-Adaptive', 'Stepper-Decay', 'Random Search']
GRAMMAR_FILE = 'config/lenet.grammar'
# parameters for training
DATASET = 'mnist'
K_FOLDS = 10
TEST_INIT_SEEDS = 0
RANDOM_SEED = 100
MAX_TRAINING_TIME = 50
MAX_TRAINING_EPOCHS = 10


def load_best_individuals(experiment_name, experiment_path=DEFAULT_EXPERIMENT_PATH):
	""" load the best individuals of all runs in an experiment """
	file_list = sorted(glob.glob(experiment_path + experiment_name + "/r??_best_parent.pkl"))
	population = []
	for file in file_list:
		with open(file, 'rb') as f:
			population.append(pickle.load(f))
	if not population:
		raise FileNotFoundError(f"No best individuals found in {experiment_path}")
	return population


def init_eval_environment(log_file="analyse.log", use_test_cache=False, use_float=False, n_gpus=0, batch_size=None):
	init_logger(DEFAULT_EXPERIMENT_PATH + log_file)
	stat = RunStatistics(RANDOM_SEED)
	grammar = StepperGrammar(GRAMMAR_FILE)
	evaluation_cache_path = DEFAULT_EXPERIMENT_PATH + "cache_test.pkl" if use_test_cache else None
	cnn_eval = Evaluator(DATASET, fitness_metric_with_size_penalty, MAX_TRAINING_TIME, MAX_TRAINING_EPOCHS, for_k_fold_validation=0, calculate_fitness_with_k_folds_accuracy=False, test_init_seeds=TEST_INIT_SEEDS,
							evaluation_cache_path=evaluation_cache_path, experiment_name='Analyse Results', use_float=use_float, n_gpus=n_gpus, batch_size=batch_size)
	return stat, grammar, cnn_eval


def get_population():
	best_list = []
	for experiment in EXPERIMENT_LIST:
		best_individuals = load_best_individuals(experiment)
		best_list += best_individuals
	# for ind in population:
	# 	log_bold(ind.description())
	# 	log('\n'.join(ind.phenotype_lines) + '\n')
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

	population = select_best_from_population(get_population(), num_individuals)
	if list_population:
		for ind in population:
			log(ind.description())
			log_bold(ind.phenotype_lines[-1])
		# log('\n'.join(ind.phenotype_lines) + '\n')
	return analyse_learning_statistics(population)


def evaluate_k_folds(log_file='K-folds.log'):
	stat, grammar, cnn_eval = init_eval_environment(log_file=log_file)
	population = select_best_from_population(get_population(), 10)
	start_time = time()
	for ind in population:
		eval_start_time = time()
		ind.evaluate_individual_k_folds(grammar, cnn_eval, K_FOLDS, stat)
		eval_time = time() - eval_start_time
		log_blue(f"K-folds time: {eval_time/K_FOLDS:.2f} original: {ind.metrics.eval_time:.2f}")
	run_time = time() - start_time
	log_blue(f"{stat.evaluations_total} evaluations, " + f" {stat.evaluations_k_folds} for k-folds")
	log_blue(f"runtime {run_time:.2f}, evaluation time {stat.eval_time_k_folds:.2f}")

	if len(stat.k_fold_accuracy_stds) > 1:
		log(f"Average folds accuracy SD: {average_standard_deviation(stat.k_fold_accuracy_stds):.5f} {stat.k_fold_accuracy_stds}")
		log(f"Average folds final accuracy SD: {average_standard_deviation(stat.k_fold_final_accuracy_stds):.5f} {stat.k_fold_final_accuracy_stds}")
		log(f"Average folds fitness SD: {average_standard_deviation(stat.k_fold_fitness_stds):.5f} {stat.k_fold_fitness_stds}")


def err(accuracy):
	return f"{(1.0 - accuracy) * 100.0:.2f}"

def percent(diff):
	return f"{diff * 100.0:.2f}"

def evaluate_best(experiment_name='evaluate_best', generation_size=12, num_individuals=24, use_float=False, batch_size=None, n_gpus=0, use_test_cache=False, plot_history=False):
	population = select_best_from_population(get_population(), num_individuals)
	population_generations = x = [ population[x:x+generation_size] for x in range(0, len(population), generation_size) ]
	ngenerations = len(population_generations)

	original_accuracy = np.array([[ind.metrics.accuracy for ind in population_generations[generation]] for generation in range(ngenerations)])
	original_final_test_accuracy = np.array([[ind.metrics.final_test_accuracy for ind in population_generations[generation]] for generation in range(ngenerations)])
	original_fitness = np.array([[ind.metrics.fitness for ind in population_generations[generation]] for generation in range(ngenerations)])
	original_eval_time = np.array([[ind.metrics.eval_time for ind in population_generations[generation]] for generation in range(ngenerations)])

	stat, grammar, cnn_eval = init_eval_environment(log_file=experiment_name + ".log", use_test_cache=use_test_cache, use_float=use_float, n_gpus=n_gpus, batch_size=batch_size)

	for generation in range(0, ngenerations):
		individuals_list = population_generations[generation]
		evaluate_generation(individuals_list, grammar, cnn_eval, stat, force_reevaluation=True)
		cnn_eval.flush_evaluation_cache()
		stat.record_generation(individuals_list)

	title = f"\n{n_gpus} GPUS, {'float16' if use_float else 'uint8'}, {batch_size=}"
	log_bold(f"{title}\n=============")
	stat.log_run_summary()

	total_eval_time = 0
	accuracy_diff = []
	final_accuracy_diff = []
	ngenerations = len(stat.eval_time_of_generation)
	nindividuals = len(stat.generation_accuracy[0])
	for generation in range(0, ngenerations):
		eval_time = stat.eval_time_of_generation[generation]
		total_eval_time += eval_time
		log_bold(f"Generation {generation}: {eval_time :.2f} sec")
		for idx in range(nindividuals):
			accuracy_diff.append(original_accuracy[generation][idx] - stat.generation_accuracy[generation][idx])
			final_accuracy_diff.append(original_final_test_accuracy[generation][idx] - stat.generation_final_test_accuracy[generation][idx])
			log(f"{generation}-{idx}:  p: {stat.generation_parameters[generation][idx]:6d} acc: {stat.generation_accuracy[generation][idx]:.5f} ({original_accuracy[generation][idx]:.5f}) final: {stat.generation_final_test_accuracy[generation][idx]:.5f} ({original_final_test_accuracy[generation][idx]:.5f}) fitness: {stat.generation_fitness[generation][idx]:.5f} ({original_fitness[generation][idx]:.5f}) t: {stat.generation_eval_time[generation][idx]:.2f} ({original_eval_time[generation][idx]:.2f})")

	total_original_eval_time = np.sum(original_eval_time)
	accuracy = np.array([[ind.metrics.accuracy for ind in population_generations[generation]] for generation in range(ngenerations)]).flatten()
	final_test_accuracy = np.array([[ind.metrics.final_test_accuracy for ind in population_generations[generation]] for generation in range(ngenerations)]).flatten()
	description = f"{MAX_TRAINING_EPOCHS} epochs t={total_eval_time:.2f}  Test error: {err(np.mean(accuracy))}±{percent(np.std(accuracy))}  final error: {err(np.mean(final_test_accuracy))}±{percent(np.std(final_test_accuracy))}\n\
                     Test diff {percent(np.mean(accuracy_diff))}±{percent(np.std(accuracy_diff))}  Final diff {percent(np.mean(final_accuracy_diff))}±{percent(np.std(final_accuracy_diff))}"
	log_bold(f"Total eval time: {total_eval_time:.2f} s (original {total_original_eval_time:.2f}, factor {total_eval_time/total_original_eval_time:.3f})")
	log_bold(description)

	# df = pd.DataFrame({'orig_accuracy' : original_accuracy.flatten(), 'orig_final' : original_final_test_accuracy.flatten(), 'accuracy' : accuracy.flatten(), 'final_accuracy' : final_test_accuracy.flatten() })
	# print(df.describe())

	if plot_history:
		plot_training_history(population, f"{title}\n{description}", experiment_name)


def plot_training_history(population: list[Individual], title: str, experiment_name: str=''):
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
		ax1.set_title(f'VLoss= {percent(ind.metrics.history_val_loss[-1])} VErr= {err(ind.metrics.history_val_accuracy[-1])} TErr= {err(ind.metrics.accuracy)}', fontsize=12)
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

if __name__ == "__main__":
	N_GPUS=4
	BATCH_SIZE=512
	USE_FLOAT=False
	evaluate_best(f"Evaluate best e{MAX_TRAINING_EPOCHS} g{N_GPUS} b{BATCH_SIZE} {'float16' if USE_FLOAT else 'uint8'}", use_float=USE_FLOAT, batch_size=BATCH_SIZE, n_gpus=N_GPUS, use_test_cache=False, plot_history=True)
