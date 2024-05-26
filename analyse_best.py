# Utility functions for analysing best architectures

import pickle
import glob
import pandas as pd

from runstatistics import *
from plot_statistics import DEFAULT_EXPERIMENT_PATH
from logger import *
from utils import Evaluator
from strategy_stepper import StepperGrammar

NUM_INDIVIDUALS = 20
LOG_FILE = 'analyse.log'
EXPERIMENT_LIST = ['Stepper-Adaptive', 'Stepper-Decay', 'Random Search']
GRAMMAR_FILE = 'config/lenet.grammar'
# parameters for training
DATASET = 'mnist'
K_FOLDS = 10
TEST_INIT_SEEDS = 0
RANDOM_SEED = 100
MAX_TRAINING_TIME = 10
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


def init_eval_environment():
	stat = RunStatistics(RANDOM_SEED)
	grammar = StepperGrammar(GRAMMAR_FILE)
	cnn_eval = Evaluator(DATASET, fitness_metric_with_size_penalty, for_k_fold_validation=K_FOLDS, calculate_fitness_with_k_folds_accuracy=False, test_init_seeds=TEST_INIT_SEEDS,
							evaluation_cache_path=None, experiment_name='Analyse Results')
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


def evaluate_k_folds(population):
	stat, grammar, cnn_eval = init_eval_environment()
	start_time = time()
	for ind in population:
		eval_start_time = time()
		ind.evaluate_individual_k_folds(grammar, cnn_eval, K_FOLDS, stat, MAX_TRAINING_TIME, MAX_TRAINING_EPOCHS)
		eval_time = time() - eval_start_time
		log_blue(f"K-folds time: {eval_time/K_FOLDS:.2f} original: {ind.metrics.eval_time:.2f}")
	run_time = time() - start_time
	log_blue(f"{stat.evaluations_total} evaluations, " + f" {stat.evaluations_k_folds} for k-folds")
	log_blue(f"runtime {run_time:.2f}, evaluation time {stat.eval_time_k_folds:.2f}")

	if len(stat.k_fold_accuracy_stds) > 1:
		log(f"Average folds accuracy SD: {average_standard_deviation(stat.k_fold_accuracy_stds):.5f} {stat.k_fold_accuracy_stds}")
		log(f"Average folds final accuracy SD: {average_standard_deviation(stat.k_fold_final_accuracy_stds):.5f} {stat.k_fold_final_accuracy_stds}")
		log(f"Average folds fitness SD: {average_standard_deviation(stat.k_fold_fitness_stds):.5f} {stat.k_fold_fitness_stds}")


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
	log_file_path = DEFAULT_EXPERIMENT_PATH + log_file
	init_logger(log_file_path)
	logger_configuration(logger_log_training=True)

	population = select_best_from_population(get_population(), num_individuals)
	if list_population:
		for ind in population:
			log(ind.description())
			log_bold(ind.phenotype_lines[-1])
		# log('\n'.join(ind.phenotype_lines) + '\n')
	return analyse_learning_statistics(population)


if __name__ == "__main__":
	learning_stats()
