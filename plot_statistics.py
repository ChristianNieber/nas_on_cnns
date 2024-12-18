import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mc
import pickle
import glob
import pandas as pd
import colorsys
import scipy
import platform
from pathlib import Path

from runstatistics import Metric, RunStatistics, decimal_hours

# plot colors and alpha channel
ALPHA_LINES = 0.3
COLOR_DOTS = 'black'
ALPHA_DOTS = 0.3
COLOR_DOTS_MULTIPLE_RUNS = '#808080'
ALPHA_DOTS_MULTIPLE_RUNS = 0.08
COLOR_DOTS_BEST_IN_GEN = 'magenta'
ALPHA_BEST_IN_GEN = 0.3

# global variables
DEFAULT_EXPERIMENT_PATH = "~/nas/experiments.NAS_PAPER/"
DEFAULT_HOME_PATH_WINDOWS = "D:"        # replaces '~' (home directory) when running on Windows
# in colab use '/content/gdrive/MyDrive/experiments/'
DEFAULT_SAVE_PATH = "~/nas/graphs/"
EXPERIMENT_NAMES = ['Random Search', 'F-DENSER', 'Stepper-Decay', 'Stepper-Adaptive']  # All experiment folders used in plots
SAVE_ALL_PICTURES_FORMAT = None
EXPERIMENT_TITLE = ''
picture_count = 0


def fixup_path(path):
	""" replace '~' with user's home directory, and make path absolute """
	if platform.system() == 'Windows':
		path = path.replace('~', DEFAULT_HOME_PATH_WINDOWS)
	else:
		path = str(Path(path).expanduser().absolute())
	if not path.endswith('/'):
		path += '/'
	return path

def load_stats(experiment_name, experiment_path=DEFAULT_EXPERIMENT_PATH, max_runs=-1):
	""" load statistics of all runs (or max_runs if given) in an experiment """
	experiment_path = fixup_path(experiment_path)
	files_path = experiment_path + experiment_name + "/r??_statistics.pkl"
	file_list = sorted(glob.glob(files_path))
	stats = []
	for i, file in enumerate(file_list):
		if max_runs < 0 or i < max_runs:
			with open(file, 'rb') as f:
				stat = pickle.load(f)
				# guess run properties not set in old saved statistics
				if not hasattr(stat, 'run_nas_strategy'):
					stat.run_nas_strategy = experiment_name
				if not hasattr(stat, 'run_number'):
					stat.run_number = i
				if not hasattr(stat, 'run_dataset'):
					stat.run_dataset = 'mnist' if stat.run_generation >= 99 else ''
				stats.append(stat)
	if not stats:
		raise FileNotFoundError(f"No statistics found in {experiment_path}")
	while stats[-1].run_generation < stats[0].run_generation:  # delete last stat if incomplete
		del stats[-1]
	return stats

def calculate_statistics(stats, m : Metric, from_k_folds=False):
	values = [(run.best.metric_k_fold(m) if from_k_folds else run.best.metric(m))[-1]   for run in stats]  # get metric of best of last generation over all runs
	worst = np.min(values)
	best = np.max(values)
	best_index = np.argmax(values)
	if m != Metric.FITNESS and m != Metric.FINAL_TEST_FITNESS:
		best, worst = worst, best
		best_index = np.argmin(values)
	std = np.std(values)
	mean = np.mean(values)
	median = np.median(values)
	if m == 1:
		mean, median, std, worst, best = int(mean), int(median), int(std), int(worst), int(best)
	return mean, median, std, worst, best, best_index


def print_statistics(stats, experiment_name):
	stat = stats[0]
	print(f"\n{experiment_name}: {stat.run_generation + 1} generations")
	total_evaluations = sum(stat.evaluations_total for stat in stats)
	k_fold_evaluations = sum(stat.evaluations_k_folds for stat in stats)
	cache_hits = sum(stat.evaluations_cache_hits for stat in stats)
	invalid = sum(stat.evaluations_invalid for stat in stats)
	constraints_violated = sum(stat.evaluations_constraints_violated for stat in stats) if hasattr(stat, 'evaluations_constraints_violated') else -1
	run_time = sum(stat.run_time for stat in stats)
	eval_time = sum(stat.eval_time for stat in stats)
	eval_time_this_run = sum(stat.eval_time_this_run for stat in stats)
	eval_time_k_folds = sum(stat.eval_time_k_folds for stat in stats)
	eval_time_k_folds_this_run = sum(stat.eval_time_k_folds_this_run for stat in stats)
	print(f"{total_evaluations} evaluations, " + (
		f" ({k_fold_evaluations} for k-folds) " if k_fold_evaluations else "") + f"{cache_hits} cache hits, {invalid} invalid, {constraints_violated} constraints violated, avg evaluation time: {run_time / total_evaluations:.2f} s")
	print(f"runtime {decimal_hours(run_time)}, evaluation time {decimal_hours(eval_time)} (this run {decimal_hours(eval_time_this_run)})" + (
		f", k-folds: {decimal_hours(eval_time_k_folds)} (this run {decimal_hours(eval_time_k_folds_this_run)})" if eval_time_k_folds else ""))
	print()
	columns = ['', 'Average', 'Median', 'Std', 'Worst', 'Best', 'Best run']
	data = []
	for m in [Metric.FITNESS, Metric.FINAL_TEST_FITNESS, Metric.PARAMETERS, Metric.ERROR_RATE, Metric.TRAINING_ERROR_RATE, Metric.FINAL_TEST_ERROR_RATE]:
		mean, median, std, worst, best, best_index = calculate_statistics(stats, m)
		data.append([RunStatistics.metric_name(m), mean, median, std, worst, best, best_index])
	if hasattr(stats[0].best, 'k_fold_accuracy') and len(stats[0].best.k_fold_accuracy):
		for m in [Metric.FITNESS, Metric.ERROR_RATE]:
			mean, median, std, worst, best, best_index = calculate_statistics(stats, m, from_k_folds=True)
			data.append([RunStatistics.metric_name(m)+" 10 seeds", mean, median, std, worst, best, best_index])
	# pd.options.display.float_format = '{: .4f}'.format
	pd.set_option('display.max_columns', None)
	pd.set_option('display.max_colwidth', None)
	pd.set_option("expand_frame_repr", False)
	df = pd.DataFrame(data, columns=columns)
	df = df.round(decimals=2)  # .astype(object)
	print(df)


def reduced_legend(ax, population_size, additional_entries=1):
	""" hide labels for population plots except one """
	handles, labels = ax.get_legend_handles_labels()
	display = [0] + [i for i in range(population_size, population_size + additional_entries)]
	ax.legend([handle for i, handle in enumerate(handles) if i in display],
				[label for i, label in enumerate(labels) if i in display], loc='best', fontsize=15)


def default_ax():
	fig, ax = plt.subplots(figsize=(8, 6))
	return ax


def show_plot():
	global SAVE_ALL_PICTURES_FORMAT
	global picture_count

	if SAVE_ALL_PICTURES_FORMAT:
		title = plt.gca().get_title()
		title = title.replace(' - ', ' ')
		title, _, _ = title.partition('(')
		title, _, _ = title.partition(':')
		title = title.strip().replace(' ', '_')
		picture_count += 1
		plt.savefig(f"{fixup_path(DEFAULT_SAVE_PATH)}{picture_count:02d}_{title}.{SAVE_ALL_PICTURES_FORMAT}", format=SAVE_ALL_PICTURES_FORMAT, dpi=600 if SAVE_ALL_PICTURES_FORMAT == 'png' else 1200, transparent=True)
		plt.show()


def plot_set_limits(stat, m : Metric, ax):
	ngenerations = stat.run_generation + 1
	ax.set_xlim(0, ngenerations)
	ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
	# ax.set_xlabel("Generation", fontsize=14)

	ylim = RunStatistics.metric_ylimits(m)
	yticks_distance = stat.metric_ticks(m)
	if stat.run_dataset != 'mnist' and m != Metric.STEP_SIZE:  # keep predefined ylimits only for mnist variants, and for step width plot
		if m < 0:  # default ylimits for different error rates etc.
			ylim = (0, None)
			yticks_distance = None
		else:
			metric = stat.best.metric(m)
			if ngenerations >= 20 and not (m == 4 or m == 5):
				metric = metric[5:]  # for maximum calculation exclude first 5 generations if already more than 20
			if m == Metric.FITNESS:
				new_ylim = np.min(metric)  # lower limit for fitness
				# print(f"limit of {stat.metric_name(m)} : {ylim[0]} -> {new_ylim}")
				ylim = (new_ylim, ylim[1])
			else:
				new_ylim = np.max(metric)  # upper limit for other metrics
				if m == Metric.NLAYERS or m == Metric.NVARIABLES:  # variable and layer counts -> space for one more in plot
					new_ylim += 1
				print(f"limit of {stat.metric_name(m)} : {ylim[1]} -> {new_ylim}")
				ylim = (ylim[0], new_ylim)
			if yticks_distance < abs(ylim[1] - ylim[0]) / 40:  # if ticks become too small, use default ticks
				yticks_distance = None
	if ylim is not None:
		ax.set_ylim(ylim)
	if yticks_distance:
		ax.yaxis.set_major_locator(ticker.MultipleLocator(yticks_distance))
	ax.grid(True)


def plot_metric(stat, m : Metric, ax=None, use_transparency=False):
	global EXPERIMENT_TITLE

	if ax is None:
		ax = default_ax()
	ngenerations = stat.run_generation + 1
	xscale = np.arange(0, ngenerations)
	if SAVE_ALL_PICTURES_FORMAT:
		title = f"{EXPERIMENT_TITLE} - Evolution of {stat.metric_name(m)}"
	else:
		title = f"{EXPERIMENT_TITLE} - {stat.metric_name(m)} (best: {round(stat.best.metric(m)[-1], 4)})"
	ax.set_title(title, fontsize=20)
	population_size = 0
	if m <= Metric.FITNESS:
		if len(stat.best.metric_k_fold(m)):
			draw_mean_and_sd(ax, xscale, stat.best.metric_k_fold(m), stat.best.metric_k_fold_std(m), 'cyan' if m == 2 else 'orange', stat.metric_name(m) + ' 10 seeds')
		generation_metric = np.array(stat.metric_generation(m))
		(_, population_size) = generation_metric.shape
		for i in range(population_size):
			ax.plot(generation_metric[:, i], 'o', markersize=4, color=COLOR_DOTS if use_transparency else '#D0D0D0', alpha=ALPHA_DOTS if use_transparency else 1.0, label='population', zorder=-32)
		# if len(stat.best.metric_k_fold(m)):
		# 	ax.plot(xscale, stat.best.metric_k_fold(m), '-', color='cyan', alpha=1, label="K-folds of best")
		# 	ax.errorbar(xscale, stat.best.metric_k_fold(m), yerr=stat.best.metric_k_fold_std(m), color='cyan', alpha=1, zorder=10)
		ax.plot(stat.best_in_gen.metric(m), 'o', markersize=4, color=COLOR_DOTS_BEST_IN_GEN, alpha=ALPHA_BEST_IN_GEN, label='best in generation')
	ax.plot(stat.best.metric(m), '-', color=RunStatistics.metric_color(m), alpha=1, label='best fitness individual')
	plot_set_limits(stat, m, ax)
	if m <= 2:
		reduced_legend(ax, population_size, 3)
	show_plot()


def plot_metric_multiple_runs(stats, m : Metric, ax=None, use_transparency=True, add_legend=True):
	global EXPERIMENT_TITLE

	if ax is None:
		ax = default_ax()
	ax.set_title(f"{EXPERIMENT_TITLE} - {RunStatistics.metric_name(m)}", fontsize=14)
	all_population_size = 0
	nruns = 0
	if m <= Metric.FITNESS:
		all_metrics = np.hstack([stat.metric_generation(m) for stat in stats])
		(ngenerations, all_population_size) = all_metrics.shape
		for i in range(all_population_size):
			ax.plot(all_metrics[:, i], 'o', markersize=4, color=COLOR_DOTS_MULTIPLE_RUNS if use_transparency else '#D0D0D0', alpha=ALPHA_DOTS_MULTIPLE_RUNS if use_transparency else 1.0, label='population', zorder=-32)
		best_in_gen_metrics = np.asarray([stat.best_in_gen.metric(m) for stat in stats]).T
		(ngenerations, nruns) = best_in_gen_metrics.shape
		for i in range(nruns):
			ax.plot(best_in_gen_metrics[:, i], 'o', markersize=4, color=COLOR_DOTS_BEST_IN_GEN, alpha=ALPHA_DOTS_MULTIPLE_RUNS if use_transparency else 1.0, label='best in generation', zorder=-31)
	for stat in stats:
		ax.plot(stat.best.metric(m), '-', color=RunStatistics.metric_color(m), alpha=ALPHA_LINES, label='best fitness')

	plot_set_limits(stats[0], m, ax)
	if add_legend:
		reduced_legend(ax, all_population_size + nruns)
	show_plot()


def lighten_color(color, amount=0.5):
	"""
		Lightens the given color by multiplying (1-luminosity) by the given amount.
		Input can be matplotlib color string, hex string, or RGB tuple.

		Examples:
		>> lighten_color('g', 0.3)
		>> lighten_color('#F034A3', 0.6)
		>> lighten_color((.3,.55,.1), 0.5)
	"""
	c = colorsys.rgb_to_hls(*mc.to_rgb(color))
	return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def plot_metric_mean_and_sd(stats, m : Metric, ax=None):
	global EXPERIMENT_TITLE

	if ax is None:
		ax = default_ax()
	ngenerations = stats[0].run_generation + 1
	xscale = np.arange(0, ngenerations)
	ax.set_title(f"{EXPERIMENT_TITLE} - {RunStatistics.metric_name(m)}", fontsize=14)
	all_metrics = np.vstack([stat.best.metric(m) for stat in stats])
	draw_mean_and_sd(ax, xscale, np.mean(all_metrics, axis=0), np.std(all_metrics, axis=0), RunStatistics.metric_color(m), 'best fitness')
	plot_set_limits(stats[0], m, ax)
	show_plot()


def draw_mean_and_sd(ax, xscale, all_means, all_std, plot_color, label):
	""" plot mean and standard deviation as solid line and lighter color filled area """
	ax.plot(xscale, all_means, color=plot_color, label=label)
	# ax.fill_between(xscale, all_means - all_std, all_means + all_std, color=lighten_color(plot_color))
	ax.fill_between(xscale, all_means - all_std, all_means + all_std, color=plot_color, alpha=0.3)


def plot_different_accuracies(stat, ax=None):
	global EXPERIMENT_TITLE

	if ax is None:
		ax = default_ax()
	ngenerations = stat.run_generation + 1
	xscale = np.arange(0, ngenerations)
	ax.set_title(f"{EXPERIMENT_TITLE} - Different error rates", fontsize=20)
	if len(stat.best.k_fold_accuracy):
		ax.plot(xscale, stat.best.metric_k_fold(0), '-', color='cyan', label="avg 10 seed error rate")
		ax.errorbar(xscale, stat.best.metric_k_fold(0), yerr=stat.best.k_fold_accuracy_std, color='cyan', zorder=10)
		ax.plot(100.0 - np.array(stat.best.k_fold_final_accuracy) * 100, label="avg 10 seed final error rate")
	ax.plot(100.0 - np.array(stat.best.train_accuracy) * 100, label="training error rate")
	ax.plot(100.0 - np.array(stat.best.val_accuracy) * 100, label="validation error rate")
	ax.plot(stat.best.metric(0), color='blue', label="(test) error rate")
	ax.plot(100.0 - np.array(stat.best.final_test_accuracy) * 100, label="final test error rate")
	plot_set_limits(stat, Metric.NONE, ax)
	ax.legend(fontsize=12)
	show_plot()


def plot_variable_counts(stat, ax=None):
	global EXPERIMENT_TITLE

	if ax is None:
		ax = default_ax()
	ax.set_title(f"{EXPERIMENT_TITLE} - Variable Counts", fontsize=20)
	if hasattr(stat.best, 'statistic_variables'):
		ax.plot(stat.best.statistic_variables, label="variables")
		ax.plot(stat.best.statistic_cats, label="categoricals")
		ax.plot(stat.best.statistic_ints, label="integers")
		ax.plot(stat.best.statistic_nlayers, label="layers")
		ax.plot(stat.best.statistic_floats, label="floats")
		ax.plot(stat.best_in_gen.statistic_variable_mutations, label="variable mutations best in gen.")
		ax.plot(stat.best_in_gen.statistic_layer_mutations, label="layer mutations")
	# ax.plot(stat.best_in_gen.training_time, label="training time (s)")
	# ax.plot(stat.best_in_gen.training_epochs, label="training epochs (s)")
	ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
	ngenerations = stat.run_generation + 1
	ax.set_xlim(0, ngenerations)
	ax.set_ylim(0)
	ax.legend(fontsize=12, loc='center right')
	show_plot()


def do_all_plots(stats, experiment_name='', plot_individual_runs=True, plot_best_run=True, group_pictures=True, save_all_pictures_format=None):
	global SAVE_ALL_PICTURES_FORMAT
	global EXPERIMENT_TITLE

	SAVE_ALL_PICTURES_FORMAT = save_all_pictures_format

	EXPERIMENT_TITLE = experiment_name

	ax1, ax2, ax3 = None, None, None

	stat = stats[0]

	if len(stats) > 1:
		if group_pictures:
			fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 7))
		plot_metric_mean_and_sd(stats, Metric.FITNESS, ax1)
		plot_metric_mean_and_sd(stats, Metric.ERROR_RATE, ax2)
		plot_metric_mean_and_sd(stats, Metric.PARAMETERS, ax3)
		plt.show()

		if hasattr(stat.best, 'step_width'):
			plot_metric_mean_and_sd(stats, Metric.STEP_SIZE)
			plt.show()

		if plot_individual_runs:
			if group_pictures:
				fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 7))
			plot_metric_multiple_runs(stats, Metric.FITNESS, ax1)
			plot_metric_multiple_runs(stats, Metric.ERROR_RATE, ax2)
			plot_metric_multiple_runs(stats, Metric.PARAMETERS, ax3)
			plt.show()

		if plot_individual_runs:
			if group_pictures:
				fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 7))
			if hasattr(stat.best, 'step_width'):
				plot_metric_multiple_runs(stats, Metric.STEP_SIZE, ax1)
			if hasattr(stat.best, 'statistic_nlayers'):
				plot_metric_multiple_runs(stats, Metric.NLAYERS, ax2)
				plot_metric_multiple_runs(stats, Metric.NVARIABLES, ax3)
			plt.show()

		best_parameter_index = calculate_statistics(stats, Metric.FITNESS)[-1]
		stat = stats[best_parameter_index]
		if plot_best_run:
			print(f"Plots for best run #{best_parameter_index}")

	if plot_best_run:
		if group_pictures:
			fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 7))
		plot_metric(stat, Metric.FITNESS, ax1)
		plot_metric(stat, Metric.ERROR_RATE, ax2)
		plot_metric(stat, Metric.PARAMETERS, ax3)
		plt.show()

		if group_pictures:
			fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 7))
		if hasattr(stat.best, 'step_width'):
			plot_metric(stat, Metric.STEP_SIZE, ax1)
		plot_different_accuracies(stat, ax2)
		plot_variable_counts(stat, ax3)
		plt.show()


def plot_all_mean_and_sd(save_path=DEFAULT_SAVE_PATH, save_format="svg"):
	global SAVE_ALL_PICTURES_FORMAT
	global EXPERIMENT_TITLE

	fig, ax = plt.subplots(3, 4, figsize=(20, 10.5), constrained_layout=True)

	for row, name in enumerate(EXPERIMENT_NAMES):
		stats = load_stats(name)
		EXPERIMENT_TITLE = name
		plot_metric_mean_and_sd(stats, Metric.FITNESS, ax[0, row])
		plot_metric_mean_and_sd(stats, Metric.ERROR_RATE, ax[1, row])
		plot_metric_mean_and_sd(stats, Metric.PARAMETERS, ax[2, row])

	plt.savefig(fixup_path(save_path) + "Fitness Error Params Mean and SD." + save_format, format=save_format, dpi=1200, transparent=True)
	plt.show()


def plot_fitness_mean_and_sd(save_path=DEFAULT_SAVE_PATH, save_format="svg"):
	global EXPERIMENT_TITLE

	fig, ax = plt.subplots(1, 4, figsize=(20, 3.5), constrained_layout=True)

	for i, name in enumerate(EXPERIMENT_NAMES):
		stats = load_stats(name)
		EXPERIMENT_TITLE = name
		plot_metric_mean_and_sd(stats, Metric.FITNESS, ax[i])

	plt.savefig(fixup_path(save_path) + "Fitness Mean and SD." + save_format, format=save_format, dpi=1200, transparent=True)
	plt.show()


def plot_multiple_runs(save_path=DEFAULT_SAVE_PATH, save_format="png"):
	global EXPERIMENT_TITLE

	fig, ax = plt.subplots(3, 4, figsize=(20, 10.5), constrained_layout=True)

	for i, name in enumerate(EXPERIMENT_NAMES):
		stats = load_stats(name)
		EXPERIMENT_TITLE = name
		add_legend = (i == 0)
		plot_metric_multiple_runs(stats, Metric.FITNESS, ax[0, i], add_legend=add_legend)
		plot_metric_multiple_runs(stats, Metric.ERROR_RATE, ax[1, i], add_legend=add_legend)
		plot_metric_multiple_runs(stats, Metric.PARAMETERS, ax[2, i], add_legend=add_legend)

	plt.savefig(fixup_path(save_path) + "Multiple Runs." + save_format, format=save_format, dpi=(300 if save_format == 'png' else 1200), transparent=True)
	plt.show()


def box_plot(m=Metric.FITNESS, save_path=DEFAULT_SAVE_PATH, save_format="svg"):
	experiment_names = ['Random Search', 'F-DENSER', 'Stepper-Decay', 'Stepper-Adaptive']
	metric_color = ['lightsalmon', 'violet', 'lightblue']
	medianprops = dict(linestyle='-', linewidth=1, color='black')

	stats_list = []
	for name in experiment_names:
		stats = load_stats(name)
		stats_list.append(stats)

	values_list = []
	for stats in stats_list:
		values = [run.best.metric(m)[-1] for run in stats]  # get metric of best of last generation over all runs
		values_list.append(values)

	fig, ax = plt.subplots(figsize=(6, 4))
	ax.set_title(f"{RunStatistics.metric_name(m)}", fontsize=14)
	bplot = ax.boxplot(values_list, patch_artist=True, tick_labels=experiment_names, medianprops=medianprops)
	for patch in bplot['boxes']:
		patch.set_facecolor(metric_color[m])
	ax.yaxis.set_major_locator(ticker.MultipleLocator(RunStatistics.metric_ticks(m)))
	ax.grid(True)

	plt.savefig(fixup_path(save_path) + f"Box Plot {RunStatistics.metric_name_lowercase(m)}." + save_format, format=save_format, dpi=(300 if save_format == 'png' else 1200))
	plt.show()


def box_plots_3(experiment_path=DEFAULT_EXPERIMENT_PATH, save_path=DEFAULT_SAVE_PATH, save_format="svg"):
	experiment_names = ['Random Search', 'F-DENSER', 'Stepper-Decay', 'Stepper-Adaptive']
	metric_color = ['lightsalmon', 'violet', 'lightblue']
	medianprops = dict(linestyle='-', linewidth=1, color='black')

	stats_list = []
	for name in experiment_names:
		stats = load_stats(name, experiment_path=experiment_path)
		stats_list.append(stats)

	fig, axes = plt.subplots(1, 3, figsize=(20, 5))
	for i in range(0, 2 + 1):
		ax = axes[i]
		m = [Metric.FITNESS, Metric.ERROR_RATE, Metric.PARAMETERS][i]
		ax.set_title(f"{RunStatistics.metric_name(m)}", fontsize=14)
		values_list = []
		for stats in stats_list:
			values = [run.best.metric(m)[-1] for run in stats]  # get metric of best of last generation over all runs
			values_list.append(values)
		bplot = ax.boxplot(values_list, patch_artist=True, labels=experiment_names, medianprops=medianprops)    # can add notch=True
		ticks = RunStatistics.metric_ticks(m)
		if m == 0:
			ax.set_ylim(0)
		if m == 1:
			ax.set_ylim(0, 500000)
			ticks = 50000
		for patch in bplot['boxes']:
			patch.set_facecolor(metric_color[m])
		ax.yaxis.set_major_locator(ticker.MultipleLocator(ticks))
		ax.grid(True)

	plt.savefig(fixup_path(save_path) + f"Box Plots Fitness Error Parameters." + save_format, format=save_format, dpi=(300 if save_format == 'png' else 1200))
	plt.show()


def run_nonparametric_tests(experiment_path=DEFAULT_EXPERIMENT_PATH):
	experiment_names = ['Random Search', 'F-DENSER', 'Stepper-Decay', 'Stepper-Adaptive']

	stats_list = []
	for name in experiment_names:
		stats = load_stats(name, experiment_path=experiment_path)
		stats_list.append(stats)

	for i in range(0, 2 + 1):
		m = [Metric.FITNESS, Metric.ERROR_RATE, Metric.PARAMETERS][i]
		print(f"\nParameter {RunStatistics.metric_name(m)}:")
		values_list = []
		for stats in stats_list:
			values = [run.best.metric(m)[-1] for run in stats]  # get metric of best of last generation over all runs
			values_list.append(values)

		for (exp1, exp2) in [(1, 0), (2, 1), (3, 1), (3, 2)]:
			result = scipy.stats.mannwhitneyu(values_list[exp1], values_list[exp2], alternative=('greater' if m == 2 else 'less'))
			print(f"{experiment_names[exp1]} > {experiment_names[exp2]}: p={result[1]:.6f}")
		# result = scipy.stats.mannwhitneyu(values_list[exp1], values_list[exp2])
		# print(f"{experiment_names[exp1]} = {experiment_names[exp2]}: p={result[1]:.6f}")


# Module's main function. Call this and uncomment to generate different plots.
if __name__ == "__main__":
	exp_path = "~/nas/experiments.fashion4"

	# exp_name = 'Stepper-Adaptive'
	# experiment_stats = load_stats(exp_name, experiment_path=exp_path)
	# print_statistics(experiment_stats, exp_name)
	# do_all_plots(experiment_stats, experiment_name=exp_name, plot_individual_runs=True, plot_best_run=True, group_pictures=True)

	run_nonparametric_tests(experiment_path=exp_path)
	box_plots_3(experiment_path=exp_path)
	# box_plot(2)
	# box_plot(0)
	# box_plot(1)

	# plot_fitness_mean_and_sd()
	# plot_all_mean_and_sd()
	# plot_multiple_runs()
