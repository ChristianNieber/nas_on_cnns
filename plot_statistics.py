import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle
import glob
# import tabulate
# from IPython.display import HTML, display
# import tabletext
import pandas as pd
import matplotlib.colors as mc
import colorsys

from runstatistics import RunStatistics

# plot colors and alpha channel
COLOR_DOTS = '#808080'
ALPHA_DOTS = 0.3
ALPHA_DOTS_MULTIPLE_RUNS = 0.08
ALPHA_LINES = 0.3
ALPHA_BEST_IN_GEN = 0.3

# global variables
DEFAULT_SAVE_PATH = "D:/Data/workspace/Dissertation/graphs/"
SAVE_ALL_PICTURES_FORMAT = False
EXPERIMENT_TITLE = ''
picture_count = 0


def load_stats(experiment_path, experiment_name, max_folder=-1):
	folders = sorted(glob.glob(experiment_path + experiment_name)) if '*' in experiment_name else [experiment_path + experiment_name]
	stats = []
	for i, path in enumerate(folders):
		if max_folder < 0 or i < max_folder:
			with open(path + '/statistics.pkl', 'rb') as f:
				stats.append(pickle.load(f))
	if not stats:
		raise FileNotFoundError(f"No statistics found in {experiment_path}")
	stat = stats[0]
	while stats[-1].run_generation < stat.run_generation:   # delete last stat if incomplete
		del stats[-1]
	return stats


def hms(seconds):
	return time.strftime("%H:%M:%S", time.gmtime(seconds))


def calculate_statistics(stats, m):
	values = [run.best.metric(m)[-1] for run in stats]    # get metric of best of last generation over all runs
	worst = np.min(values)
	best = np.max(values)
	best_index = np.argmax(values)
	if m != 2 and m != 8:
		best, worst = worst, best
		best_index = np.argmin(values)
	std = np.std(values)
	mean = np.mean(values)
	median = np.median(values)
	if m == 1:
		mean, median, std, worst, best = int(mean), int(median), int(std), int(worst), int(best)
	return mean, median, std, worst, best, best_index


def print_statistics(stats, experiments_path, experiment_name):
	stat = stats[0]
	print(f"\n{experiment_name}: {stat.run_generation+1} generations")
	total_evaluations = sum(stat.evaluations_total for stat in stats)
	k_fold_evaluations = sum(stat.evaluations_k_folds for stat in stats)
	cache_hits = sum(stat.evaluations_cache_hits for stat in stats)
	invalid = sum(stat.evaluations_invalid for stat in stats)
	print(f"{total_evaluations} evaluations, " + (f" {k_fold_evaluations} for k-folds), " if k_fold_evaluations else "") + f"{cache_hits} cache hits, {invalid} invalid")
	run_time = sum(stat.run_time for stat in stats)
	eval_time = sum(stat.eval_time for stat in stats)
	eval_time_this_run = sum(stat.eval_time_this_run for stat in stats)
	eval_time_k_folds = sum(stat.eval_time_k_folds for stat in stats)
	eval_time_k_folds_this_run = sum(stat.eval_time_k_folds_this_run for stat in stats)
	print(f"runtime {hms(run_time)}, evaluation time {hms(eval_time)} (this run {hms(eval_time_this_run)})" + (f", k-folds: {hms(eval_time_k_folds)} (this run {hms(eval_time_k_folds_this_run)})" if eval_time_k_folds else ""))
	print()
	columns = ['', 'Average', 'Std', 'Worst', 'Best', 'Best run']
	data = []
	for m in [2, 8, 1, 0, 7, 6]:
		mean, _median, std, worst, best, best_index = calculate_statistics(stats, m)
		data.append([RunStatistics.metric_name(m), mean, std, worst, best, best_index])
	# pd.options.display.float_format = '{: .4f}'.format
	df = pd.DataFrame(data, columns=columns)
	df = df.round(decimals=2).astype(object)
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
		plt.savefig(f"D:/Data/workspace/Dissertation/graphs2/{picture_count:02d}_{title}.{SAVE_ALL_PICTURES_FORMAT}", format=SAVE_ALL_PICTURES_FORMAT, dpi=600 if SAVE_ALL_PICTURES_FORMAT=='png' else 1200, transparent=True)
		plt.show()


def plot_set_limits(stat, m, ax):
	ax.set_xlim(0, stat.run_generation + 1)
	ax.set_ylim(RunStatistics.metric_ylimits(m))
	# ax.set_xlabel("Generation", fontsize=14)
	ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
	ax.yaxis.set_major_locator(ticker.MultipleLocator(stat.metric_ticks(m)))
	ax.grid(True)


def plot_metric(stat, m, use_transparency=False, ax=None):
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
	if m <= 2:
		generation_metric = np.array(stat.metric_generation(m))
		(_, population_size) = generation_metric.shape
		dot_color = COLOR_DOTS
		if not use_transparency:
			dot_color = '#D0D0D0'
		for i in range(population_size):
			ax.plot(generation_metric[:, i], 'o', markersize=4, color=dot_color, alpha=ALPHA_DOTS if use_transparency else 1.0, label='population', zorder=-32)
		if len(stat.best.metric_k_fold(m)):
			ax.plot(xscale, stat.best.metric_k_fold(m), '-', color='cyan', alpha=1, label="K-folds of best")
			ax.errorbar(xscale, stat.best.metric_k_fold(m), yerr=stat.best.metric_k_fold_std(m), color='cyan', alpha=1, zorder=10)
		ax.plot(stat.best_in_gen.metric(m), '-', color='magenta', alpha=ALPHA_BEST_IN_GEN, label='best in generation')
	ax.plot(stat.best.metric(m), '-', color=RunStatistics.metric_color(m), alpha=1, label='best fitness individual')
	plot_set_limits(stat, m, ax)
	if m <= 2:
		reduced_legend(ax, population_size, 3)
	show_plot()


def plot_metric_multiple_runs(stats, m, ax=None, use_transparency=True, add_legend=True):
	global EXPERIMENT_TITLE

	if ax is None:
		ax = default_ax()
	nruns=len(stats)
	ax.set_title(f"{EXPERIMENT_TITLE} - {RunStatistics.metric_name(m)}", fontsize=14)
	all_population_size = 0
	if m <= 2:
		all_metrics = np.hstack([stat.metric_generation(m) for stat in stats])
		(ngenerations, all_population_size) = all_metrics.shape
		dot_color = COLOR_DOTS
		if not use_transparency:
			dot_color = '#D0D0D0'
		for i in range(all_population_size):
			ax.plot(all_metrics[:, i], 'o', markersize=4, color=dot_color, alpha=ALPHA_DOTS_MULTIPLE_RUNS if use_transparency else 1.0, label='population', zorder=-32)
	for stat in stats:
		ax.plot(stat.best.metric(m), '-', color=RunStatistics.metric_color(m), alpha=ALPHA_LINES, label='best fitness')

	plot_set_limits(stats[0], m, ax)
	if add_legend:
		reduced_legend(ax, all_population_size)
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
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def plot_metric_mean_and_sd(stats, m, ax=None):
	global EXPERIMENT_TITLE

	if ax is None:
		ax = default_ax()
	ngenerations = stats[0].run_generation + 1
	xscale = np.arange(0, ngenerations)
	nruns=len(stats)
	ax.set_title(f"{EXPERIMENT_TITLE} - {RunStatistics.metric_name(m)}", fontsize=14)
	all_metrics = np.vstack([stat.best.metric(m) for stat in stats])
	all_means = np.mean(all_metrics, axis=0)
	all_std = np.std(all_metrics, axis=0)
	plot_color = RunStatistics.metric_color(m)
	ax.plot(xscale, all_means, color=plot_color, label='best fitness')
	ax.fill_between(xscale, all_means - all_std, all_means + all_std, color=lighten_color(plot_color))
	plot_set_limits(stats[0], m, ax)
	show_plot()

def plot_different_accuracies(stat, ax=None):
	global EXPERIMENT_TITLE

	if ax is None:
		ax = default_ax()
	ngenerations = stat.run_generation + 1
	xscale = np.arange(0, ngenerations)
	ax.set_title(f"{EXPERIMENT_TITLE} - Different error rates", fontsize=20)
	ax.plot(100.0 - np.array(stat.best.train_accuracy) * 100, label="training error rate")
	ax.plot(100.0 - np.array(stat.best.val_accuracy) * 100, label="validation error rate")
	ax.plot(stat.best.metric(0), color='blue', label="(test) error rate")
	ax.plot(100.0 - np.array(stat.best.final_test_accuracy) * 100, label="final test error rate")
	if len(stat.best.k_fold_accuracy):
		ax.plot(xscale, stat.best.metric_k_fold(0), '-', color='cyan', label="avg k-fold error rate")
		ax.errorbar(xscale, stat.best.metric_k_fold(0), yerr=stat.best.k_fold_accuracy_std, color='cyan', zorder=10)
		ax.plot(100.0 - np.array(stat.best.k_fold_final_accuracy) * 100, label="avg k-fold final error rate")
	plot_set_limits(stat, -1, ax)
	ax.legend(fontsize=12)
	show_plot()

def plot_variable_counts(stat, ax=None):
	global EXPERIMENT_TITLE

	if ax is None:
		ax = default_ax()
	ax.set_title(f"{EXPERIMENT_TITLE} - Variable Counts", fontsize=20)
	if hasattr(stat.best, 'statistic_variables'):
		ax.plot(stat.best.statistic_variables, label="variables")
		ax.plot(stat.best.statistic_nlayers, label="layers")
		ax.plot(stat.best.statistic_floats, label="floats")
		ax.plot(stat.best.statistic_ints, label="integers")
		ax.plot(stat.best.statistic_cats, label="categoricals")
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

def do_test_plots(stats, experiment_title='', plot_individual_runs=True, plot_stepwidth=True, plot_best_run=True, group_pictures=True, save_all_pictures_format=None):
	global SAVE_ALL_PICTURES_FORMAT
	global EXPERIMENT_TITLE

	SAVE_ALL_PICTURES_FORMAT = save_all_pictures_format
	EXPERIMENT_TITLE = experiment_title

	ax1, ax2, ax3 = None, None, None

	stat = stats[0]

	if len(stats) > 1:
		if plot_individual_runs:
			if group_pictures:
				fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 7))
			plot_metric_multiple_runs(stats, 2, ax1)
			plot_metric_multiple_runs(stats, 0, ax2)
			plot_metric_multiple_runs(stats, 1, ax3)

def do_all_plots(stats, experiment_title='', plot_individual_runs=True, plot_stepwidth=True, plot_best_run=True, group_pictures=True, save_all_pictures_format=None):
	global SAVE_ALL_PICTURES_FORMAT
	global EXPERIMENT_TITLE

	SAVE_ALL_PICTURES_FORMAT = save_all_pictures_format

	EXPERIMENT_TITLE = experiment_title

	ax1, ax2, ax3 = None, None, None

	stat = stats[0]

	if len(stats) > 1:
		if group_pictures:
			fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 7))
		plot_metric_mean_and_sd(stats, 2, ax1)
		plot_metric_mean_and_sd(stats, 0, ax2)
		plot_metric_mean_and_sd(stats, 1, ax3)

		if hasattr(stat.best, 'step_width'):
			plot_metric_mean_and_sd(stats, 3)

		if plot_individual_runs:
			if group_pictures:
				fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 7))
			plot_metric_multiple_runs(stats, 2, ax1)
			plot_metric_multiple_runs(stats, 0, ax2)
			plot_metric_multiple_runs(stats, 1, ax3)

		if plot_individual_runs:
			if group_pictures:
				fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 7))
				if hasattr(stat.best, 'step_width'):
					plot_metric_multiple_runs(stats, 3, ax1)
			if hasattr(stat.best, 'statistic_nlayers'):
				plot_metric_multiple_runs(stats, 4, ax2)
				plot_metric_multiple_runs(stats, 5, ax3)

		best_parameter_index = calculate_statistics(stats, 2)[-1]
		stat = stats[best_parameter_index]
		if plot_best_run:
			print(f"Plots for best run #{best_parameter_index}")

	if plot_best_run:
		if group_pictures:
			fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 7))
		plot_metric(stat, 2, ax1)
		plot_metric(stat, 0, ax2)
		plot_metric(stat, 1, ax3)

		if group_pictures:
			fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 7))
		if hasattr(stat.best, 'step_width'):
			plot_metric(stat, 3, ax1)
		plot_different_accuracies(stat, ax2)
		plot_variable_counts(stat, ax3)


def plot_all_mean_and_sd(experiments_path, save_path=DEFAULT_SAVE_PATH, save_format="svg"):
	global SAVE_ALL_PICTURES_FORMAT
	global EXPERIMENT_TITLE

	experiment_names = ['RANDOM20/RANDOM*', 'FDENSER20/FDENSER*', 'DECAY20/DECAY*', 'ADAPTIVE20/ADAPTIVE*']
	experiment_titles = ['Random Search', 'F-DENSER', 'Stepper-Decay', 'Stepper-Adaptive' ]

	fig, ax = plt.subplots(3, 4, figsize=(20, 10.5), constrained_layout=True)

	for row in range(0 , len(experiment_names)):
		stats = load_stats(experiments_path, experiment_names[row])
		EXPERIMENT_TITLE = experiment_titles[row]
		plot_metric_mean_and_sd(stats, 2, ax[0, row])
		plot_metric_mean_and_sd(stats, 0, ax[1, row])
		plot_metric_mean_and_sd(stats, 1, ax[2, row])

	plt.savefig(save_path + "Fitness Error Params Mean and SD." + save_format, format=save_format, dpi=1200, transparent=True)
	plt.show()

def plot_fitness_mean_and_sd(experiments_path, save_path=DEFAULT_SAVE_PATH, save_format="svg"):
	global EXPERIMENT_TITLE

	experiment_names = ['RANDOM20/RANDOM*', 'FDENSER20/FDENSER*', 'DECAY20/DECAY*', 'ADAPTIVE20/ADAPTIVE*']
	experiment_titles = ['Random Search', 'F-DENSER', 'Stepper-Decay', 'Stepper-Adaptive' ]

	fig, ax = plt.subplots(1, 4, figsize=(20, 3.5), constrained_layout=True)

	for i in range(0 , len(experiment_names)):
		stats = load_stats(experiments_path, experiment_names[i])
		EXPERIMENT_TITLE = experiment_titles[i]
		plot_metric_mean_and_sd(stats, 2, ax[i])

	plt.savefig(save_path + "Fitness Mean and SD." + save_format, format=save_format, dpi=1200, transparent=True)
	plt.show()

def plot_multiple_runs(experiments_path, save_path = DEFAULT_SAVE_PATH, save_format="png"):
	global EXPERIMENT_TITLE

	experiment_names = ['RANDOM20/RANDOM*', 'FDENSER20/FDENSER*', 'DECAY20/DECAY*', 'ADAPTIVE20/ADAPTIVE*']
	experiment_titles = ['Random Search', 'F-DENSER', 'Stepper-Decay', 'Stepper-Adaptive' ]

	fig, ax = plt.subplots(3, 4, figsize=(20, 10.5), constrained_layout=True)

	for i in range(0 , len(experiment_names)):
		stats = load_stats(experiments_path, experiment_names[i])
		EXPERIMENT_TITLE = experiment_titles[i]
		add_legend = 1 if i == 0 else 0
		plot_metric_multiple_runs(stats, 2, ax[0, i], add_legend=add_legend)
		plot_metric_multiple_runs(stats, 0, ax[1, i], add_legend=add_legend)
		plot_metric_multiple_runs(stats, 1, ax[2, i], add_legend=add_legend)

	plt.savefig(save_path + "Multiple Runs." + save_format, format=save_format, dpi=(300 if save_format=='png' else 1200), transparent=True)
	plt.show()

def box_plot(experiments_path, m=2, save_path = DEFAULT_SAVE_PATH, save_format="svg"):

	experiment_stat_paths = ['RANDOM20/RANDOM*', 'FDENSER20/FDENSER*', 'DECAY20/DECAY*', 'ADAPTIVE20/ADAPTIVE*']
	experiment_names = ['Random Search', 'F-DENSER', 'Stepper-Decay', 'Stepper-Adaptive']
	experiment_colors = ['yellow', 'yellow', 'yellow', 'yellow']

	stats_list = []
	for path in experiment_stat_paths:
		stats = load_stats(experiments_path, path)
		stats_list.append(stats)

	values_list = []
	for stats in stats_list:
		values = [run.best.metric(m)[-1] for run in stats]  # get metric of best of last generation over all runs
		values_list.append(values)

	fig, ax = plt.subplots(figsize=(6, 4))
	ax.set_title(f"{RunStatistics.metric_name(m)}", fontsize=14)
	bplot = ax.boxplot(values_list, patch_artist=True, labels=experiment_names)
	for patch, color in zip(bplot['boxes'], experiment_colors):
		patch.set_facecolor(color)

	ax.yaxis.set_major_locator(ticker.MultipleLocator(RunStatistics.metric_ticks(m)))

	ax.grid(True)

	plt.savefig(save_path + f"Box Plot {RunStatistics.metric_name_lowercase(m)}." + save_format, format=save_format, dpi=(300 if save_format == 'png' else 1200))
	plt.show()

# Module's main function. Call this and uncomment to generate different plots.

if __name__ == "__main__":
	experiments_path = 'D:/experiments/'
	# experiments_path = '/content/gdrive/MyDrive/experiments/'

	# experiment_name = 'Stepper_test'
	# experiment_title = 'Stepper test'

	# experiment_name = 'FDENSER20/FDENSER*'
	# experiment_title = 'F-DENSER'

	# experiment_name = 'DECAY20/DECAY*'
	# experiment_title = 'Stepper-decay'

	#experiment_name = 'ADAPTIVE20/ADAPTIVE*'
	#experiment_title = 'Stepper-Adaptive'

	#stats = load_stats(experiments_path, experiment_name)
	#print_statistics(stats, experiments_path, experiment_name)
	#do_all_plots(stats, experiment_title=experiment_title, plot_individual_runs=True, plot_best_run=True, group_pictures=False, save_all_pictures_format='svg')

	box_plot(experiments_path, 2)
	box_plot(experiments_path, 0)
	box_plot(experiments_path, 1)

	plot_fitness_mean_and_sd(experiments_path)
	plot_multiple_runs(experiments_path)