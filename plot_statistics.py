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

from runstatistics import RunStatistics

ALPHA_DOTS = 0.3
ALPHA_LINES = 0.7
SAVE_ALL_PICTURES = False
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
	values = [stat.best.metric(m)[-1] for stat in stats]
	worst = np.min(values)
	best = np.max(values)
	best_index = np.argmax(values)
	if m != 2:
		best, worst = worst, best
		best_index = np.argmin(values)
	std = np.std(values)
	mean = np.mean(values)
	if m == 1:
		mean, std, worst, best = int(mean), int(std), int(worst), int(best)
	return mean, std, worst, best, best_index

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
	for m in range(0, 2 + 1):
		mean, std, worst, best, best_index = calculate_statistics(stats, m)
		data.append([stats[0].metric_name(m), mean, std, worst, best, best_index])
	# pd.options.display.float_format = '{: .4f}'.format
	df = pd.DataFrame(data, columns=columns)
	df = df.round(decimals=2).astype(object)
	print(df.head())

def reduced_legend(ax, population_size, additional_entries=1):
	""" hide labels for population plots except one """
	handles, labels = ax.get_legend_handles_labels()
	display = [0] + [i for i in range(population_size, population_size + additional_entries)]
	ax.legend([handle for i,handle in enumerate(handles) if i in display],
				[label for i,label in enumerate(labels) if i in display], loc = 'best', fontsize=15)

def default_ax():
	fig, ax = plt.subplots(figsize = (8, 6))
	return ax

def show_plot():
	global SAVE_ALL_PICTURES
	global picture_count

	if SAVE_ALL_PICTURES:
		title = plt.gca().get_title()
		title, _, _ = title.partition('(')
		title, _, _ = title.partition(':')
		name, _, _ = experiment_name.partition('/')
		name, _, _ = name.partition('_')
		picture_count += 1
		plt.savefig(f"{experiments_path}graphs/{picture_count:02d} - {title}.svg", format="svg", transparent=True)
	plt.show()

def plot_set_limits(stat, m, ax):
	ax.set_xlim(0, stat.run_generation + 1)
	ax.set_ylim(RunStatistics.metric_ylimits(m))
	ax.set_xlabel("generation")
	ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
	ax.yaxis.set_major_locator(ticker.MultipleLocator(stat.metric_ticks(m)))
	ax.grid(True)

def plot_metric(stat, m, ax=None):
	global EXPERIMENT_TITLE

	if ax is None:
		ax = default_ax()
	generation_metric = np.array(stat.metric_generation(m))
	(ngenerations, population_size) = generation_metric.shape
	xscale = np.arange(0, ngenerations)
	ax.set_title(f"{EXPERIMENT_TITLE} {stat.metric_name(m)} (best: {round(stat.best.metric(m)[-1], 4)})", fontsize=22)
	for i in range(population_size):
		ax.plot(generation_metric[:, i], 'o', markersize=4, color ='#808080', alpha=ALPHA_DOTS, label = 'population', zorder=-32)
	if len(stat.best.metric_k_fold(m)):
		ax.plot(xscale, stat.best.metric_k_fold(m), '-', color='cyan', alpha=ALPHA_LINES, label="K-folds of best")
		ax.errorbar(xscale, stat.best.metric_k_fold(m), yerr = stat.best.metric_k_fold_std(m), color='cyan', alpha=1, zorder=10)
	ax.plot(stat.best_in_gen.metric(m), '-', color = 'magenta', alpha=ALPHA_LINES, label='best in generation')
	ax.plot(stat.best.metric(m), '-', color = 'blue', alpha=ALPHA_LINES, label='best fitness individual')
	plot_set_limits(stat, m, ax)
	reduced_legend(ax, population_size, 2)
	show_plot()

def plot_different_accuracies(stat, ax=None):
	global EXPERIMENT_TITLE

	if ax is None:
		ax = default_ax()
	ngenerations = stat.run_generation + 1
	xscale = np.arange(0, ngenerations)
	ax.set_title(f"{EXPERIMENT_TITLE} Different error rates", fontsize=22)
	ax.plot(100.0 - np.array(stat.best.train_accuracy) * 100, label="training error rate")
	ax.plot(100.0 - np.array(stat.best.val_accuracy) * 100, label="validation error rate")
	ax.plot(stat.best.metric(0), color='blue', label="(test) error rate")
	ax.plot(100.0 - np.array(stat.best.final_test_accuracy) * 100, label="final test error rate")
	if len(stat.best.k_fold_accuracy):
		ax.plot(xscale, stat.best.metric_k_fold(0), '-', color='cyan', label="avg K-fold error rate")
		ax.errorbar(xscale, stat.best.metric_k_fold(0), yerr = stat.best.k_fold_accuracy_std, color='cyan', zorder=10)
		ax.plot(100.0 - np.array(stat.best.k_fold_final_accuracy) * 100, label="avg K-fold final error rate")
	plot_set_limits(stat, -1, ax)
	ax.legend(fontsize=12)
	show_plot()

def multi_plot_metric(stats, m, ax=None):
	global EXPERIMENT_TITLE

	if ax is None:
		ax = default_ax()
	nruns = len(stats)
	all_metrics = np.hstack([stat.metric_generation(m) for stat in stats])
	(ngenerations, all_population_size) = all_metrics.shape
	ax.set_title(f"{EXPERIMENT_TITLE} {stats[0].metric_name(m)} over {nruns} runs", fontsize=22)
	for i in range(all_population_size):
		ax.plot(all_metrics[:, i], 'o', markersize=4, color ='#808080', alpha=ALPHA_DOTS, label = 'population', zorder=-32)
	for stat in stats:
		ax.plot(stat.best.metric(m), '-', color = 'blue', alpha=ALPHA_LINES, label='best fitness')
	plot_set_limits(stat, m, ax)
	reduced_legend(ax, all_population_size)
	show_plot()

def do_all_plots(stats, display_table=False, save_all_pictures=False, experiment_title=''):
	global SAVE_ALL_PICTURES
	global EXPERIMENT_TITLE

	SAVE_ALL_PICTURES = save_all_pictures
	EXPERIMENT_TITLE = experiment_title

	if len(stats) > 1:
		multi_plot_metric(stats, 0)
		multi_plot_metric(stats, 1)
		multi_plot_metric(stats, 2)

		_, _, _, _, best_parameter_index = calculate_statistics(stats, 1)
		stat = stats[best_parameter_index]
		print(f"Plots for run #{best_parameter_index}")

	plot_metric(stat, 0)
	plot_metric(stat, 1)
	plot_metric(stat, 2)
	plot_different_accuracies(stat)

if __name__ == "__main__":
	# experiment_name = 'FDENSER256_10/FDENSER*'
	# experiment_title = 'F-DENSER'

	experiment_name = 'DECAY256X/DECAY*'
	experiment_title = 'Stepper'

	experiments_path = 'D:/experiments/'
	# experiments_path = '/content/gdrive/MyDrive/experiments/'

	save_pictures = True

	stats = load_stats(experiments_path, experiment_name)
	print_statistics(stats, experiments_path, experiment_name)
	do_all_plots(stats, save_all_pictures=True, experiment_title=experiment_title)