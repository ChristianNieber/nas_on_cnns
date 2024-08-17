# Implementation Notes

The Python code of this work started with the F-DENSER code, more precisely the newer F-DENSER++ version.
The code was taken from <https://github.com/fillassuncao/fast-denser3>, which originally included the following features:
* Construct and evaluate a CNN using tensorflow/keras from a grammatical representation.
* The ability to resume a previously aborted run by storing all relevant state on disk with the pickle library
* The sequence of mutations depends deterministically on the random number seed and is repeatable, as long as nothing changes on the algorithms.
* The ability to re-train a previously trained model for more epochs

I rewrote most of the code and extended the project to add features:

* Change the representation of an individual's layer parameters, so that mutation step size control can be implemented more easily.
* Before applying mutations, collect all variables of an individual in a list, so that an algorithm can easily process all variables at once.
* Mutated variables are written back from the list into the tree representation.
* More logging: Show more statistics for every individual, show when an individual becomes the new best in a run, optional logging of mutations, different colors for different types of logging.
* Record many statistics for each evaluated individual, including training loss, training accuracy, validation loss, validation accuracy, fitness, number of parameters, step width, number of layers and variables by type.
* Record the evolutionary history of each individual in terms of parents and applied mutations.
* Optional re-evaluation of candidates for the new best individual using k-folds, to calculate a more precise average error rate.
* Store complete statistics on disk after each generation, so that statistics and plots can be displayed in real-time by a separate process in a Jupyter notebook.
* Caching of all evaluation results in a file. This will not speed up new runs significantly, but often speeds up re-runs if a run fails, or when small changes of the algorithm are tested. This also collects data points about the search space that could be statistically analysed later.
* An option to run multiple NAS searches with different random number seeds. 
* NAS searches often run for several hours. When the process is aborted or crashes (which happened quite often, especially when running in the cloud), the search can resume from the last completed generation.
* A collection of functions to plot results, combining different statistics from one run or from multiple runs.

# Run unit tests

To run all tests:

`python -m unittest discover -s tests`

# To do:
- unity evaluation