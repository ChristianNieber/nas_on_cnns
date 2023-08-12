import engine

experiments_path = "D:/experiments"

engine.do_nas_search(experiments_path, config_file='config/config.json', grammar_file='config/lenet.grammar')

#for run in range(0, 10):
#	print(f"***** Starting FDENSER{run:02d} of 10, random seed {100+run} **********")
#	engine.do_nas_search(experiments_path + "/FDENSER10", config_file='config/config_fdenser.json', grammar_file='config/lenet.grammar', override_experiment_name=f"FDENSER{run:02d}", override_random_seed=100+run)

#for run in range(0, 10):
#	print(f"***** Starting Stepper{run:02d} of 10, random seed {100+run} **********")
#	engine.do_nas_search(experiments_path + "/Stepper10", config_file='config/config10.json', grammar_file='config/lenet.grammar', override_experiment_name=f"Stepper{run:02d}", override_random_seed=100+run)

# from tests.test_utils import Test
# test = Test()
# test.test_add_layer_random()

# run unittests:
# python -m unittest discover -s tests

'''
To do

add more test code

multiple runs with different random seeds

- more activation functions and parameters?
- Comparison with GE?
- numeric default values
- ES integers
- ES categorical values

- initial stepwidth/default values for some variables?
- geometric normal distribution for some variables?

EA strategies later
-------------------
- crossover
- early stopping optimisation

CNN features to try
-------------------
- dropout
- weight regularization (weight decay)
- scaling of input image
- add noise
- data augmentation

Done
----
- Input 28*28*3 instead of 32*32*3
- lenet test code
- better configuration variables
- reduced numeric parameters groups to single value for simplicity
- better metrics
- phenotypes with \n
- record mutations
- penalty function for large networks
- test variance of accuracy (K-fold, or random number seeds)
- final test for new best
- save/restore random state
- save log
- colored log
- early stopping
- early stop parameter?
- simplify layer representation
'''
