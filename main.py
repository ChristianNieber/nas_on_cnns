import engine

engine.do_nas_search(experiments_path='D:/experiments', config_file='config/config_debug.json', dataset='mnist', grammar_file='lenet.grammar', reevaluate_best_10_seeds=True)

# from tests.test_utils import Test
# test = Test()
# test.test_add_layer_random()

# run unittests:
# python -m unittest discover -s tests

'''
To do

add more test code

later
-----
- more activation functions and parameters?
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
- scaling/brightness of input image
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
- simplify layer representation

Install keras with jax:
- only runs on Linux
pip install --upgrade jax[cuda12]==0.4.28
set environment variable:
export KERAS_BACKEND=jax
'''
