
import engine
# from keras.models import load_model
# from keras.preprocessing.image import ImageDataGenerator
# from pickle import load
# import visualkeras

engine.do_nas_search("D:/experiments", grammar_file='config/lenet.grammar')

#for run in range(0, 10):
#	print(f"***** Starting FDENSER 10 Ep run {run:02d} of 10, random seed {100+run} **********")
#	engine.do_nas_search("D:/experiments/F-DENSER 10 Ep", config_file='config/config_fdenser.json', grammar_file='config/lenet.grammar', override_experiment_name=f"FDENSER {run:02d}", override_random_seed=100+run)

# model = engine.test_saved_model()
# model.summary(line_length=120)

# print("layer:conv num-filters:26 filter-shape:4 stride:2 padding:valid act:sigmoid bias:False batch-norm:False input:-1\nlayer:conv num-filters:4 filter-shape:5 stride:3 padding:valid act:linear bias:True batch-norm:True input:0\nlayer:fc act:elu num-units:367 bias:True batch-norm:False input:1\nlayer:output num-units:10 bias:True input:2\nlearning:adam lr:0.0005 beta1:0.9 beta2:0.999 batch_size:535")
# print()

# visualkeras.layered_view(model, legend=True)


# from tests.test_utils import Test
# test = Test()
# test.test_add_layer_random()

'''
To do
save model raus

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
