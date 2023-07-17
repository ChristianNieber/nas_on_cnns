import engine
#from keras.models import load_model
#from keras.preprocessing.image import ImageDataGenerator
#from pickle import load
# import visualkeras

engine.do_nas_search("D:/experiments")

#model = engine.test_saved_model()
#model.summary(line_length=120)
#
#print("layer:conv num-filters:26 filter-shape:4 stride:2 padding:valid act:sigmoid bias:False batch-normalization:False input:-1\nlayer:conv num-filters:4 filter-shape:5 stride:3 padding:valid act:linear bias:True batch-normalization:True input:0\nlayer:fc act:elu num-units:367 bias:True batch-normalization:False input:1\nlayer:output num-units:10 bias:True input:2\nlearning:adam lr:0.0005 beta1:0.9 beta2:0.999 batch_size:535")
#print()

# visualkeras.layered_view(model, legend=True)

# TODO

# - simplify layer representation
# - more activation functions and parameters?
# - Comparison with GE?
# - numeric default values
# - ES integers
# - ES categorical values

# - real time plots & visualisation

# - stepwidth mutation, initial stepwidth
# - geometric normal distribution for some variables?

# EA strategies to try
# -------------------
# - verify fitness with K-fold validation
# - crossover?
# - early stopping

# CNN features to try
# -------------------
# - weight regularization (weight decay)
# - add noise

# Done
# ----
# - Input 28*28*3 instead of 32*32*3
# - lenet test code
# - better configuration variables
# - reduced numeric parameters groups to single value for simplicity
# - better metrics
# - phenotypes with \n
# - record mutations
# - penalty function for large networks
# - test variance of accuracy (K-fold, or random number seeds)
# - final test for new best
# - save/restore random state
# - save log
# - colored log
# - early stopping
# - early stop parameter?
