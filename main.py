import engine
#from keras.models import load_model
#from keras.preprocessing.image import ImageDataGenerator
# import visualkeras
#from pickle import load

engine.do_nas_search(0)

# engine.test_saved_model()

# visualkeras.layered_view(model, legend=True)

# TODO
# - save log
# - simplify layer representation
# - more activation functions and parameters?
# - Comparison with GE?
# - numeric default values
# - ES integers
# - ES categorical values

# - real time plots & visualisation

# - stepwidth mutation, initial stepwidth
# - crossover?
# - geometric normal distribution for some variables?

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
