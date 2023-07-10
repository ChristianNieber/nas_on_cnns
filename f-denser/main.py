import fast_denser
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
# import visualkeras
from pickle import load

fast_denser.search(0, 'mnist', 'config/config.json', 'config/lenet.grammar')

with open('D:/experiments/run_0/evaluator.pkl', 'rb') as f_data:
    evaluator = load(f_data)
    x_test = evaluator.dataset['x_test']
    y_test = evaluator.dataset['y_test']

model = load_model('D:/experiments/run_0/best.h5')
datagen_test = ImageDataGenerator()

# model.summary(line_length=120)
# visualkeras.layered_view(model, legend=True)

# TODO
# - layer mutation
# - learning optimizers
# - simplify ga/ge?
# - record mutations
# - activation functions
# - final test for new best
# - penalty function for large networks
# - test keras error
# - evaluation K-Fold validation / random number validation
# - test: resume
# --> optimizer test run

# - real time plots & visualisation

# Later
# -----
# - crossover?
# - test variance of accuracy
# - test retraining after load
# - learning code
# - elu etc.
# - Comparison with GE
# - numeric default values
# - ES integers
# - ES categorical values
# - stepwidth mutation, initial stepwidth
# - geometric normal distribution for some variables?

# Done
# ----
# - Input 28*28*3 instead of 32*32*3
# - lenet test code
# - better configuration variables
# - reduced numeric parameters groups to single value for simplicity
# - better metrics
# - phenotypes with \n
