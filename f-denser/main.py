import fast_denser
fast_denser.search(0, 'mnist', 'config/config.json', 'config/lenet.grammar')

from pickle import load

with open('D:/experiments/run_0/evaluator.pkl', 'rb') as f_data:
    evaluator = load(f_data)
    x_test = evaluator.dataset['x_test']
    y_test = evaluator.dataset['y_test']

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import visualkeras

model = load_model('D:/experiments/run_0/best.h5')
datagen_test = ImageDataGenerator(rescale=1/255.0)

model.summary(line_length=120)
# visualkeras.layered_view(model, legend=True)

# Todo
# - check dataset usage
# - simplify specification?
# - mutation test code
# - crossover test code
# - better configuration variables
# - penalty function for large networks
# --> optimizer test run

# - real time plots & visualisation

# Later
# -----
# - learning code
# - elu etc.
# - single numeric values
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
