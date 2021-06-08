from utils import util
from hyperParam import no_of_samples, name, threshold, no_of_input_var

# Generating training samples
training_samples, outs = util.fractional_sampling(no_of_samples, name, threshold, no_of_input_var)
training_samples = training_samples.T
print("total samples: ", training_samples.shape)
print("outs: ", outs.shape)