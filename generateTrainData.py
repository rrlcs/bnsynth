# from utils import util
# from hyperParam import no_of_samples, name, threshold, no_of_input_var, P

# Generating training samples
def generateTrainData(P, util, no_of_samples, name, threshold, no_of_input_var):
    if P == 0 or P == 1:
        training_samples = util.fractional_sampling(no_of_samples, name, threshold, no_of_input_var)
        training_samples = training_samples.T
    elif P == 2:
        training_samples = util.fractional_sampling_pos_and_neg(
            no_of_samples, name, threshold, no_of_input_var
            )
    
    return training_samples