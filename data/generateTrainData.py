# Generating training samples
def generateTrainData(P, util, no_of_samples, threshold, num_of_vars, input_var_idx, correlated_sampling):
    if P == 0 or P == 1:
        if correlated_sampling:
            training_samples = util.correlated_fractional_sampling(no_of_samples, util, threshold, num_of_vars)
        else:
            training_samples = util.fractional_sampling(no_of_samples, util, threshold, num_of_vars)
    elif P == 2:
        if correlated_sampling:
            training_samples = util.correlated_fractional_sampling(
            no_of_samples, util, threshold, num_of_vars
            )
        else:
            training_samples = util.fractional_sampling_pos_and_neg(
                no_of_samples, util, threshold, num_of_vars
                )
    
    return training_samples