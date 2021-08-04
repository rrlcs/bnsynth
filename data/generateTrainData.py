# Generating training samples
def generateTrainData(P, util, no_of_samples, name, spec, threshold, no_of_input_var, correlated_sampling):
    if P == 0 or P == 1:
        if correlated_sampling:
            training_samples = util.correlated_fractional_sampling(no_of_samples, name, threshold, no_of_input_var)
        else:
            training_samples = util.fractional_sampling(no_of_samples, util, name, threshold, no_of_input_var)
        # training_samples = training_samples.T
    elif P == 2:
        if correlated_sampling:
            training_samples = util.correlated_fractional_sampling(
            no_of_samples, name, threshold, no_of_input_var, spec
            )
        else:
            training_samples = util.fractional_sampling_pos_and_neg(
                no_of_samples, name, threshold, no_of_input_var, spec
                )
    # else:
    #     training_samples = util.correlated_fractional_sampling(no_of_samples, name, threshold, no_of_input_var)
    
    return training_samples