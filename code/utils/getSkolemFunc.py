import numpy as np
import torch.nn as nn


def get_skolem_function(gcln, no_of_input_var, input_var_idx, num_of_outputs, output_var_idx, io_dict, threshold, K):
    '''
    Input: Learned model parameters
    Output: Skolem Functions
    Functionality: Reads the model weights (G1, G2) and builds the skolem function based on it.
    '''

    sigmoid = nn.Sigmoid()
    G1 = sigmoid(gcln.G1.cpu().detach()).numpy()
    G2 = sigmoid(gcln.G2.cpu().detach()).numpy()

    literals = []
    neg_literals = []
    for i in input_var_idx:
        literals.append(io_dict.get(i.item()))
        neg_literals.append("~"+io_dict.get(i.item()))

    clause = np.array(literals + neg_literals)
    clauses = []
    for i in range(num_of_outputs * K):
        mask = G1[:, i] > threshold
        clauses.append(clause[mask])

    ored_clauses = []
    for i in range(len(clauses)):
        ored_clauses.append("("+" | ".join(clauses[i])+")".replace("() &", ""))
    ored_clauses = np.array(ored_clauses)

    masks = []
    for i in range(num_of_outputs):
        mask = G2[i*K:(i+1)*K, :] > threshold
        masks.append(mask.reshape((-1, 1)))
    
    gated_ored_clauses = []
    for i in range(num_of_outputs):
        gated_ored_clauses.append(
            np.unique(ored_clauses[masks[i][:, 0].flatten()]))

    anded_clauses = []
    for i in range(num_of_outputs):
        anded_clauses.append(str(io_dict.get(
            output_var_idx[i]))+" = "+"("+" & ".join(gated_ored_clauses[i])+")")

    skfs = []
    for i in range(num_of_outputs):
        skfs.append(" & ".join(gated_ored_clauses[i])+"\n")

    print("-----------------------------------------------------------------------------")
    print("skolem function: ", anded_clauses)
    print("-----------------------------------------------------------------------------")

    return skfs
