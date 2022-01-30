import numpy as np
import torch.nn as nn


def get_skolem_function(gcln, no_of_input_var, input_var_idx, num_of_outputs, output_var_idx, io_dict, threshold, K):
    '''
    Input: Learned model parameters
    Output: Skolem Functions
    Functionality: Reads the model weights (G1, G2) and builds the skolem function based on it.
    '''

    # sigmoid = nn.Sigmoid()
    layer_or_weights = gcln.layer_or_weights.cpu().detach().numpy() # input_size x K
    layer_and_weights = gcln.layer_and_weights.cpu().detach().numpy() # K x num_of_outputs

    literals = []
    neg_literals = []
    for i in input_var_idx:
        literals.append(io_dict.get(i.item()))
        neg_literals.append("~"+io_dict.get(i.item()))
    for i in range(len(neg_literals)):
        neg_literals[i] = neg_literals[i].replace(" ", "")
    print(neg_literals)
    clause = np.array(literals + neg_literals)

    clauses = []
    for j in range(num_of_outputs * K):
        mask = layer_or_weights[:, j] > threshold
        clauses.append(clause[mask])
    clauses = np.array(clauses)


    ored_clauses = []
    for j in range(len(clauses)):
        ored_clauses.append("("+" | ".join(clauses[j])+")")
    ored_clauses = np.array(ored_clauses)


    gated_ored_clauses = []
    for i in range(num_of_outputs):
        mask = layer_and_weights[i*K:(i+1)*K, :] > threshold
        ored_clause = ored_clauses.reshape((-1, 1))[i*K:(i+1)*K, :]
        gated_ored_clauses.append(
            np.unique(ored_clause[mask]))

    skfs = []
    for i in range(num_of_outputs):
        skf = " & ".join(gated_ored_clauses[i])+"\n"
        if " & ()" in skf:
            skf = skf.replace(" & ()", "")
        if "() & " in skf:
            skf = skf.replace("() & ", "")
        skfs.append(skf)

    # print("-----------------------------------------------------------------------------")
    print("skolem function in getSkolemFunc4z3.py: ", skfs)
    # print("-----------------------------------------------------------------------------")

    return skfs
