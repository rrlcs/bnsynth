import numpy as np
import torch.nn as nn


def get_skolem_function(gcln, no_of_input_var, input_var_idx, num_of_outputs, output_var_idx, io_dict, threshold, K):
    '''
    Input: Learned model parameters
    Output: Skolem Functions
    Functionality: Reads the model weights (G1, G2) and builds the skolem function based on it.
    '''

    sigmoid = nn.Sigmoid()
    layer_or_weights = []
    layer_and_weights = []
    for i in range(num_of_outputs):
        layer_or_weights.append(sigmoid(gcln.layer_or_weights[i].cpu().detach()).numpy()) # input_size x K
        layer_and_weights.append(sigmoid(gcln.layer_and_weights[i].cpu().detach()).numpy()) # K x num_of_outputs

    literals = []
    neg_literals = []
    for i in input_var_idx:
        literals.append("i"+str(i.item()))
        neg_literals.append("~i"+str(i.item()))

    clause = np.array(literals + neg_literals)

    clauses = []
    for i in range(num_of_outputs):
        G1 = layer_or_weights[i]
        clauses_per_output = []
        for j in range(K):
            mask = G1[:, j] > threshold
            clauses_per_output.append(clause[mask])
        clauses.append(np.array(clauses_per_output))

    ored_clauses = []
    for i in range(num_of_outputs):
        ored_clause = []
        clause = clauses[i]
        print(clause.shape)
        for j in range(len(clause)):
            ored_clause.append("("+" | ".join(clause[i])+")".replace("() &", ""))
        ored_clauses.append(np.array(ored_clause))
    print(ored_clauses)

    gated_ored_clauses = []
    for i in range(num_of_outputs):
        G2 = layer_and_weights[i]
        mask = G2 > threshold
        ored_clause = ored_clauses[i].reshape((-1, 1))
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
    # print("skolem function in getSkolemFunc.py: ", skfs)
    # print("-----------------------------------------------------------------------------")

    return skfs
