import numpy as np
import torch.nn as nn


def get_skolem_function(gcln, no_of_input_var, input_var_idx, num_of_outputs, output_var_idx, io_dict, threshold, K):
    '''
    Input: Learned model parameters
    Output: Skolem Functions
    Functionality: Reads the model weights (G1, G2) and builds the skolem function based on it.
    '''

    # sigmoid = nn.Sigmoid()
    gcln.G1.data.clamp_(0.0, 1.0)
    gcln.G2.data.clamp_(0.0, 1.0)
    G1 = (gcln.G1.cpu().detach()).numpy() # input_size x K
    G2 = (gcln.G2.cpu().detach()).numpy() # K x num_of_outputs
    print("===========================================")
    print("gates g1: ", G1)
    print("gates g2: ", G2)

    literals = []
    neg_literals = []
    for i in input_var_idx:
        literals.append("i"+str(i.item()))
        neg_literals.append("~i"+str(i.item()))

    clause = np.array(literals + neg_literals)

    clauses = []
    for i in range(K):
        mask = G1[:, i] > threshold
        clauses.append(clause[mask])

    ored_clauses = []
    for i in range(len(clauses)):
        ored_clauses.append("("+" | ".join(clauses[i])+")")
    ored_clauses = np.array(ored_clauses)

    gated_ored_clauses = []
    for i in range(num_of_outputs):
        mask = G2[:, i] > threshold
        gated_ored_clauses.append(
            np.unique(ored_clauses[mask]))

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
