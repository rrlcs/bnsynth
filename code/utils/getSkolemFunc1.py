import io
import numpy as np
import torch.nn as nn

def get_skolem_function(gcln, no_of_input_var, input_var_idx, num_of_outputs, output_var_idx, io_dict, threshold, K):
	'''
	Input: Learned model parameters
	Output: Skolem Functions
	Functionality: Reads the model weights (G1, G2) and builds the skolem function based on it.
	'''

	sigmoid = nn.Sigmoid()
	G11 = sigmoid(gcln.G11.cpu().detach()).numpy()
	G21 = sigmoid(gcln.G21.cpu().detach()).numpy()
	G12 = sigmoid(gcln.G12.cpu().detach()).numpy()
	G22 = sigmoid(gcln.G22.cpu().detach()).numpy()
	b1 = gcln.b1.cpu().detach().numpy()
	b2 = gcln.b2.cpu().detach().numpy()

	literals = []
	neg_literals = []
	for i in input_var_idx:
		literals.append(io_dict.get(i.item()))
		neg_literals.append("~"+io_dict.get(i.item()))
	clause = np.array(literals + neg_literals)
	clauses = []
	for i in range(K):
		mask1 = G11[:, i] > threshold
		clauses.append(clause[mask1])
	for i in range(K):
		mask2 = G11[:, i] > threshold
		clauses.append(clause[mask2])
	# print(mask)
	# clauses = np.array(clauses)#.reshape(K, num_of_outputs)
	# print("shape of clauses: ", clauses[:K])
	ored_clauses = []
	for i in range(len(clauses)):
		ored_clauses.append("("+" | ".join(clauses[i])+")".replace("() &", ""))
	ored_clauses = np.array(ored_clauses)
	# print("ored clauses shape: ", ored_clauses[:K])

	masks = []

	# for i in range(num_of_outputs):
		# mask = G2[i*K:(i+1)*K,i] > threshold
	mask1 = G21 > threshold
	# print(mask.shape)
	masks.append(mask1.reshape((-1, 1)))
	mask2 = G22 > threshold
	masks.append(mask2.reshape((-1, 1)))
	print(len(masks))
	gated_ored_clauses = []
	for i in range(num_of_outputs):
		ored = ored_clauses[i*K:(i+1)*K]
		gated_ored_clauses.append(np.unique(ored[masks[i][:,0].flatten()]))
	print(gated_ored_clauses)
	# gated_ored_clauses = np.unique(ored_clauses[mask.flatten()])
	# print("len of gated or: ", gated_ored_clauses)
	anded_clauses = []
	# anded_clauses2 = []
	for i in range(len(output_var_idx)):
		anded_clauses.append(str(io_dict.get(output_var_idx[i]))+" = "+"("+" & ".join(gated_ored_clauses[i])+")")
	# anded_clauses = str(io_dict.get(output_var_pos))+" = "+"("+" & ".join(gated_ored_clauses)+")"
	skfs = []
	for i in range(num_of_outputs):
		skfs.append(" & ".join(gated_ored_clauses[i])+"\n")
	# skf = " & ".join(gated_ored_clauses)+"\n"
	print("-----------------------------------------------------------------------------")
	print("skolem function: ",anded_clauses)
	print("-----------------------------------------------------------------------------")
	# spec = spec.replace("i_2", anded_clauses)
	# print(spec)
	return skfs
