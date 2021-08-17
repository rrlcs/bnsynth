import io
import numpy as np

def get_skolem_function(gcln, no_of_input_var, input_var_idx, output_var_pos, io_dict, threshold, K):
	'''
	Input: Learned model parameters
	Output: Skolem Functions
	Functionality: Reads the model weights (G1, G2) and builds the skolem function based on it.
	'''

	G1 = gcln.G1.cpu().detach().numpy()
	G2 = gcln.G2.cpu().detach().numpy()
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
		mask = G1[:, i] > threshold
		clauses.append(clause[mask])
	ored_clauses = []
	for i in range(len(clauses)):
		ored_clauses.append("("+" | ".join(clauses[i])+")".replace("() &", ""))
	ored_clauses = np.array(ored_clauses)
	mask = G2 > threshold
	gated_ored_clauses = np.unique(ored_clauses[mask.flatten()])
	anded_clauses = str(io_dict.get(output_var_pos))+" = "+"("+" & ".join(gated_ored_clauses)+")"
	skf = " & ".join(gated_ored_clauses)+"\n"
	print("-----------------------------------------------------------------------------")
	print("skolem function: ",anded_clauses)
	print("-----------------------------------------------------------------------------")
	# spec = spec.replace("i_2", anded_clauses)
	# print(spec)
	return skf
