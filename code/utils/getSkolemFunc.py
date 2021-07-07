import numpy as np

def get_skolem_function(cln, no_of_input_var, threshold, K):
	G1 = cln.G1.cpu().detach().numpy()
	G2 = cln.G2.cpu().detach().numpy()
	b1 = cln.b1.cpu().detach().numpy()
	b2 = cln.b2.cpu().detach().numpy()
	spec = "(i_0 | i_1) ^ i_2"

	literals = []
	neg_literals = []
	for i in range(no_of_input_var):
		literals.append("i_"+str(i))
		neg_literals.append("~i_"+str(i))
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
	anded_clauses = "i"+str(no_of_input_var)+" = "+"("+" & ".join(gated_ored_clauses)+")"
	f = open("nn_output", "w")
	f.write(" & ".join(gated_ored_clauses))
	print("-----------------------------------------------------------------------------")
	print("skolem function: ",anded_clauses)
	print("-----------------------------------------------------------------------------")
	# spec = spec.replace("i_2", anded_clauses)
	# print(spec)
