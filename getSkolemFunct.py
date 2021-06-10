from gcln import CLN
from hyperParam import no_of_input_var, threshold, K, input_size, device, name
from imports import np, torch

def get_skolem_function(cln):
    G1 = cln.G1.cpu().detach().numpy()
    G2 = cln.G2.cpu().detach().numpy()
    b1 = cln.b1.cpu().detach().numpy()
    b2 = cln.b2.cpu().detach().numpy()
    spec = "(i_0 | i_1) ^ i_2"

    literals = []
    neg_literals = []
    for i in range(no_of_input_var):
        literals.append("i"+str(i))
        neg_literals.append("~i"+str(i))
    clause = np.array(literals + neg_literals)
    clauses = []
    for i in range(K):
        mask = G1[:, i] > threshold
        clauses.append(clause[mask])
    ored_clauses = []
    for i in range(len(clauses)):
        ored_clauses.append("("+" | ".join(clauses[i])+")")
    ored_clauses = np.array(ored_clauses)
    mask = G2 > threshold
    gated_ored_clauses = np.unique(ored_clauses[mask.flatten()])
    anded_clauses = "i2 = "+"("+" & ".join(gated_ored_clauses)+")"
    print("-----------------------------------------------------------------------------")
    print("skolem function: ",anded_clauses)
    print("-----------------------------------------------------------------------------")
    # spec = spec.replace("i_2", anded_clauses)
    # print(spec)

if __name__ == "__main__":
    cln = CLN(input_size, K, device, name, classify=True, p=0).to(device)
    cln.load_state_dict(torch.load("classifier"))
    # cln.load_state_dict(torch.load("regressor"))
    cln.eval()
    get_skolem_function(cln)