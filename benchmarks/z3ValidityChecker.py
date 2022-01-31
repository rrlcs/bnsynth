from z3 import *


def generate_all_counterexamples(input_formula):
    
    all_counterexamples = []
    i = 0

    s = Solver()
    s.add(Not(input_formula))
    # print(s.check())

    model_dict = {}
    if s.check() == sat:
        model = s.model()
        all_counterexamples.append(model)
        # print("Counterexample model generated in round:", i)
        # print(model)
        
        #Do this for every variable in the model. decls(). Please note that we will use DISJUNCTION OF EACH VARIABLE notation. Hence, Or(i0!=False, i1!=True, ...)
        additional_constraints = []
        for j in range(len(model.decls())):
            x = model.decls()[j]
            d = x
            if d.name() not in model_dict:
                model_dict[d.name()] = [0] if str(model[d])=="False" else [1]
            else:
                model_dict[d.name()].append(0) if str(model[d])=="False" else model_dict[d.name()].append(1)
            # print(x)
            additional_constraints.append(Bool(x.name()) != model[x])
        s.add(Or(additional_constraints[:]))
        i += 1
    # print("model_dict: ", model_dict)
    return model_dict

# if __name__ == "__main__":
#     i_0 ,i_1, i_2 = Bools('i_0 i_1 i_2')
#     w1 = Bool('w1')
#     w1 = Xor(i_0, i_1)
#     lhs = Exists([i_2], Xor(w1, i_2))
#     i_2 = And((Or(i_0, (i_1))), (Or((i_0), i_1)))
#     rhs = Xor(w1, i_2)
#     print(lhs, rhs)
#     formula = lhs == rhs
#     # formula = Xor(i_0, i_1, i_2) == Xor(w1, i_2)
#     all_counterexamples = generate_all_counterexamples(formula)
#     print("\n List of all counterexamples for the given formula: ")
#     print(all_counterexamples)

def check_validity():
	i_0 ,i_1 = Bools('i_0 i_1')
	
	out = Bool('out')
	nn_out0 = simplify((Not(i_0)))
	
	out = (Or((i_0),(i_1),))
	z = Exists([i_1], out)
	
	i_1 = nn_out0
	
	out = (Or((i_0),(i_1),))
	z1 = out
	formula = z==z1
	all_counterexamples = generate_all_counterexamples(formula)
	print('all_counterexamples', all_counterexamples)
	if len(all_counterexamples) == 0:
		return True, all_counterexamples
	else:
		return False, all_counterexamples

if __name__ == "__main__":
    is_valid, ce = check_validity()
    # print(is_valid, ce)
