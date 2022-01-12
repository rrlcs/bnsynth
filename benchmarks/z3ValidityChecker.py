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
	i_10 ,i_1 ,i_22 = Bools('i_10 i_1 i_22')
	w1 = Bool('w1')
	
	out = Bool('out')
	nn_out0 = simplify(And((Or((x_0),(x_4),(x_5),(x_6),Not(x_0),Not(x_4),Not(x_6),)),(Or((x_0),(x_4),(x_5),(x_6),Not(x_0),Not(x_5),Not(x_6),)),(Or((x_0),(x_4),(x_5),(x_6),Not(x_5),Not(x_6),)),(Or((x_0),(x_4),(x_5),Not(x_0),Not(x_5),Not(x_6),)),(Or((x_0),(x_4),(x_5),Not(x_0),Not(x_6),)),(Or((x_0),(x_4),(x_6),Not(x_0),Not(x_4),Not(x_5),)),(Or((x_0),(x_4),(x_6),Not(x_0),Not(x_4),Not(x_6),)),(Or((x_0),(x_4),(x_6),Not(x_0),Not(x_5),)),(Or((x_0),(x_4),(x_6),Not(x_4),Not(x_6),)),(Or((x_0),(x_5),Not(x_0),Not(x_5),)),(Or((x_0),Not(x_4),Not(x_5),Not(x_6),)),(Or((x_4),(x_5),(x_6),Not(x_4),Not(x_5),Not(x_6),)),(Or((x_4),(x_5),(x_6),)),(Or((x_4),(x_5),Not(x_5),)),(Or((x_4),(x_6),Not(x_4),Not(x_5),Not(x_6),)),(Or((x_4),(x_6),Not(x_4),Not(x_5),)),(Or((x_4),(x_6),Not(x_6),)),(Or((x_4),Not(x_0),)),(Or((x_4),Not(x_4),Not(x_5),Not(x_6),)),(Or((x_6),Not(x_0),)),))
	nn_out1 = simplify(And((Or((x_0),(x_4),(x_5),(x_6),Not(x_0),Not(x_4),Not(x_5),)),(Or((x_0),(x_4),(x_5),(x_6),Not(x_0),Not(x_4),)),(Or((x_0),(x_4),(x_5),(x_6),Not(x_5),Not(x_6),)),(Or((x_0),(x_4),(x_5),(x_6),Not(x_5),)),(Or((x_0),(x_4),(x_5),(x_6),Not(x_6),)),(Or((x_0),(x_4),(x_5),(x_6),)),(Or((x_0),(x_4),(x_5),Not(x_0),Not(x_4),)),(Or((x_0),(x_4),(x_5),Not(x_0),Not(x_5),Not(x_6),)),(Or((x_0),(x_4),(x_6),Not(x_5),Not(x_6),)),(Or((x_0),(x_4),Not(x_0),Not(x_4),)),(Or((x_0),(x_5),(x_6),Not(x_0),Not(x_6),)),(Or((x_0),(x_5),Not(x_5),Not(x_6),)),(Or((x_0),Not(x_0),Not(x_4),Not(x_5),Not(x_6),)),(Or((x_4),(x_5),(x_6),Not(x_0),Not(x_4),Not(x_5),)),(Or((x_4),(x_5),Not(x_4),Not(x_5),)),(Or((x_4),(x_5),)),(Or((x_4),(x_6),Not(x_0),Not(x_5),)),(Or((x_5),(x_6),Not(x_0),Not(x_4),)),))
	nn_out2 = simplify(And((Or((x_0),(x_4),(x_5),(x_6),Not(x_0),Not(x_4),Not(x_6),)),(Or((x_0),(x_4),(x_5),(x_6),Not(x_0),Not(x_6),)),(Or((x_0),(x_4),(x_5),(x_6),Not(x_4),Not(x_5),Not(x_6),)),(Or((x_0),(x_4),(x_5),(x_6),Not(x_5),Not(x_6),)),(Or((x_0),(x_4),(x_5),Not(x_0),Not(x_4),)),(Or((x_0),(x_4),(x_6),Not(x_0),Not(x_6),)),(Or((x_0),(x_4),Not(x_6),)),(Or((x_0),(x_5),(x_6),Not(x_0),Not(x_5),Not(x_6),)),(Or((x_0),(x_5),(x_6),Not(x_4),Not(x_5),Not(x_6),)),(Or((x_0),(x_5),(x_6),Not(x_5),Not(x_6),)),(Or((x_0),(x_5),Not(x_0),Not(x_5),)),(Or((x_0),(x_5),Not(x_0),)),(Or((x_0),(x_6),)),(Or((x_4),(x_5),Not(x_0),Not(x_6),)),(Or((x_4),(x_6),Not(x_0),Not(x_5),)),(Or((x_5),(x_6),)),(Or(Not(x_0),Not(x_4),Not(x_6),)),))
	nn_out3 = simplify(And((Or((x_0),(x_5),)),(Or((x_4),(x_5),(x_6),Not(x_0),Not(x_4),)),(Or((x_4),(x_5),(x_6),Not(x_0),Not(x_5),Not(x_6),)),(Or((x_4),(x_5),(x_6),Not(x_0),Not(x_5),)),(Or((x_4),(x_5),(x_6),Not(x_5),)),(Or((x_4),(x_5),(x_6),Not(x_6),)),(Or((x_4),(x_5),Not(x_0),Not(x_5),)),(Or((x_4),(x_6),Not(x_0),)),(Or((x_4),(x_6),Not(x_6),)),(Or((x_5),(x_6),Not(x_0),Not(x_4),)),(Or((x_5),(x_6),Not(x_0),Not(x_6),)),(Or((x_5),(x_6),Not(x_0),)),(Or(Not(x_0),Not(x_5),)),))
	nn_out4 = simplify(And((Or((x_0),(x_4),(x_6),)),(Or((x_0),Not(x_4),Not(x_6),)),(Or((x_4),(x_5),(x_6),)),(Or((x_4),(x_6),Not(x_0),Not(x_4),Not(x_5),)),(Or((x_4),Not(x_0),Not(x_5),Not(x_6),)),(Or((x_5),Not(x_4),Not(x_6),)),(Or((x_6),Not(x_0),Not(x_4),Not(x_5),)),))
	nn_out5 = simplify(And((Or((x_0),(x_4),(x_5),Not(x_6),)),(Or((x_0),(x_4),Not(x_0),Not(x_4),Not(x_5),)),(Or((x_0),(x_4),Not(x_6),)),(Or((x_0),(x_5),(x_6),Not(x_0),Not(x_4),Not(x_6),)),(Or((x_0),(x_5),(x_6),Not(x_0),Not(x_6),)),(Or((x_0),(x_5),(x_6),Not(x_4),)),(Or((x_0),(x_5),(x_6),Not(x_6),)),(Or((x_0),(x_5),Not(x_4),Not(x_6),)),(Or((x_0),(x_6),Not(x_5),)),(Or((x_4),(x_5),(x_6),Not(x_0),)),(Or((x_4),(x_5),(x_6),Not(x_4),Not(x_5),Not(x_6),)),(Or((x_4),(x_5),(x_6),Not(x_4),Not(x_6),)),(Or((x_4),(x_5),(x_6),Not(x_5),Not(x_6),)),(Or((x_4),(x_5),Not(x_0),Not(x_4),Not(x_5),Not(x_6),)),(Or((x_5),(x_6),Not(x_4),Not(x_6),)),(Or(Not(x_0),Not(x_4),Not(x_6),)),))
	nn_out6 = simplify(And((Or((x_0),(x_4),(x_5),(x_6),Not(x_0),Not(x_4),Not(x_5),Not(x_6),)),(Or((x_0),(x_4),(x_5),(x_6),Not(x_0),Not(x_5),Not(x_6),)),(Or((x_0),(x_4),(x_5),Not(x_0),Not(x_4),Not(x_5),Not(x_6),)),(Or((x_0),(x_4),(x_5),)),(Or((x_0),(x_5),Not(x_4),Not(x_5),Not(x_6),)),(Or((x_0),(x_5),Not(x_6),)),(Or((x_0),(x_6),Not(x_4),Not(x_5),)),(Or((x_4),(x_5),(x_6),Not(x_0),Not(x_4),Not(x_5),Not(x_6),)),(Or((x_4),(x_5),(x_6),Not(x_0),Not(x_5),Not(x_6),)),(Or((x_4),(x_6),Not(x_4),)),(Or((x_5),(x_6),Not(x_0),)),(Or((x_5),Not(x_0),Not(x_5),Not(x_6),)),(Or((x_5),Not(x_4),Not(x_5),Not(x_6),)),))
	nn_out7 = simplify(And((Or((x_0),(x_4),(x_5),(x_6),Not(x_0),Not(x_4),Not(x_5),)),(Or((x_0),(x_4),(x_5),(x_6),Not(x_0),Not(x_5),Not(x_6),)),(Or((x_0),(x_4),(x_5),(x_6),Not(x_0),Not(x_6),)),(Or((x_0),(x_4),(x_5),(x_6),Not(x_0),)),(Or((x_0),(x_4),(x_5),Not(x_0),Not(x_4),Not(x_5),)),(Or((x_0),(x_4),(x_5),Not(x_0),Not(x_4),)),(Or((x_0),(x_4),(x_5),Not(x_0),Not(x_5),)),(Or((x_0),(x_4),(x_5),Not(x_4),Not(x_5),)),(Or((x_0),(x_4),(x_5),Not(x_4),Not(x_6),)),(Or((x_0),(x_4),(x_5),Not(x_5),Not(x_6),)),(Or((x_0),(x_4),(x_6),Not(x_0),Not(x_6),)),(Or((x_0),(x_4),(x_6),Not(x_4),Not(x_6),)),(Or((x_0),(x_4),Not(x_0),Not(x_4),Not(x_6),)),(Or((x_0),(x_4),Not(x_0),Not(x_5),Not(x_6),)),(Or((x_0),(x_4),Not(x_0),)),(Or((x_0),(x_4),Not(x_4),Not(x_5),Not(x_6),)),(Or((x_0),(x_4),Not(x_5),Not(x_6),)),(Or((x_0),(x_5),(x_6),Not(x_5),Not(x_6),)),(Or((x_0),(x_5),Not(x_0),)),(Or((x_0),(x_5),Not(x_6),)),(Or((x_0),(x_6),Not(x_5),)),(Or((x_4),(x_5),(x_6),Not(x_0),Not(x_4),Not(x_6),)),(Or((x_4),(x_5),(x_6),Not(x_0),Not(x_6),)),(Or((x_4),(x_5),(x_6),Not(x_0),)),(Or((x_4),(x_5),(x_6),Not(x_5),Not(x_6),)),(Or((x_4),(x_5),Not(x_0),Not(x_5),Not(x_6),)),(Or((x_4),(x_5),Not(x_4),Not(x_5),Not(x_6),)),(Or((x_4),(x_5),Not(x_5),Not(x_6),)),(Or((x_4),(x_5),Not(x_5),)),((x_4)),(Or(Not(x_0),Not(x_4),Not(x_5),Not(x_6),)),))
	nn_out8 = simplify(And((Or((x_0),(x_4),(x_5),(x_6),Not(x_0),Not(x_4),Not(x_5),Not(x_6),)),(Or((x_0),(x_4),(x_5),(x_6),Not(x_0),Not(x_5),Not(x_6),)),(Or((x_0),(x_4),(x_5),Not(x_0),Not(x_4),Not(x_5),)),(Or((x_0),(x_4),(x_6),Not(x_0),Not(x_4),Not(x_5),Not(x_6),)),(Or((x_0),(x_4),(x_6),Not(x_0),Not(x_4),Not(x_6),)),(Or((x_0),(x_4),(x_6),Not(x_0),Not(x_5),Not(x_6),)),(Or((x_0),(x_4),(x_6),Not(x_0),Not(x_6),)),(Or((x_0),(x_4),Not(x_0),Not(x_4),Not(x_5),Not(x_6),)),(Or((x_0),(x_4),Not(x_0),Not(x_4),Not(x_6),)),(Or((x_0),(x_4),Not(x_0),)),(Or((x_0),(x_5),(x_6),Not(x_0),Not(x_5),)),(Or((x_0),(x_5),(x_6),Not(x_4),Not(x_5),)),(Or((x_0),(x_5),Not(x_0),Not(x_4),Not(x_5),)),(Or((x_0),(x_5),Not(x_6),)),(Or((x_4),(x_5),(x_6),Not(x_0),Not(x_5),Not(x_6),)),(Or((x_4),(x_5),)),(Or((x_4),(x_6),Not(x_0),Not(x_4),Not(x_6),)),(Or((x_4),(x_6),Not(x_4),Not(x_6),)),(Or((x_5),(x_6),Not(x_0),Not(x_4),)),(Or((x_5),Not(x_0),Not(x_5),Not(x_6),)),))
	
	w1 = (Xor((i_10),(i_1),))
	o_1 = (Xor((w1),(i_22),))
	z = Exists([i_1, i_22], o_1)
	
	i_1 = nn_out0
	i_22 = nn_out1
	
	w1 = (Xor((i_10),(i_1),))
	o_1 = (Xor((w1),(i_22),))
	z2 = o_1
	formula = z==z2
	all_counterexamples = generate_all_counterexamples(formula)
	print('all_counterexamples', all_counterexamples)
	if len(all_counterexamples) == 0:
		return True, all_counterexamples
	else:
		return False, all_counterexamples

if __name__ == "__main__":
    is_valid, ce = check_validity()
    # print(is_valid, ce)
