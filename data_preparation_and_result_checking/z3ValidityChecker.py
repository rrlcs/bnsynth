from z3 import *

#Function to check validity of a formula

def valid(formula):

    s = Solver()

    s.add(Not(formula))

    if s.check() == unsat:

        print("Valid")

        return True

    else:

        # print s.model()

        print("Not Valid")

        return False

def check_validity():
	i_0 ,i_1 ,i_2 = Bools('i_0 i_1 i_2')
	w1 = Bool('w1')
	
	out = Bool('out')
	nn_out0 = And((Or((i_2),Not(i_2),)),((i_2)),)
	nn_out1 = (Or((i_2),Not(i_2),))
	
	w1 = (Xor((i_0),(i_1),))
	out = (Xor((w1),(i_2),))
	z1 = Exists(i_0, out)
	i_0 = nn_out0
	i_1 = nn_out1
	
	w1 = (Xor((i_0),(i_1),))
	out = (Xor((w1),(i_2),))
	z2 = out
	formula = z1==z2
	if valid(formula):
		return 'Valid'
	else:
		return 'Not Valid'