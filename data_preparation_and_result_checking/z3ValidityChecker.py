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
	i_0 ,i_1 ,i_2 ,i_3 = Bools('i_0 i_1 i_2 i_3')
	w1,w2 = Bools('w1 w2')
	
	out = Bool('out')
	nn_out0 = And((Or((i_0),(i_1),Not(i_1),Not(i_2),)),(Or((i_0),(i_2),Not(i_0),Not(i_1),)),(Or((i_0),(i_2),Not(i_1),)),(Or((i_1),(i_2),Not(i_0),)),(Or(Not(i_0),Not(i_1),Not(i_2),)),)
	
	w2 = (Xor((i_2),(i_3),))
	w1 = (Xor((i_0),(i_1),))
	out = (Xor((w1),(w2),))
	z1 = Exists(i_3, out)
	i_3 = nn_out0
	
	w2 = (Xor((i_2),(i_3),))
	w1 = (Xor((i_0),(i_1),))
	out = (Xor((w1),(w2),))
	z2 = out
	formula = z1==z2
	if valid(formula):
		return 'Valid'
	else:
		return 'Not Valid'