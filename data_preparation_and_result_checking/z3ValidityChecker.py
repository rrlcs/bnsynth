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

i_0 ,i_1 ,i_2 ,i_3 ,i_4 = Bools('i_0 i_1 i_2 i_3 i_4')
w1,w2,w3 = Bools('w1 w2 w3')

out = Bool('out')
nn_out0 = And((Or((i_0),(i_1),(i_2),(i_3),Not(i_0),Not(i_1),Not(i_2),Not(i_3),)),(Or((i_0),(i_1),(i_2),(i_3),Not(i_0),Not(i_1),Not(i_3),)),(Or((i_0),(i_1),(i_2),Not(i_0),Not(i_1),Not(i_2),Not(i_3),)),(Or((i_0),(i_1),(i_2),Not(i_0),Not(i_1),Not(i_3),)),(Or((i_0),(i_2),(i_3),Not(i_0),Not(i_1),Not(i_2),Not(i_3),)),)

w2 = (And((i_2),(i_3),))
w1 = (Or((i_0),(i_1),))
w3 = (Or((w1),(w2),))
out = (Xor((w3),(i_4),))
z1 = Exists(i_4, out)
i_4 = nn_out0

w2 = (And((i_2),(i_3),))
w1 = (Or((i_0),(i_1),))
w3 = (Or((w1),(w2),))
out = (Xor((w3),(i_4),))
z2 = out
formula = z1==z2
valid(formula)