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

i_0 ,i_1 ,i_2 = Bools('i_0 i_1 i_2')
w1 = Bool('w1')

out = Bool('out')
nn_out = And((Or((i_0),(i_1),Not(i_0),Not(i_1),)),(Or((i_0),Not(i_0),Not(i_1),)),(Or((i_0),Not(i_1),)),(Or((i_1),Not(i_0),Not(i_1),)),(Or((i_1),Not(i_0),)),)
w1 = Xor((i_0),(i_1),)
out = Xor((w1),(i_2),)
z1 = Exists(i_2, out)
i_2 = nn_out
w1 = Xor((i_0),(i_1),)
out = Xor((w1),(i_2),)
z2 = out
formula = z1==z2
valid(formula)