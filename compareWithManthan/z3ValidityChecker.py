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
out_2 = Bool('out_2')
out_1 = Bool('out_1')
w1,w2 = Bools('w1 w2')

out = Bool('out')
i_2 = And(((i_0)),((i_1)),)
w1 = Xor((i_0),(i_1),)
w2 = And((i_1),(i_2),)
out = Xor((w1),(w2),)

formula = out
valid(formula)