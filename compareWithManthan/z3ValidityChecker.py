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
w1 = Bool('w1')

out = Bool('out')
A = out_1 == ((i_0))
B = (And((w1 == Xor((i_0),(i_1),)), (out_2 == Xor((w1),(i_2),))))
formula = Implies(And(A,B), Implies(out_1, out_2))
valid(formula)