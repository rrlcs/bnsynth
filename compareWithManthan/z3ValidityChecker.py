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

i0 ,i1 = Bools('i0 i1')
i2_2 = Bool('i2_2')
i2_1 = Bool('i2_1')

i2 = Bool('i2')
A = i2_1 == (Or((i0),(i1),))
B = (And((i2_2 == And((i0),(i1),))))
formula = Implies(And(A,B), Implies(i2_1, i2_2))
valid(formula)