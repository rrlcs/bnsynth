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

i0, i1, i2_2 = Bools('i0 i1 i2_2')
i2_2 = Bool('i2_2')
zero = Bool('zero')
one = Bool('one')
wi1 = Bool('wi1')
wt2 = Bool('wt2')

out = Bool('out')
i2_1 = Bool('i2_1')
A = i2_1 == And((Or((i0), Not(i1))), (Or((i1), Not(i0))))
B = (And((zero == (Int(0))), (one == (Int(1))), (wi1 == (((one)))), (wt2 == (Not(Xor((wi1),(i1),)))), (out == (wt2))))
formula = Implies(And(A,B), i2_1 == i2_2)
valid(formula)