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

i0, i1 = Bools('i0 i1')
i21, i22 = Bools('i21 i22')
n4, n5 = Bools('n4 n5')

A = i21 == And((Or(Not(i0), i1)), (Or(i0, Not(i1))))
B = (And((n4 == And((i0), Not(i1))), (n5 == And(Not(i0), (i1))), (i22 == And(Not(n4), Not(n5)))))
formula = Implies(And(A,B), i21 == i22)

valid(formula)
