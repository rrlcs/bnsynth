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

i0 ,i1 ,i2 = Bools('i0 i1 i2')
i3_2 = Bool('i3_2')
n5,n6,n7,n8,n9,n10,n11,n12,n13,n14 = Bools('n5 n6 n7 n8 n9 n10 n11 n12 n13 n14')
i3_1 = Bool('i3_1')
A = i3_1 == And((Or((i0), (i1), Not(i2))), (Or((i0), (i2), Not(i1))), (Or((i1), (i2), Not(i0))), (Or(Not(i0), Not(i1), Not(i2))))
B = (And((n5 == And((i0),Not(i1),)), (n6 == And(Not(i2),(n5),)), (n7 == And(Not(i0),(i1),)), (n8 == And(Not(i2),(n7),)), (n9 == And(Not(i0),Not(i1),)), (n10 == And((i2),(n9),)), (n11 == And((i0),(i1),)), (n12 == And((i2),(n11),)), (n13 == And(Not(n6),Not(n8),)), (n14 == And(Not(n10),(n13),)), (i3_2  == And(Not(n12),(n14),))))
formula = Implies(And(A,B), i3_1 == i3_2 )
valid(formula)