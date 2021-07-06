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

i0 ,i1 ,i2 ,i3 = Bools('i0 i1 i2 i3')
i4_2 = Bool('i4_2')
i4_1 = Bool('i4_1')
n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,n16 = Bools('n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16')

i4 = Bool('i4')
A = i4_1 == $$
B = (And((n6 == And(Not(i0),Not(i3),)), (n7 == And(Not(i1),(n6),)), (n8 == And(Not(i2),(i3),)), (n9 == And(Not(i0),(n8),)), (n10 == And(Not(i1),(n9),)), (n11 == And(Not(n7),Not(n10),)), (n12 == And((i2),(i3),)), (n13 == And(Not(i1),(n12),)), (n14 == And((i0),(n13),)), (n15 == And((n11),Not(n14),)), (n16 == And((i1),(n12),)), (i4_2 == Or(Not(n15),(n16),))))
formula = Implies(And(A,B), Implies(i4_1, i4_2))
valid(formula)