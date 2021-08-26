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

x_0, i_1, i_2, i_3, x_4, x_5, x_6, i_7, i_8, i_9, i_10, i_11, i_12 = Bools('x_0 i_1 i_2 i_3 x_4 x_5 x_6 i_7 i_8 i_9 i_10 i_11 i_12')
c1 = Bool('c1')
carry1 = Bool('carry1')
c2 = Bool('c2')
carry2 = Bool('carry2')
a1 = Bool('a1')
a2 = Bool('a2')
a3 = Bool('a3')
c3 = Bool('c3')
a4 = Bool('a4')
c4 = Bool('c4')
a5 = Bool('a5')
c5 = Bool('c5')
a6 = Bool('a6')
c6 = Bool('c6')
a7 = Bool('a7')

out = Bool('out')
nn_out0 = And((Or((x_0),(x_4),Not(x_5),)),(Or((x_0),Not(x_4),)),(Or((x_4),(x_5),Not(x_6),)),(Or((x_6),Not(x_4),)),)
nn_out1 = And((Or((x_0),(x_5),)),(Or((x_0),(x_6),Not(x_0),)),(Or((x_0),(x_6),Not(x_5),)),(Or((x_0),Not(x_0),Not(x_4),)),(Or((x_5),Not(x_5),Not(x_6),)),(Or((x_5),Not(x_5),)),)
nn_out2 = And((Or((x_0),(x_4),Not(x_6),)),(Or((x_0),(x_6),)),(Or((x_4),(x_5),Not(x_4),Not(x_5),)),(Or((x_6),Not(x_6),)),)
nn_out3 = And((Or((x_0),Not(x_0),Not(x_4),Not(x_5),)),(Or((x_4),(x_5),Not(x_5),)),(Or((x_5),Not(x_4),)),)
nn_out4 = $$4
nn_out5 = $$5
nn_out6 = $$6
nn_out7 = $$7
nn_out8 = $$8

carry1 = (And((x_0),(x_5),))
c6 = (Or((x_5),(i_11),))
c5 = (And((x_4),(i_10),))
c4 = (Or((x_0),(i_12),))
c3 = Not(i_9)
carry2 = (Or(((And((x_4),(x_6),))),((And((carry1),((Xor((x_4),(x_6),))),))),))
c2 = (Xor(((Xor((carry1),(x_4),))),(x_6),))
c1 = (Xor((x_0),(x_5),))
a7 = Not((Xor((i_12),(c6),)))
a6 = Not((Xor((i_11),(c5),)))
a5 = Not((Xor((i_10),(c4),)))
a4 = Not((Xor((c3),(i_1),)))
a3 = Not((Xor((carry2),(i_3),)))
a2 = Not((Xor((c2),(i_8),)))
a1 = Not((Xor((c1),(i_7),)))
out = (And(((And(((And(((And(((And(((And((a1),(a2),))),(a3),))),(a4),))),(a5),))),(a6),))),(a7),))
z1 = Exists(i_1, out)
i_1 = nn_out0
i_2 = nn_out1
i_3 = nn_out2
i_7 = nn_out3
i_8 = nn_out4
i_9 = nn_out5
i_10 = nn_out6
i_11 = nn_out7
i_12 = nn_out8

carry1 = (And((x_0),(x_5),))
c6 = (Or((x_5),(i_11),))
c5 = (And((x_4),(i_10),))
c4 = (Or((x_0),(i_12),))
c3 = Not(i_9)
carry2 = (Or(((And((x_4),(x_6),))),((And((carry1),((Xor((x_4),(x_6),))),))),))
c2 = (Xor(((Xor((carry1),(x_4),))),(x_6),))
c1 = (Xor((x_0),(x_5),))
a7 = Not((Xor((i_12),(c6),)))
a6 = Not((Xor((i_11),(c5),)))
a5 = Not((Xor((i_10),(c4),)))
a4 = Not((Xor((c3),(i_1),)))
a3 = Not((Xor((carry2),(i_3),)))
a2 = Not((Xor((c2),(i_8),)))
a1 = Not((Xor((c1),(i_7),)))
out = (And(((And(((And(((And(((And(((And((a1),(a2),))),(a3),))),(a4),))),(a5),))),(a6),))),(a7),))
z2 = out
formula = z1==z2
valid(formula)