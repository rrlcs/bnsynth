from z3 import *

set_option(max_depth=1000000000, rational_to_decimal=True,
           precision=100000, max_lines=100000000)
tactic_simplify = Then(Repeat('ctx-solver-simplify'),
                       (Tactic('ctx-solver-simplify')))

def getz3formula():
	i0 ,i1 ,i2 ,i3 ,i4 ,i5 ,i6 = Bools('i0 i1 i2 i3 i4 i5 i6')
	N10,N11,N16,N19,N221,N222,N231,N232 = Bools('N10 N11 N16 N19 N221 N222 N231 N232')
	
	out = Bool('out')
	i5 = (Or((And((i2),(i3),Not(i0),)),(And(Not(i0),Not(i1),)),(And(Not(i1),Not(i2),)),))
	i6 = (((i3)))
	
	N10 = Not(And((i0),(i2),))
	N11 = Not(And((i2),(i3),))
	N16 = Not(And((i1),(N11),))
	N19 = Not(And((N11),(i4),))
	N221 = Not(And((N10),(N16),))
	N231 = Not(And((N16),(N19),))
	N222 = Not(And((N10),(i5),))
	N232 = Not(And((i5),(i6),))
	out = Not(Xor((N221),(N222),))
	outs = [i5, i6]
	return outs


outs = getz3formula()
total_chars = 0
clause = 0
cl = 0
count = 0
ftext = ''
for v in outs:
    formula = v
    fstr = str(formula)
    cl += fstr.count("And")
    if cl == 0:
        cl = 1
    g = Goal()
    g.add(formula)
    wp = tactic_simplify(g).as_expr()
    text = str(wp).replace("\n", "").replace(" ", "")
    ftext += text

    total_chars += len(text)
    count = text.count("And")
    if count <= 1:
        clause += 1
    else:
        clause += count

f = open('simplified.skf', 'w')
f.write(ftext)
f.close()
print("Number of Chars: ", total_chars, "Number of Clauses: ", cl, clause)
