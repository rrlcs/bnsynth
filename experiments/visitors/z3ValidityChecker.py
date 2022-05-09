from z3 import *

set_option(max_depth=1000000000, rational_to_decimal=True,
           precision=100000, max_lines=100000000)
tactic_simplify = Then(Repeat('ctx-solver-simplify'),
                       (Tactic('ctx-solver-simplify')))

def getz3formula():
	i0 ,i1 ,i2 = Bools('i0 i1 i2')
	w1 = Bool('w1')
	
	o1 = Bool('o1')
	i1 = ((And((i0),Not(i0),)))
	i2 = (Or((And((i0),Not(i0),)),(Not(i0)),))
	
	w1 = (Xor((i0),(i1),))
	o1 = (Xor((w1),(i2),))
	outs = [i1, i2]
	return outs


outs = getz3formula()
total_chars = 0
clause = 0
cl = 0
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

    total_chars += len(text)
    clause += text.count("And")


print("Number of Chars: ", total_chars, "Number of Clauses: ", cl, clause)
