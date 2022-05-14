from z3 import *

set_option(max_depth=1000000000, rational_to_decimal=True,
           precision=100000, max_lines=100000000)
tactic_simplify = Repeat(Then(Repeat('ctx-solver-simplify'),
                              Then(Tactic('nnf'), Tactic('simplify'))))

def getz3formula():
	i0 ,i1 ,i2 = Bools('i0 i1 i2')
	w1 = Bool('w1')
	
	o1 = Bool('o1')
	i1 = ((And((i0),Not(i0),)))
	i2 = ((Not(i0)))
	
	w1 = (Xor((i0),(i1),))
	o1 = (Xor((w1),(i2),))
	outs = [i1, i2]
	return outs


outs = getz3formula()
char_count_pre_simplification = 0
char_count_post_simplification = 0
clause_count_pre_simplification = 0
clause_count_post_simplification = 0
ftext = ''
for formula in outs:
    fstr = str(formula)
    char_count_pre_simplification += len(fstr)
    childs = formula.children()
    clause_count_pre_simplification += len(childs)

    # Simplify formula using Tactics
    g = Goal()
    g.add(formula)
    wp = tactic_simplify(g).as_expr()
    text = str(wp).replace("\n", "").replace(" ", "")
    ftext += text
    char_count_post_simplification += len(text)
    childs = wp.children()
    clause_count_post_simplification += len(childs)

f = open('experiments/simplified.skf', 'w')
f.write(ftext)
f.close()
print("Counts: ")
print(clause_count_pre_simplification)
print(clause_count_post_simplification)
print(char_count_pre_simplification)
print(char_count_post_simplification)
