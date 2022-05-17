from z3 import *

set_option(max_depth=1000000000, rational_to_decimal=True,
           precision=100000, max_lines=100000000)
tactic_simplify = Repeat(Then(Repeat('ctx-solver-simplify'),
                              Tactic('simplify')))
nnf = Tactic('nnf')
# nnf = Then(Tactic('nnf'), Tactic('simplify'))
dnf = Repeat(OrElse(Tactic('split-clause'),
                    Tactic('skip')))
cnf = Tactic('tseitin-cnf')

def getz3formula():
	i0 ,i1 ,i2 ,i3 ,i4 ,i5 = Bools('i0 i1 i2 i3 i4 i5')
	
	out = Bools('out')
	i2 = Or((And((i0),(i1),Not(i0),Not(i1),)),(And((i0),(i1),Not(i1),)),)
	i3 = Or((And((i0),(i1),Not(i0),)),(And((i0),Not(i0),)),)
	i4 = Or((And((i0),Not(i0),Not(i1),)),(And((i1),Not(i1),)),)
	i5 = Or((And((i0),(i1),)),(And(Not(i0),Not(i1),)),)
	
	out = (Xor((Xor((Xor((Xor((Xor((i0),(i1),)),(i2),)),(i3),)),(i4),)),(i5),))
	outs = [i2, i3, i4, i5]
	return outs


outs = getz3formula()
char_count_pre_simplification = 0
char_count_post_simplification = 0
clause_count_pre_simplification = 0
clause_count_post_simplification = 0
literals_pre_simplification = 0
literals_post_simplification = 0
ftext = ''
for formula in outs:
    fstr = str(formula).replace("\n", "").replace(" ", "")
    char_count_pre_simplification += len(fstr)
    if formula == True or formula == False:
        childs = [1]
    else:
        childs = formula.children()
    if len(childs) != 0:
        clause_count_pre_simplification += len(childs)
    else:
        clause_count_pre_simplification += 1
    # for f in childs:
    #     ch = f.children()
    #     # print("literals: ", ch)
    #     literals_pre_simplification += len(ch)

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
print(ftext)
print(clause_count_pre_simplification)
print(clause_count_post_simplification)
print(char_count_pre_simplification)
print(char_count_post_simplification)
# print("literals pre simplification: ", literals_pre_simplification)
