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
	i0 ,i1 ,i2 ,i3 ,i4 ,i5 ,i6 = Bools('i0 i1 i2 i3 i4 i5 i6')
	n9,n10,n11,n12,n13,n14,n15,n16,n17,n18,n19,n20,n21,n22,n23,n24,n25 = Bools('n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25')
	
	i7 = Bools('i7')
	i7 = ()
	
	n9 = And(Not(i0),(i1),)
	n10 = And((i0),Not(i1),)
	n11 = And(Not(n9),Not(n10),)
	n12 = And((i2),(n11),)
	n13 = And(Not(i2),Not(n11),)
	n14 = And(Not(n12),Not(n13),)
	n15 = And((i3),(n14),)
	n16 = And(Not(i3),Not(n14),)
	n17 = And(Not(n15),Not(n16),)
	n18 = And((i4),(n17),)
	n19 = And(Not(i4),Not(n17),)
	n20 = And(Not(n18),Not(n19),)
	n21 = And((i5),(n20),)
	n22 = And(Not(i5),Not(n20),)
	n23 = And(Not(n21),Not(n22),)
	n24 = And((i6),(n23),)
	n25 = And(Not(i6),Not(n23),)
	i7 = And(Not(n24),Not(n25),)
	outs = [i7]
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
