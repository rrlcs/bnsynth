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
	i0 ,i1 ,i2 ,i3 ,i4 = Bools('i0 i1 i2 i3 i4')
	n9,n10,n11,n12,n13,n14,n15,n16,n17 = Bools('n9 n10 n11 n12 n13 n14 n15 n16 n17')
	
	i5 ,i6 = Bools('i5 i6')
	i5 = ()
	i6 = ()
	
	n9 = And(Not(i1),(i2),)
	n10 = And((i0),(n9),)
	n11 = And((i1),Not(i3),)
	n12 = And(Not(n10),Not(n11),)
	n13 = And((i1),(i3),)
	n14 = And(Not(i2),(n13),)
	n15 = And((n12),Not(n14),)
	n16 = And((i2),(n13),)
	n17 = And((i0),(n16),)
	i5 = Or(Not(n15),(n17),)
	i6 = (True)
	outs = [i5, i6]
	return outs



def get_child_count(formula):
    if formula == True:
        childs = [Bool(True)]
    elif formula == False:
        childs = [Bool(False)]
    else:
        childs = formula.children()

    if len(childs) != 0:
        return len(childs), childs
    else:
        return 1, childs


f = open("cnf", "r")
is_cnf = (f.read()) == 'cnf'

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
    g = Goal()
    g.add(formula)
    if is_cnf:
        formula = cnf(g).as_expr()
    cl, childs = get_child_count(formula)
    clause_count_pre_simplification += cl
    if len(childs) > 0:
        for f in childs:
            cl, ch = get_child_count(f)
            literals_pre_simplification += cl
    else:
        literals_pre_simplification += 1

    # Simplify formula using Tactics
    g = Goal()
    g.add(formula)
    wp = tactic_simplify(g).as_expr()
    text = str(wp).replace("\n", "").replace(" ", "")
    ftext += (text+"\n").replace(",", ", ")
    char_count_post_simplification += len(text)
    cl, childs = get_child_count(wp)
    clause_count_post_simplification += cl
    if len(childs) > 0:
        for f in childs:
            cl, ch = get_child_count(f)
            literals_post_simplification += cl
    else:
        literals_post_simplification += 1

f = open('experiments/simplified.skf', 'w')
f.write(ftext)
f.close()
ftext = ftext.replace(" ", "").replace("\n", "")
print("Counts: ")
print(ftext)
print(clause_count_pre_simplification)
print(clause_count_post_simplification)
print(char_count_pre_simplification)
print(char_count_post_simplification)
print(literals_pre_simplification)
print(literals_post_simplification)
