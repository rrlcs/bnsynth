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
	i0 ,i1 ,i2 ,i3 ,i4 ,i5 ,i6 ,i7 = Bools('i0 i1 i2 i3 i4 i5 i6 i7')
	n12,n13,n14,n15,n16,n17,n18,n19,n20,n21,n22,n23,n24,n25,n26,n27,n28,n29,n30,n31 = Bools('n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31')
	
	i8 ,i9 = Bools('i8 i9')
	i8 = ()
	i9 = ()
	
	n12 = And(Not(i0),(i1),)
	n13 = And((i0),Not(i1),)
	n14 = And(Not(n12),Not(n13),)
	n15 = And((i2),Not(n14),)
	n16 = And(Not(i2),(n14),)
	n17 = And(Not(n15),Not(n16),)
	n18 = And((i3),Not(n17),)
	n19 = And(Not(i3),(n17),)
	n20 = And(Not(n18),Not(n19),)
	n21 = And((i4),Not(n20),)
	n22 = And(Not(i4),(n20),)
	n23 = And(Not(n21),Not(n22),)
	n24 = And((i5),Not(n23),)
	n25 = And(Not(i5),(n23),)
	n26 = And(Not(n24),Not(n25),)
	n27 = And((i6),Not(n26),)
	n28 = And(Not(i6),(n26),)
	n29 = And(Not(n27),Not(n28),)
	n30 = And((i7),Not(n29),)
	n31 = And(Not(i7),(n29),)
	i8 = Or((n30),(n31),)
	i9 = (True)
	outs = [i8, i9]
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
    ftext += text
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
print("Counts: ")
print(ftext)
print(clause_count_pre_simplification)
print(clause_count_post_simplification)
print(char_count_pre_simplification)
print(char_count_post_simplification)
print(literals_pre_simplification)
print(literals_post_simplification)
