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
	i0 ,i1 = Bools('i0 i1')
	
	i2 ,i3 = Bools('i2 i3')
	i2 = ()
	i3 = ()
	
	i3 = (True)
	i2 = (i0)
	outs = [i2, i3]
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
