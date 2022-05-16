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
	i0 ,i1 ,i2 ,i3 ,i4 ,i5 ,i6 ,i7 ,i8 ,i9 = Bools('i0 i1 i2 i3 i4 i5 i6 i7 i8 i9')
	
	i10 ,i11 ,i12 ,i13 ,i14 ,i15 ,i16 ,i17 ,i18 ,i19 = Bools('i10 i11 i12 i13 i14 i15 i16 i17 i18 i19')
	i10 = ()
	i11 = ()
	i12 = ()
	i13 = ()
	i14 = ()
	i15 = ()
	i16 = ()
	i17 = ()
	i18 = ()
	i19 = ()
	
	i10 = (i0)
	i11 = (i1)
	i12 = (i2)
	i13 = (i3)
	i14 = (i4)
	i15 = (i5)
	i16 = (i6)
	i17 = (i7)
	i18 = (i8)
	i19 = (i9)
	outs = [i10, i11, i12, i13, i14, i15, i16, i17, i18, i19]
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
