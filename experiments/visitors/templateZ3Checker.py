from z3 import *

set_option(max_depth=1000000000, rational_to_decimal=True,
           precision=100000, max_lines=100000000)
tactic_simplify = Then(Repeat('ctx-solver-simplify'),
                       (Tactic('ctx-solver-simplify')))

#*#

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
