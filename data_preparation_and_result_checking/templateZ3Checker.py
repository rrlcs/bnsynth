from z3 import *

#Function to check validity of a formula

def valid(formula):

    s = Solver()
    s1 = Solver()
    s.add(Not(formula))
    # s1.add(formula)
    # if s1.check() == unsat:
    #     print("============= model =============", s1.model())
    if s.check() == unsat:

        # print("counter examples: ", s.model())
        print("Valid")

        return True, {}

    else:
        m = s.model()
        
        model_dict = {}
        for d in m.decls():
            for i in range(1):
                # print(s.check())
                if s.check() == sat:
                    m = s.model()
                    # print(m)
                s.add(Bool(d.name()) != m[d])
                if i == 0:
                    model_dict[d.name()] = [0] if str(m[d])=="False" else [1]
                else:
                    model_dict[d.name()].append(0) if str(m[d])=="False" else model_dict[d.name()].append(1)
            # print("%s = %s" % (d.name(), str(m[d])))
        print("model_dict: ", model_dict)
        if m[0]() == False:
            print("false")
        else:
            print("true")
        print("counter examples: ", m)

        print("Not Valid")

        # for v, k in enumerate(model_dict):
            # print("key value pair: ", type(k), v)
            # vars()[k] = Bool((vars()[k]))
            # globals()[k] = 1 #Bool(str(globals()[k]))
        # i_0 = Bool(str(globals()[k]))
        # print("******************************", i_0, i_1)
        return False, model_dict

#*#