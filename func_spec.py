

def F(XY_vars, util):
    i_0 = XY_vars[0, :]
    i_1 = XY_vars[1, :]
    i_2 = XY_vars[2, :]
    i_3 = XY_vars[3, :]
    w1 = util.tnorm_vectorized(i_0, i_1)
    w2 = util.continuous_xor(i_2, i_3)
    out = util.tconorm_vectorized(w1, w2)
    return out
