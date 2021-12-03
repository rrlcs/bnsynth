def F(XY_vars, util):
	i_0 = XY_vars[0, :]
	i_1 = XY_vars[1, :]
	out = util.tnorm_vectorized(((util.tconorm_vectorized(util.negation(i_0),(i_1)))),((util.tconorm_vectorized(util.negation(i_1),(i_0)))))
	return out