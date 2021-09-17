def F(XY_vars, util):
	i_0 = XY_vars[0, :]
	i_1 = XY_vars[1, :]
	i_2 = XY_vars[2, :]
	i_3 = XY_vars[3, :]
	i_4 = XY_vars[4, :]
	w2 = util.tnorm_vectorized((i_2),(i_3))
	w1 = util.tconorm_vectorized((i_0),(i_1))
	w3 = util.tconorm_vectorized((w1),(w2))
	out = util.continuous_xor((w3),(i_4))
	return out