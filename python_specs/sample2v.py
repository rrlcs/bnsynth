def F(XY_vars, util):
	i_0 = XY_vars[0, :]
	i_1 = XY_vars[1, :]
	i_2 = XY_vars[2, :]
	w2 = util.tnorm_vectorized((i_1),(i_2))
	w1 = util.continuous_xor((i_0),(i_1))
	out = util.continuous_xor((w1),(w2))
	return out