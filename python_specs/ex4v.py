def F(XY_vars, util):
	i_0 = XY_vars[0, :]
	i_1 = XY_vars[1, :]
	out = util.tconorm_vectorized((i_0),(i_1))
	return out