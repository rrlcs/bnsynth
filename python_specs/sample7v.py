def F(XY_vars, util):
	x_0 = XY_vars[0, :]
	i_1 = XY_vars[1, :]
	i_2 = XY_vars[2, :]
	i_3 = XY_vars[3, :]
	x_4 = XY_vars[4, :]
	x_5 = XY_vars[5, :]
	x_6 = XY_vars[6, :]
	i_7 = XY_vars[7, :]
	i_8 = XY_vars[8, :]
	i_9 = XY_vars[9, :]
	i_10 = XY_vars[10, :]
	i_11 = XY_vars[11, :]
	i_12 = XY_vars[12, :]
	c1 = util.continuous_xor((x_0),(x_5))
	carry1 = util.tnorm_vectorized((x_0),(x_5))
	c3 = util.negation(i_9)
	c5 = util.tnorm_vectorized((x_4),(i_10))
	c6 = util.tconorm_vectorized((x_5),(i_11))
	c4 = util.tconorm_vectorized((x_0),(i_12))
	a1 = util.negation((util.continuous_xor((c1),(i_7))))
	c2 = util.continuous_xor(((util.continuous_xor((carry1),(x_4)))),(x_6))
	carry2 = util.tconorm_vectorized(((util.tnorm_vectorized((x_4),(x_6)))),((util.tnorm_vectorized((carry1),((util.continuous_xor((x_4),(x_6))))))))
	a4 = util.negation((util.continuous_xor((c3),(i_1))))
	a6 = util.negation((util.continuous_xor((i_11),(c5))))
	a7 = util.negation((util.continuous_xor((i_12),(c6))))
	a5 = util.negation((util.continuous_xor((i_10),(c4))))
	a2 = util.negation((util.continuous_xor((c2),(i_8))))
	a3 = util.negation((util.continuous_xor((carry2),(i_3))))
	out = util.tnorm_vectorized(((util.tnorm_vectorized(((util.tnorm_vectorized(((util.tnorm_vectorized(((util.tnorm_vectorized(((util.tnorm_vectorized((a1),(a2)))),(a3)))),(a4)))),(a5)))),(a6)))),(a7))
	return out