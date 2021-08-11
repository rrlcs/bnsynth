def F(inp_vars, util):
	i_0 = inp_vars[0, :]
	i_1 = inp_vars[1, :]
	i_2 = inp_vars[2, :]
	w1 = util.continuous_xor((i_0),(i_1))
	out = util.continuous_xor((w1),(i_2))
	return out