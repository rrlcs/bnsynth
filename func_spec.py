def F(inp_vars, name, util):
	i_0  = inp_vars[0, :]
	i_1  = inp_vars[1, :]
	i_2 = inp_vars[2, :]
	w1 = util.continuous_xor((i_0),(i_1),name)
	out = util.continuous_xor((w1),(i_2),name)
	return out