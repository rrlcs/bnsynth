def F(XY_vars, util):
	i_0 = XY_vars[0, :]
	i_1 = XY_vars[1, :]
	i_2 = XY_vars[2, :]
	w1 = (i_0 ^ i_1)
	out = (w1 ^ i_2)
	return out