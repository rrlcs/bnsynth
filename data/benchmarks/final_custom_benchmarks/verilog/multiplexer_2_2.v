module formula(i_0,i_1,i_2,i_3, out);
	input i_0, i_1,i_2,i_3;
	output out;
	assign out = ((i_2 & ~i_3) | (~i_2 & i_3)) & (~i_2 | i_0) & (~i_3 | i_1);
endmodule
