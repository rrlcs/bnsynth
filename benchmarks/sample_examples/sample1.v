module formula(i_10,i_1,i_22,o_1);
	input i_10 ,i_1 ,i_22;
	wire w1;
	output o_1;
	assign w1 = (i_10 ^ i_1);
	assign o_1 = (w1 ^ i_22);
endmodule
