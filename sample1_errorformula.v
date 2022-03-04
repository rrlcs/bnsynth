module FORMULA(i_0,i_1,i_2,o_1);
	input i_0 ,i_1 ,i_2;
	wire w1;
	wire w2, w3, w4;
	output o_1;
	assign w1 = (i_0 ^ i_1);
	assign o_1 = (w1 ^ i_2);
	assign o_1 = o_1 & (i_0 = 0);
endmodule