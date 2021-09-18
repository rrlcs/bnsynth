module formula(i_0,i_1,i_2,out);
	input i_0 ,i_1 ,i_2;
	wire w1;
	output out;
	assign w1 = (i_0 ^ i_1);
	assign out = (w1 ^ i_2);
endmodule
