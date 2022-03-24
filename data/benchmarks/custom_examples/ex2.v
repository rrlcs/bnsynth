module formula(i_0,i_1, i_2,out);
	input i_0 ,i_1, i_2;

	output out;
	assign out = i_1 & ~i_2;
endmodule
