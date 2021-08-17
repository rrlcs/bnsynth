module formula(i_0,i_1,i_2,out);
	input i_0 ,i_1 ,i_2;
	wire w1;
	output out;
	assign out = (w1 ^ i_2) ;
	assign w1 = (i_0 ^ i_1) ;
endmodule
