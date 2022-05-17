module formula(i_0,i_1,i_2,i_3, i_4, i_5, i_6, i_7, i_8, i_9, i_10,i_11,i_12,i_13, i_14, i_15, i_16, i_17, i_18, i_19, out);
	input i_0, i_1,i_2,i_3, i_4, i_5, i_6, i_7, i_8, i_9, i_10,i_11,i_12,i_13, i_14, i_15, i_16, i_17, i_18, i_19;
	output out;
	wire w1, w2, w3, w4, w5, w6, w7, w8, w9, w10;
	assign w1 = (~i_0 | i_10) & (i_0 | ~i_10);
	assign w2 = (~i_1 | i_11) & (i_1 | ~i_11);
	assign w3 = (~i_2 | i_12) & (i_2 | ~i_12);
	assign w4 = (~i_3 | i_13) & (i_3 | ~i_13);
	assign w5 = (~i_4 | i_14) & (i_4 | ~i_14);
	assign w6 = (~i_5 | i_15) & (i_5 | ~i_15);
	assign w7 = (~i_6 | i_16) & (i_6 | ~i_16);
	assign w8 = (~i_7 | i_17) & (i_7 | ~i_17);
	assign w9 = (~i_8 | i_18) & (i_8 | ~i_18);
	assign w10 = (~i_9 | i_19) & (i_9 | ~i_19);
	assign out = (((((((((w1 & w2) & w3) & w4) & w5) & w6) & w7) & w8) & w9) & w10);
endmodule
