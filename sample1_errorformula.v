module MAIN (i0, i1, i2, ip1, ip2, out );
input i0 ;
input i1 ;
input i2 ;
input ip1 ;
input ip2 ;
output out;
wire out1;
wire out2;
wire out3;
FORMULA F1 (i0, i1, i2, out1 );
SKOLEMFORMULA F2 (i0, ip1, ip2, out2 );
FORMULA F2 (i0, ip1, ip2, out3 );
assign out = ( out1 & out2 & ~(out3) );
endmodule
module FORMULA(i_0,i_1,i_2,o_1);
	input i_0 ,i_1 ,i_2;
	wire w1;
	output o_1;
	assign w1 = (i_0 ^ i_1);
	assign o_1 = (w1 ^ i_2);
endmodule
module SKOLEMFORMULA (i0, i1, i2,  out );
input i0;
input i1;
input i2;
output out ;
wire zero;
wire one;
assign zero = 0;
assign one = 1;
wire wi1;
wire wi2;
wire wt3;
assign wi1 = ((i0 | ~i0) & (i0) & (~i0));
assign wi2 = ((i0 | ~i0) & (~i0));
assign wt3 = (~(wi1 ^ i1)) & (~(wi2 ^ i2));
assign out = wt3;
endmodule