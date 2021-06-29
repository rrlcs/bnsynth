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
wire wt2;
assign wi1 = (( one ));
assign wt2 = (~(wi1 ^ i1));
assign out = wt2;
endmodule