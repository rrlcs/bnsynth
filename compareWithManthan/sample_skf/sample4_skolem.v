module SKOLEMFORMULA (i0, i1, i2, i3,  out );
input i0;
input i1;
input i2;
input i3;
output out ;
wire zero;
wire one;
assign zero = 0;
assign one = 1;
wire wi3;
wire wt2;
assign wi3 = (( one ));
assign wt2 = (~(wi3 ^ i3));
assign out = wt2;
endmodule