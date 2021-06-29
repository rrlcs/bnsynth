module SKOLEMFORMULA (i0, i1, i2, i3, i4,  out );
input i0;
input i1;
input i2;
input i3;
input i4;
output out ;
wire zero;
wire one;
assign zero = 0;
assign one = 1;
wire wi4;
wire wt2;
assign wi4 = (( ~i3 & ~i0 & ~i1 & one ) | ( i3 & ~i2 & ~i0 & ~i1 & one ) | ( i3 & i2 & ~i1 & i0 & one ) | ( i3 & i2 & i1 & one ));
assign wt2 = (~(wi4 ^ i4));
assign out = wt2;
endmodule