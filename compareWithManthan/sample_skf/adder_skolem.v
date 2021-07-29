module SKOLEMFORMULA (i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12,  out );
input i0;
input i1;
input i2;
input i3;
input i4;
input i5;
input i6;
input i7;
input i8;
input i9;
input i10;
input i11;
input i12;
output out ;
wire zero;
wire one;
assign zero = 0;
assign one = 1;
wire wi1;
wire wi2;
wire wi3;
wire wi7;
wire wi8;
wire wi9;
wire wi10;
wire wi11;
wire wi12;
wire wt10;
assign wi1 = (( ~i9 & one ));
assign wi2 = ( one );
assign wi3 = (( ~i11 & i6 & ~i8 & one ) | ( i11 & ~i6 & ~i7 & one ) | ( i11 & i6 & one ));
assign wi7 = (( i10 & ~i12 & one ) | ( i10 & i12 & ~i0 & i5 & one ) | ( i10 & i12 & i0 & ~i5 & one ));
assign wi8 = (( ~i10 & i1 & one ) | ( i10 & ~i9 & ~i4 & ~i5 & i6 & one ) | ( i10 & ~i9 & ~i4 & i5 & one ) | ( i10 & i9 & ~i11 & i6 & ~i0 & one ) | ( i10 & i9 & i11 & ~i6 & one ) | ( i10 & i9 & i11 & i6 & ~i7 & one ));
assign wi9 = (( ~i12 & ~i0 & ~i6 & ~i4 & one ) | ( i12 & ~i7 & i6 & i5 & one ) | ( i12 & i7 & ~i6 & one ) | ( i12 & i7 & i6 & ~i0 & ~i4 & one ) | ( i12 & i7 & i6 & i0 & one ));
assign wi10 = (( ~i12 & i0 & one ) | ( i12 & one ));
assign wi11 = (( i4 & i10 & one ));
assign wi12 = (( ~i5 & i4 & ~i6 & i0 & one ) | ( ~i5 & i4 & i6 & one ) | ( i5 & one ));
assign wt10 = (~(wi1 ^ i1)) & (~(wi2 ^ i2)) & (~(wi3 ^ i3)) & (~(wi7 ^ i7)) & (~(wi8 ^ i8)) & (~(wi9 ^ i9)) & (~(wi10 ^ i10)) & (~(wi11 ^ i11)) & (~(wi12 ^ i12));
assign out = wt10;
endmodule