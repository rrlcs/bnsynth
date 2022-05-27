// Benchmark "SKOLEMFORMULA" written by ABC on Fri May 27 20:52:59 2022

module SKOLEMFORMULA ( 
    i0, i1, i2, i3, i4,
    i5, i6  );
  input  i0, i1, i2, i3, i4;
  output i5, i6;
  wire n9, n10, n11, n12, n13, n14, n15, n16, n17;
  assign n9 = ~i1 & i2;
  assign n10 = i0 & n9;
  assign n11 = i1 & ~i3;
  assign n12 = ~n10 & ~n11;
  assign n13 = i1 & i3;
  assign n14 = ~i2 & n13;
  assign n15 = n12 & ~n14;
  assign n16 = i2 & n13;
  assign n17 = i0 & n16;
  assign i5 = ~n15 | n17;
  assign i6 = 1'b1;
endmodule


