// Benchmark "SKOLEMFORMULA" written by ABC on Fri May 27 20:52:53 2022

module SKOLEMFORMULA ( 
    i0, i1, i2, i3, i4,
    i5, i6  );
  input  i0, i1, i2, i3, i4;
  output i5, i6;
  wire n9, n10, n11, n12, n13, n14, n15, n16, n17;
  assign n9 = ~i3 & ~i4;
  assign n10 = i1 & n9;
  assign n11 = ~i3 & i4;
  assign n12 = ~n10 & ~n11;
  assign n13 = ~i2 & i3;
  assign n14 = ~i4 & n13;
  assign n15 = i1 & n14;
  assign n16 = n12 & ~n15;
  assign n17 = i4 & n13;
  assign i6 = ~n16 | n17;
  assign i5 = 1'b1;
endmodule


