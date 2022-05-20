// Benchmark "SKOLEMFORMULA" written by ABC on Fri May 20 20:24:08 2022

module SKOLEMFORMULA ( 
    i0, i1, i2, i3, i4,
    i5  );
  input  i0, i1, i2, i3, i4;
  output i5;
  wire n7, n8, n9, n10, n11, n12, n13, n14, n15, n16, n17;
  assign n7 = ~i0 & i1;
  assign n8 = i0 & ~i1;
  assign n9 = ~n7 & ~n8;
  assign n10 = ~i2 & i3;
  assign n11 = i2 & ~i3;
  assign n12 = ~n10 & ~n11;
  assign n13 = n9 & ~n12;
  assign n14 = ~n9 & n12;
  assign n15 = ~n13 & ~n14;
  assign n16 = ~i4 & n15;
  assign n17 = i4 & ~n15;
  assign i5 = n16 | n17;
endmodule


