// Benchmark "SKOLEMFORMULA" written by ABC on Fri May 20 22:23:25 2022

module SKOLEMFORMULA ( 
    i0, i1, i2, i3, i4, i5,
    i6  );
  input  i0, i1, i2, i3, i4, i5;
  output i6;
  wire n8, n9, n10, n11, n12, n13, n14, n15, n16, n17, n18, n19, n20, n21;
  assign n8 = ~i0 & i1;
  assign n9 = i0 & ~i1;
  assign n10 = ~n8 & ~n9;
  assign n11 = i2 & n10;
  assign n12 = ~i2 & ~n10;
  assign n13 = ~n11 & ~n12;
  assign n14 = i3 & n13;
  assign n15 = ~i3 & ~n13;
  assign n16 = ~n14 & ~n15;
  assign n17 = i4 & n16;
  assign n18 = ~i4 & ~n16;
  assign n19 = ~n17 & ~n18;
  assign n20 = i5 & n19;
  assign n21 = ~i5 & ~n19;
  assign i6 = ~n20 & ~n21;
endmodule


