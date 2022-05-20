// Benchmark "SKOLEMFORMULA" written by ABC on Fri May 20 21:19:49 2022

module SKOLEMFORMULA ( 
    i0, i1, i2,
    i3, i4, i5  );
  input  i0, i1, i2;
  output i3, i4, i5;
  wire n7, n9, n10, n11, n12, n13, n15, n16, n17, n18, n19, n20, n21, n22,
    n23, n24, n25;
  assign n7 = i1 & i2;
  assign i4 = ~i2 | n7;
  assign n9 = i0 & ~i4;
  assign n10 = i0 & ~n9;
  assign n11 = i0 & i4;
  assign n12 = ~i2 & n11;
  assign n13 = i1 & n12;
  assign i3 = ~n10 | n13;
  assign n15 = ~i4 & i3;
  assign n16 = ~i0 & n15;
  assign n17 = i3 & ~n16;
  assign n18 = i4 & i3;
  assign n19 = ~i1 & n18;
  assign n20 = n17 & ~n19;
  assign n21 = i1 & n18;
  assign n22 = ~i2 & n21;
  assign n23 = i0 & n22;
  assign n24 = n20 & ~n23;
  assign n25 = i2 & n21;
  assign i5 = ~n24 | n25;
endmodule


