// Benchmark "SKOLEMFORMULA" written by ABC on Fri May 20 19:55:04 2022

module SKOLEMFORMULA ( 
    i0, i1, i2, i3, i4, i5, i6,
    i7  );
  input  i0, i1, i2, i3, i4, i5, i6;
  output i7;
  wire n9, n10, n11, n12, n13, n14, n15, n16, n17, n18, n19, n20, n21, n22,
    n23, n24, n25;
  assign n9 = ~i0 & i1;
  assign n10 = i0 & ~i1;
  assign n11 = ~n9 & ~n10;
  assign n12 = i2 & n11;
  assign n13 = ~i2 & ~n11;
  assign n14 = ~n12 & ~n13;
  assign n15 = i3 & n14;
  assign n16 = ~i3 & ~n14;
  assign n17 = ~n15 & ~n16;
  assign n18 = i4 & n17;
  assign n19 = ~i4 & ~n17;
  assign n20 = ~n18 & ~n19;
  assign n21 = i5 & n20;
  assign n22 = ~i5 & ~n20;
  assign n23 = ~n21 & ~n22;
  assign n24 = i6 & n23;
  assign n25 = ~i6 & ~n23;
  assign i7 = n24 | n25;
endmodule


