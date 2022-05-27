// Benchmark "SKOLEMFORMULA" written by ABC on Fri May 27 20:55:12 2022

module SKOLEMFORMULA ( 
    i0, i1, i2, i3, i4,
    i5, i6  );
  input  i0, i1, i2, i3, i4;
  output i5, i6;
  wire n8, n9, n10, n11, n12, n13, n14, n15, n16, n17, n18, n20, n21, n22,
    n23, n24, n25, n26, n27, n28, n29;
  assign n8 = ~i1 & ~i3;
  assign n9 = ~i4 & n8;
  assign n10 = i4 & n8;
  assign n11 = ~n9 & ~n10;
  assign n12 = i1 & ~i3;
  assign n13 = i0 & n12;
  assign n14 = ~i4 & n13;
  assign n15 = i2 & n14;
  assign n16 = n11 & ~n15;
  assign n17 = i4 & n13;
  assign n18 = n16 & ~n17;
  assign i6 = i3 | ~n18;
  assign n20 = ~i4 & i6;
  assign n21 = ~i1 & n20;
  assign n22 = i6 & ~n21;
  assign n23 = i1 & n20;
  assign n24 = i2 & n23;
  assign n25 = i3 & n24;
  assign n26 = n22 & ~n25;
  assign n27 = i4 & i6;
  assign n28 = i2 & n27;
  assign n29 = i3 & n28;
  assign i5 = ~n26 | n29;
endmodule


