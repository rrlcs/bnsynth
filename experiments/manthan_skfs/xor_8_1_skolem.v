// Benchmark "SKOLEMFORMULA" written by ABC on Sat May 21 11:51:13 2022

module SKOLEMFORMULA ( 
    i0, i1, i2, i3, i4, i5, i6, i7,
    i8  );
  input  i0, i1, i2, i3, i4, i5, i6, i7;
  output i8;
  wire n10, n11, n12, n13, n14, n15, n16, n17, n18, n19, n20, n21, n22, n23,
    n24, n25, n26, n27, n28, n29;
  assign n10 = ~i0 & i1;
  assign n11 = i0 & ~i1;
  assign n12 = ~n10 & ~n11;
  assign n13 = i2 & n12;
  assign n14 = ~i2 & ~n12;
  assign n15 = ~n13 & ~n14;
  assign n16 = i3 & n15;
  assign n17 = ~i3 & ~n15;
  assign n18 = ~n16 & ~n17;
  assign n19 = i4 & n18;
  assign n20 = ~i4 & ~n18;
  assign n21 = ~n19 & ~n20;
  assign n22 = i5 & n21;
  assign n23 = ~i5 & ~n21;
  assign n24 = ~n22 & ~n23;
  assign n25 = i6 & n24;
  assign n26 = ~i6 & ~n24;
  assign n27 = ~n25 & ~n26;
  assign n28 = i7 & n27;
  assign n29 = ~i7 & ~n27;
  assign i8 = ~n28 & ~n29;
endmodule


