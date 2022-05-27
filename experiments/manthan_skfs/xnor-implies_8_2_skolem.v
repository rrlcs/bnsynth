// Benchmark "SKOLEMFORMULA" written by ABC on Tue May 24 17:40:07 2022

module SKOLEMFORMULA ( 
    i0, i1, i2, i3, i4, i5, i6, i7,
    i8, i9  );
  input  i0, i1, i2, i3, i4, i5, i6, i7;
  output i8, i9;
  wire n12, n13, n14, n15, n16, n17, n18, n19, n20, n21, n22, n23, n24, n25,
    n26, n27, n28, n29, n30, n31;
  assign n12 = ~i0 & i1;
  assign n13 = i0 & ~i1;
  assign n14 = ~n12 & ~n13;
  assign n15 = i2 & ~n14;
  assign n16 = ~i2 & n14;
  assign n17 = ~n15 & ~n16;
  assign n18 = i3 & ~n17;
  assign n19 = ~i3 & n17;
  assign n20 = ~n18 & ~n19;
  assign n21 = i4 & ~n20;
  assign n22 = ~i4 & n20;
  assign n23 = ~n21 & ~n22;
  assign n24 = i5 & ~n23;
  assign n25 = ~i5 & n23;
  assign n26 = ~n24 & ~n25;
  assign n27 = i6 & ~n26;
  assign n28 = ~i6 & n26;
  assign n29 = ~n27 & ~n28;
  assign n30 = i7 & ~n29;
  assign n31 = ~i7 & n29;
  assign i8 = n30 | n31;
  assign i9 = 1'b1;
endmodule


