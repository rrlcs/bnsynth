// Benchmark "SKOLEMFORMULA" written by ABC on Wed May 18 12:22:30 2022

module SKOLEMFORMULA ( 
    i0, i1, i2, i3, i4, i5,
    i6  );
  input  i0, i1, i2, i3, i4, i5;
  output i6;
  wire n8, n9, n10, n11, n12, n13, n14, n15, n16, n17, n18, n19, n20, n21,
    n22, n23, n24, n25, n26, n27, n28, n29, n30, n31, n32;
  assign n8 = ~i0 & ~i1;
  assign n9 = i2 & n8;
  assign n10 = i3 & n9;
  assign n11 = i4 & n10;
  assign n12 = i0 & ~i1;
  assign n13 = ~i2 & n12;
  assign n14 = i3 & n13;
  assign n15 = i4 & n14;
  assign n16 = ~i0 & i1;
  assign n17 = ~i2 & n16;
  assign n18 = i3 & n17;
  assign n19 = i4 & n18;
  assign n20 = i0 & i1;
  assign n21 = i2 & n20;
  assign n22 = i3 & n21;
  assign n23 = i4 & n22;
  assign n24 = ~i4 & ~i5;
  assign n25 = i4 & ~i5;
  assign n26 = ~i3 & n25;
  assign n27 = ~n24 & ~n26;
  assign n28 = i3 & n25;
  assign n29 = n27 & ~n28;
  assign n30 = ~n11 & ~n29;
  assign n31 = ~n15 & n30;
  assign n32 = ~n19 & n31;
  assign i6 = ~n23 & n32;
endmodule


