// Benchmark "SKOLEMFORMULA" written by ABC on Fri May 20 20:54:54 2022

module SKOLEMFORMULA ( 
    i0, i1, i2, i3,
    i4  );
  input  i0, i1, i2, i3;
  output i4;
  wire n6, n7, n8, n9, n10, n11, n12, n13, n14, n15, n16, n17, n18, n19, n20,
    n21, n22, n23, n24, n25, n26, n27, n28, n29, n30, n31;
  assign n6 = i0 & ~i1;
  assign n7 = ~i2 & n6;
  assign n8 = ~i3 & n7;
  assign n9 = ~i0 & ~i1;
  assign n10 = ~i2 & n9;
  assign n11 = i3 & n10;
  assign n12 = ~i0 & i1;
  assign n13 = ~i2 & n12;
  assign n14 = ~i3 & n13;
  assign n15 = i2 & n9;
  assign n16 = ~i3 & n15;
  assign n17 = i0 & i1;
  assign n18 = ~i2 & n17;
  assign n19 = i3 & n18;
  assign n20 = i2 & n17;
  assign n21 = ~i3 & n20;
  assign n22 = i2 & n12;
  assign n23 = i3 & n22;
  assign n24 = i2 & n6;
  assign n25 = i3 & n24;
  assign n26 = ~n8 & ~n11;
  assign n27 = ~n14 & n26;
  assign n28 = ~n16 & n27;
  assign n29 = ~n19 & n28;
  assign n30 = ~n21 & n29;
  assign n31 = ~n23 & n30;
  assign i4 = ~n25 & n31;
endmodule


