// Benchmark "SKOLEMFORMULA" written by ABC on Fri May 27 01:48:57 2022

module SKOLEMFORMULA ( 
    i0, i1, i2, i3, i4,
    i5, i6  );
  input  i0, i1, i2, i3, i4;
  output i5, i6;
  wire n8, n9, n10, n11, n12, n13, n14, n15, n16, n17, n18, n19, n20, n21,
    n22, n24, n25, n26, n27, n28, n29, n30, n31, n32, n33, n34, n35, n36,
    n37;
  assign n8 = i2 & ~i3;
  assign n9 = ~i1 & n8;
  assign n10 = ~i0 & n9;
  assign n11 = i2 & ~n10;
  assign n12 = i0 & n9;
  assign n13 = i4 & n12;
  assign n14 = n11 & ~n13;
  assign n15 = i1 & n8;
  assign n16 = n14 & ~n15;
  assign n17 = i2 & i3;
  assign n18 = ~i4 & n17;
  assign n19 = ~i1 & n18;
  assign n20 = i0 & n19;
  assign n21 = n16 & ~n20;
  assign n22 = i4 & n17;
  assign i6 = ~n21 | n22;
  assign n24 = ~i0 & i1;
  assign n25 = i0 & ~i1;
  assign n26 = ~n24 & ~n25;
  assign n27 = i2 & n26;
  assign n28 = ~i2 & ~n26;
  assign n29 = ~n27 & ~n28;
  assign n30 = i3 & n29;
  assign n31 = ~i3 & ~n29;
  assign n32 = ~n30 & ~n31;
  assign n33 = i4 & n32;
  assign n34 = ~i4 & ~n32;
  assign n35 = ~n33 & ~n34;
  assign n36 = i6 & ~n35;
  assign n37 = ~i6 & n35;
  assign i5 = n36 | n37;
endmodule


