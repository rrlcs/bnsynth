// Benchmark "SKOLEMFORMULA" written by ABC on Wed May 18 00:57:32 2022

module SKOLEMFORMULA ( 
    i0, i1, i2, i3,
    i4, i5  );
  input  i0, i1, i2, i3;
  output i4, i5;
  wire n7, n8, n9, n10, n11, n12, n13, n14, n16, n17, n18, n19, n20, n21,
    n22, n23, n24, n25, n26, n27, n28, n29, n30, n31, n32, n33, n34, n35,
    n36, n37, n38, n39, n40, n41, n42, n43, n44;
  assign n7 = ~i0 & i2;
  assign n8 = ~i3 & n7;
  assign n9 = ~i1 & n8;
  assign n10 = i2 & ~n9;
  assign n11 = i3 & n7;
  assign n12 = n10 & ~n11;
  assign n13 = i0 & i2;
  assign n14 = i1 & n13;
  assign i5 = ~n12 | n14;
  assign n16 = ~i0 & ~i1;
  assign n17 = ~i2 & n16;
  assign n18 = ~i3 & n17;
  assign n19 = i5 & n18;
  assign n20 = i0 & i1;
  assign n21 = ~i2 & n20;
  assign n22 = ~i3 & n21;
  assign n23 = i5 & n22;
  assign n24 = i0 & ~i1;
  assign n25 = ~i2 & n24;
  assign n26 = i3 & n25;
  assign n27 = i5 & n26;
  assign n28 = ~i0 & i1;
  assign n29 = ~i2 & n28;
  assign n30 = i3 & n29;
  assign n31 = i5 & n30;
  assign n32 = ~i2 & ~i3;
  assign n33 = i2 & ~i3;
  assign n34 = ~n32 & ~n33;
  assign n35 = i3 & i5;
  assign n36 = ~i2 & n35;
  assign n37 = n34 & ~n36;
  assign n38 = i2 & n35;
  assign n39 = i1 & n38;
  assign n40 = ~i0 & n39;
  assign n41 = n37 & ~n40;
  assign n42 = ~n19 & ~n41;
  assign n43 = ~n23 & n42;
  assign n44 = ~n27 & n43;
  assign i4 = ~n31 & n44;
endmodule


