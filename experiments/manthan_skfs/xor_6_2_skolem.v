// Benchmark "SKOLEMFORMULA" written by ABC on Fri May 20 22:15:50 2022

module SKOLEMFORMULA ( 
    i0, i1, i2, i3, i4, i5,
    i6, i7  );
  input  i0, i1, i2, i3, i4, i5;
  output i6, i7;
  wire n9, n10, n11, n12, n13, n14, n15, n16, n17, n18, n19, n20, n21, n22,
    n23, n24, n25, n26, n28, n29, n30, n31, n32, n33, n34, n35, n36, n37,
    n38, n39, n40, n41, n42, n43, n44;
  assign n9 = ~i1 & ~i5;
  assign n10 = ~i3 & n9;
  assign n11 = ~i2 & n10;
  assign n12 = ~i0 & n11;
  assign n13 = i0 & n11;
  assign n14 = ~i4 & n13;
  assign n15 = ~n12 & ~n14;
  assign n16 = i2 & n10;
  assign n17 = i4 & n16;
  assign n18 = ~i0 & n17;
  assign n19 = n15 & ~n18;
  assign n20 = i3 & n9;
  assign n21 = ~i4 & n20;
  assign n22 = n19 & ~n21;
  assign n23 = i4 & n20;
  assign n24 = n22 & ~n23;
  assign n25 = i1 & ~i5;
  assign n26 = n24 & ~n25;
  assign i7 = i5 | ~n26;
  assign n28 = ~i0 & i1;
  assign n29 = i0 & ~i1;
  assign n30 = ~n28 & ~n29;
  assign n31 = i2 & n30;
  assign n32 = ~i2 & ~n30;
  assign n33 = ~n31 & ~n32;
  assign n34 = i3 & n33;
  assign n35 = ~i3 & ~n33;
  assign n36 = ~n34 & ~n35;
  assign n37 = i4 & n36;
  assign n38 = ~i4 & ~n36;
  assign n39 = ~n37 & ~n38;
  assign n40 = i5 & n39;
  assign n41 = ~i5 & ~n39;
  assign n42 = ~n40 & ~n41;
  assign n43 = i7 & ~n42;
  assign n44 = ~i7 & n42;
  assign i6 = n43 | n44;
endmodule


