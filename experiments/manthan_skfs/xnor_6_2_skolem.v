// Benchmark "SKOLEMFORMULA" written by ABC on Mon May 23 18:29:27 2022

module SKOLEMFORMULA ( 
    i0, i1, i2, i3, i4, i5,
    i6, i7  );
  input  i0, i1, i2, i3, i4, i5;
  output i6, i7;
  wire n9, n10, n11, n12, n13, n14, n15, n16, n17, n18, n19, n20, n21, n22,
    n23, n24, n25, n27, n28, n29, n30, n31, n32, n33, n34, n35, n36, n37,
    n38, n39, n40, n41, n42;
  assign n9 = ~i1 & ~i2;
  assign n10 = ~i5 & n9;
  assign n11 = i5 & n9;
  assign n12 = ~i4 & n11;
  assign n13 = ~n10 & ~n12;
  assign n14 = i4 & n11;
  assign n15 = ~i3 & n14;
  assign n16 = i0 & n15;
  assign n17 = n13 & ~n16;
  assign n18 = ~i1 & i2;
  assign n19 = n17 & ~n18;
  assign n20 = ~i0 & i1;
  assign n21 = n19 & ~n20;
  assign n22 = i0 & i1;
  assign n23 = ~i3 & n22;
  assign n24 = n21 & ~n23;
  assign n25 = i3 & n22;
  assign i7 = ~n24 | n25;
  assign n27 = i0 & ~i1;
  assign n28 = ~n20 & ~n27;
  assign n29 = i2 & n28;
  assign n30 = ~i2 & ~n28;
  assign n31 = ~n29 & ~n30;
  assign n32 = i3 & n31;
  assign n33 = ~i3 & ~n31;
  assign n34 = ~n32 & ~n33;
  assign n35 = i4 & n34;
  assign n36 = ~i4 & ~n34;
  assign n37 = ~n35 & ~n36;
  assign n38 = i5 & n37;
  assign n39 = ~i5 & ~n37;
  assign n40 = ~n38 & ~n39;
  assign n41 = i7 & ~n40;
  assign n42 = ~i7 & n40;
  assign i6 = ~n41 & ~n42;
endmodule


