// Benchmark "SKOLEMFORMULA" written by ABC on Sun May 22 06:01:25 2022

module SKOLEMFORMULA ( 
    i0, i1, i2, i3,
    i4, i5, i6  );
  input  i0, i1, i2, i3;
  output i4, i5, i6;
  wire n8, n9, n10, n11, n12, n13, n14, n16, n17, n18, n19, n20, n21, n22,
    n24, n25, n26, n27, n28, n29, n30, n31, n32, n33, n34, n35, n36, n37,
    n38, n39, n40, n41, n42, n43, n44, n45, n46, n47, n48, n49, n50, n51,
    n52, n53, n54, n55, n56, n57, n58, n59, n60, n61, n62;
  assign n8 = i1 & ~i3;
  assign n9 = ~i0 & n8;
  assign n10 = i2 & n9;
  assign n11 = i1 & ~n10;
  assign n12 = i0 & n8;
  assign n13 = n11 & ~n12;
  assign n14 = i1 & i3;
  assign i6 = ~n13 | n14;
  assign n16 = ~i2 & ~i3;
  assign n17 = ~i1 & n16;
  assign n18 = ~i0 & n17;
  assign n19 = i1 & n16;
  assign n20 = ~n18 & ~n19;
  assign n21 = i2 & ~i3;
  assign n22 = n20 & ~n21;
  assign i5 = i3 | ~n22;
  assign n24 = ~i0 & i1;
  assign n25 = i2 & n24;
  assign n26 = i3 & n25;
  assign n27 = i5 & n26;
  assign n28 = i6 & n27;
  assign n29 = ~i0 & ~i1;
  assign n30 = i2 & n29;
  assign n31 = ~i3 & n30;
  assign n32 = i5 & n31;
  assign n33 = i6 & n32;
  assign n34 = i0 & i1;
  assign n35 = i2 & n34;
  assign n36 = ~i3 & n35;
  assign n37 = i5 & n36;
  assign n38 = i6 & n37;
  assign n39 = ~i2 & n34;
  assign n40 = i3 & n39;
  assign n41 = i5 & n40;
  assign n42 = i6 & n41;
  assign n43 = ~i2 & i3;
  assign n44 = ~i1 & n43;
  assign n45 = i0 & n44;
  assign n46 = ~n16 & ~n45;
  assign n47 = i1 & n43;
  assign n48 = n46 & ~n47;
  assign n49 = i2 & ~i6;
  assign n50 = n48 & ~n49;
  assign n51 = i2 & i6;
  assign n52 = ~i3 & n51;
  assign n53 = n50 & ~n52;
  assign n54 = i3 & n51;
  assign n55 = ~i0 & n54;
  assign n56 = n53 & ~n55;
  assign n57 = i0 & n54;
  assign n58 = i1 & n57;
  assign n59 = n56 & ~n58;
  assign n60 = ~n28 & ~n59;
  assign n61 = ~n33 & n60;
  assign n62 = ~n38 & n61;
  assign i4 = ~n42 & n62;
endmodule


