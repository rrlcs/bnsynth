// Benchmark "SKOLEMFORMULA" written by ABC on Sun May 22 02:42:12 2022

module SKOLEMFORMULA ( 
    i0, i1, i2, i3, i4, i5,
    i6, i7  );
  input  i0, i1, i2, i3, i4, i5;
  output i6, i7;
  wire n9, n10, n11, n12, n13, n14, n15, n16, n17, n18, n19, n20, n21, n22,
    n23, n24, n25, n26, n27, n28, n29, n30, n31, n32, n33, n34, n35, n36,
    n37, n38, n39, n40, n41, n42, n43, n44, n45, n47, n48, n49, n50, n51,
    n52, n53, n54, n55, n56, n57, n58, n59, n60, n61;
  assign n9 = ~i0 & ~i1;
  assign n10 = ~i2 & n9;
  assign n11 = ~i3 & n10;
  assign n12 = ~i4 & n11;
  assign n13 = i5 & n12;
  assign n14 = i0 & ~i1;
  assign n15 = i2 & n14;
  assign n16 = i3 & n15;
  assign n17 = i4 & n16;
  assign n18 = i5 & n17;
  assign n19 = ~i0 & i1;
  assign n20 = i2 & n19;
  assign n21 = i3 & n20;
  assign n22 = ~i4 & n21;
  assign n23 = ~i5 & n22;
  assign n24 = i4 & n21;
  assign n25 = i5 & n24;
  assign n26 = ~i4 & n16;
  assign n27 = ~i5 & n26;
  assign n28 = ~i3 & n15;
  assign n29 = i4 & n28;
  assign n30 = ~i5 & n29;
  assign n31 = ~i4 & n28;
  assign n32 = i5 & n31;
  assign n33 = ~i2 & n19;
  assign n34 = ~i3 & n33;
  assign n35 = ~i4 & n34;
  assign n36 = ~i5 & n35;
  assign n37 = i4 & n34;
  assign n38 = i5 & n37;
  assign n39 = ~n13 & ~n18;
  assign n40 = ~n23 & n39;
  assign n41 = ~n25 & n40;
  assign n42 = ~n27 & n41;
  assign n43 = ~n30 & n42;
  assign n44 = ~n32 & n43;
  assign n45 = ~n36 & n44;
  assign i6 = n38 | ~n45;
  assign n47 = ~n14 & ~n19;
  assign n48 = i2 & ~n47;
  assign n49 = ~i2 & n47;
  assign n50 = ~n48 & ~n49;
  assign n51 = i3 & ~n50;
  assign n52 = ~i3 & n50;
  assign n53 = ~n51 & ~n52;
  assign n54 = i4 & ~n53;
  assign n55 = ~i4 & n53;
  assign n56 = ~n54 & ~n55;
  assign n57 = i5 & ~n56;
  assign n58 = ~i5 & n56;
  assign n59 = ~n57 & ~n58;
  assign n60 = i6 & ~n59;
  assign n61 = ~i6 & n59;
  assign i7 = n60 | n61;
endmodule


