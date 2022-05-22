// Benchmark "SKOLEMFORMULA" written by ABC on Sun May 22 05:57:59 2022

module SKOLEMFORMULA ( 
    i0, i4, i5, i6,
    i1, i2, i3, i7, i8, i9, i10, i11, i12  );
  input  i0, i4, i5, i6;
  output i1, i2, i3, i7, i8, i9, i10, i11, i12;
  wire n15, n16, n17, n18, n19, n21, n23, n24, n25, n26, n27, n28, n29, n31,
    n32, n33, n34, n35, n36, n37, n38, n39, n40, n41, n42, n43, n44, n45,
    n48, n49, n50, n51, n52, n53, n54, n55, n56, n57, n58, n59, n60, n61,
    n62, n63, n64, n65, n67, n68, n69, n70, n71, n72;
  assign n15 = i4 & ~i5;
  assign n16 = ~i6 & n15;
  assign n17 = i0 & n16;
  assign n18 = i6 & n15;
  assign n19 = ~n17 & ~n18;
  assign i12 = i5 | ~n19;
  assign n21 = i0 & ~i12;
  assign i10 = i12 | n21;
  assign n23 = ~i12 & i10;
  assign n24 = i12 & i10;
  assign n25 = ~i0 & n24;
  assign n26 = i5 & n25;
  assign n27 = ~n23 & ~n26;
  assign n28 = i0 & n24;
  assign n29 = ~i5 & n28;
  assign i7 = ~n27 | n29;
  assign n31 = ~i0 & ~i12;
  assign n32 = ~i6 & n31;
  assign n33 = ~i4 & n32;
  assign n34 = i12 & ~i7;
  assign n35 = i6 & n34;
  assign n36 = i5 & n35;
  assign n37 = ~n33 & ~n36;
  assign n38 = i12 & i7;
  assign n39 = ~i6 & n38;
  assign n40 = n37 & ~n39;
  assign n41 = i6 & n38;
  assign n42 = ~i0 & n41;
  assign n43 = ~i4 & n42;
  assign n44 = n40 & ~n43;
  assign n45 = i0 & n41;
  assign i1 = n44 & ~n45;
  assign i11 = i4 & i10;
  assign n48 = ~i10 & i1;
  assign n49 = i10 & i1;
  assign n50 = ~i4 & n49;
  assign n51 = ~i5 & n50;
  assign n52 = i6 & n51;
  assign n53 = ~n48 & ~n52;
  assign n54 = i5 & n50;
  assign n55 = n53 & ~n54;
  assign n56 = i10 & ~i1;
  assign n57 = ~i11 & n56;
  assign n58 = i6 & n57;
  assign n59 = ~i0 & n58;
  assign n60 = n55 & ~n59;
  assign n61 = i11 & n56;
  assign n62 = ~i6 & n61;
  assign n63 = n60 & ~n62;
  assign n64 = i6 & n61;
  assign n65 = ~i7 & n64;
  assign i8 = ~n63 | n65;
  assign n67 = i6 & ~i11;
  assign n68 = ~i8 & n67;
  assign n69 = ~i6 & i11;
  assign n70 = ~i7 & n69;
  assign n71 = ~n68 & ~n70;
  assign n72 = i6 & i11;
  assign i3 = ~n71 | n72;
  assign i2 = 1'b1;
  assign i9 = ~i1;
endmodule


