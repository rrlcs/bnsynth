// Benchmark "SKOLEMFORMULA" written by ABC on Sun May 22 04:06:51 2022

module SKOLEMFORMULA ( 
    i0, i1, i2, i3, i4, i5, i6, i7,
    i8, i9, i10, i11, i12, i13, i14, i15  );
  input  i0, i1, i2, i3, i4, i5, i6, i7;
  output i8, i9, i10, i11, i12, i13, i14, i15;
  wire n18, n19, n20, n22, n23, n24, n25, n26, n27, n28, n29, n30, n31, n32,
    n33, n34, n35, n37, n38, n39, n40, n41, n42, n43, n44, n45, n46, n47,
    n48, n49, n50, n51, n52, n53, n54, n55, n56, n57, n59, n60, n61, n62,
    n63, n64, n65, n66, n68, n69, n70, n71, n72, n73, n74, n75, n76, n77,
    n78, n79, n81, n82, n83, n84, n85, n86, n87, n88, n89, n90, n91, n92,
    n93, n94, n95, n96, n97, n98, n99, n100, n101, n102, n103, n104, n105,
    n106, n107, n108, n109, n110, n111, n112, n113, n114, n115;
  assign n18 = i1 & ~i4;
  assign n19 = i1 & ~n18;
  assign n20 = i1 & i4;
  assign i13 = ~n19 | n20;
  assign n22 = i0 & i13;
  assign n23 = ~i0 & i13;
  assign n24 = ~i1 & n23;
  assign n25 = i13 & ~n24;
  assign n26 = i1 & n23;
  assign n27 = ~i6 & n26;
  assign n28 = n25 & ~n27;
  assign n29 = i6 & n26;
  assign n30 = i5 & n29;
  assign n31 = ~i4 & n30;
  assign n32 = n28 & ~n31;
  assign n33 = i4 & n30;
  assign n34 = ~i3 & n33;
  assign n35 = n32 & ~n34;
  assign i12 = n22 | ~n35;
  assign n37 = ~i4 & ~i5;
  assign n38 = ~i0 & n37;
  assign n39 = ~i7 & n38;
  assign n40 = i3 & n39;
  assign n41 = i7 & n38;
  assign n42 = ~n40 & ~n41;
  assign n43 = i0 & n37;
  assign n44 = n42 & ~n43;
  assign n45 = i4 & ~i5;
  assign n46 = ~i3 & n45;
  assign n47 = n44 & ~n46;
  assign n48 = i3 & n45;
  assign n49 = i6 & n48;
  assign n50 = ~i2 & n49;
  assign n51 = i7 & n50;
  assign n52 = n47 & ~n51;
  assign n53 = i2 & n49;
  assign n54 = n52 & ~n53;
  assign n55 = i5 & ~i12;
  assign n56 = n54 & ~n55;
  assign n57 = i5 & i12;
  assign i11 = ~n56 | n57;
  assign n59 = ~i5 & i13;
  assign n60 = i13 & ~n59;
  assign n61 = i5 & i13;
  assign n62 = ~i12 & n61;
  assign n63 = n60 & ~n62;
  assign n64 = i12 & n61;
  assign n65 = i0 & n64;
  assign n66 = i2 & n65;
  assign i9 = ~n63 | n66;
  assign n68 = i7 & ~i9;
  assign n69 = ~i4 & n68;
  assign n70 = ~i11 & n69;
  assign n71 = ~i2 & n70;
  assign n72 = i7 & ~n71;
  assign n73 = i2 & n70;
  assign n74 = i3 & n73;
  assign n75 = n72 & ~n74;
  assign n76 = i4 & n68;
  assign n77 = i13 & n76;
  assign n78 = n75 & ~n77;
  assign n79 = i7 & i9;
  assign i10 = ~n78 | n79;
  assign n81 = ~i0 & i1;
  assign n82 = i0 & ~i1;
  assign n83 = ~n81 & ~n82;
  assign n84 = i2 & n83;
  assign n85 = ~i2 & ~n83;
  assign n86 = ~n84 & ~n85;
  assign n87 = i3 & n86;
  assign n88 = ~i3 & ~n86;
  assign n89 = ~n87 & ~n88;
  assign n90 = i4 & n89;
  assign n91 = ~i4 & ~n89;
  assign n92 = ~n90 & ~n91;
  assign n93 = i5 & n92;
  assign n94 = ~i5 & ~n92;
  assign n95 = ~n93 & ~n94;
  assign n96 = i6 & n95;
  assign n97 = ~i6 & ~n95;
  assign n98 = ~n96 & ~n97;
  assign n99 = i7 & n98;
  assign n100 = ~i7 & ~n98;
  assign n101 = ~n99 & ~n100;
  assign n102 = i9 & ~n101;
  assign n103 = ~i9 & n101;
  assign n104 = ~n102 & ~n103;
  assign n105 = i10 & n104;
  assign n106 = ~i10 & ~n104;
  assign n107 = ~n105 & ~n106;
  assign n108 = i11 & n107;
  assign n109 = ~i11 & ~n107;
  assign n110 = ~n108 & ~n109;
  assign n111 = i12 & n110;
  assign n112 = ~i12 & ~n110;
  assign n113 = ~n111 & ~n112;
  assign n114 = i13 & n113;
  assign n115 = ~i13 & ~n113;
  assign i8 = n114 | n115;
  assign i14 = 1'b1;
  assign i15 = 1'b1;
endmodule


