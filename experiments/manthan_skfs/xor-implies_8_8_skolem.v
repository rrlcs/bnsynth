// Benchmark "SKOLEMFORMULA" written by ABC on Fri May 20 22:00:31 2022

module SKOLEMFORMULA ( 
    i0, i1, i2, i3, i4, i5, i6, i7,
    i8, i9, i10, i11, i12, i13, i14, i15  );
  input  i0, i1, i2, i3, i4, i5, i6, i7;
  output i8, i9, i10, i11, i12, i13, i14, i15;
  wire n18, n19, n20, n21, n22, n23, n24, n26, n27, n28, n30, n31, n32, n33,
    n34, n35, n36, n37, n38, n39, n40, n41, n42, n43, n44, n45, n46, n47,
    n49, n50, n51, n52, n54, n55, n56, n57, n59, n60, n61, n62, n63, n64,
    n65, n66, n68, n69, n70, n71, n72, n73, n74, n75, n76, n77, n78, n79,
    n80, n81, n82, n83, n84, n85, n86, n87, n88, n89, n90, n91, n92, n93,
    n94, n95, n96, n97, n98, n99, n100, n101, n102, n103, n104, n105, n106,
    n107, n108, n109, n110, n111, n112;
  assign n18 = ~i3 & ~i4;
  assign n19 = ~i7 & n18;
  assign n20 = ~i6 & n19;
  assign n21 = i6 & n19;
  assign n22 = ~n20 & ~n21;
  assign n23 = ~i3 & i4;
  assign n24 = n22 & ~n23;
  assign i12 = i3 | ~n24;
  assign n26 = ~i6 & ~i12;
  assign n27 = ~i6 & i12;
  assign n28 = ~n26 & ~n27;
  assign i8 = i6 | ~n28;
  assign n30 = ~i0 & ~i1;
  assign n31 = ~i6 & n30;
  assign n32 = i6 & n30;
  assign n33 = ~i5 & n32;
  assign n34 = ~n31 & ~n33;
  assign n35 = i5 & n32;
  assign n36 = n34 & ~n35;
  assign n37 = ~i0 & i1;
  assign n38 = ~i3 & n37;
  assign n39 = i5 & n38;
  assign n40 = i7 & n39;
  assign n41 = n36 & ~n40;
  assign n42 = i3 & n37;
  assign n43 = ~i4 & n42;
  assign n44 = i2 & n43;
  assign n45 = n41 & ~n44;
  assign n46 = i4 & n42;
  assign n47 = n45 & ~n46;
  assign i15 = i0 | ~n47;
  assign n49 = ~i1 & ~i15;
  assign n50 = ~i2 & n49;
  assign n51 = ~i1 & i15;
  assign n52 = ~n50 & ~n51;
  assign i14 = i1 | ~n52;
  assign n54 = i3 & ~i15;
  assign n55 = ~i14 & n54;
  assign n56 = i3 & ~n55;
  assign n57 = i3 & i15;
  assign i10 = ~n56 | n57;
  assign n59 = ~i2 & ~i15;
  assign n60 = i10 & n59;
  assign n61 = ~i2 & i15;
  assign n62 = ~i5 & n61;
  assign n63 = ~i7 & n62;
  assign n64 = ~n60 & ~n63;
  assign n65 = i5 & n61;
  assign n66 = n64 & ~n65;
  assign i13 = i2 | ~n66;
  assign n68 = i0 & ~i1;
  assign n69 = ~n37 & ~n68;
  assign n70 = i2 & n69;
  assign n71 = ~i2 & ~n69;
  assign n72 = ~n70 & ~n71;
  assign n73 = i3 & n72;
  assign n74 = ~i3 & ~n72;
  assign n75 = ~n73 & ~n74;
  assign n76 = i4 & n75;
  assign n77 = ~i4 & ~n75;
  assign n78 = ~n76 & ~n77;
  assign n79 = i5 & n78;
  assign n80 = ~i5 & ~n78;
  assign n81 = ~n79 & ~n80;
  assign n82 = i6 & n81;
  assign n83 = ~i6 & ~n81;
  assign n84 = ~n82 & ~n83;
  assign n85 = i7 & n84;
  assign n86 = ~i7 & ~n84;
  assign n87 = ~n85 & ~n86;
  assign n88 = i8 & n87;
  assign n89 = ~i8 & ~n87;
  assign n90 = ~n88 & ~n89;
  assign n91 = i10 & ~n90;
  assign n92 = ~i10 & n90;
  assign n93 = ~n91 & ~n92;
  assign n94 = i12 & ~n93;
  assign n95 = ~i12 & n93;
  assign n96 = ~n94 & ~n95;
  assign n97 = i13 & n96;
  assign n98 = ~i13 & ~n96;
  assign n99 = ~n97 & ~n98;
  assign n100 = i14 & n99;
  assign n101 = ~i14 & ~n99;
  assign n102 = ~n100 & ~n101;
  assign n103 = i15 & n102;
  assign n104 = ~i15 & ~n102;
  assign n105 = ~n103 & ~n104;
  assign n106 = i0 & ~i15;
  assign n107 = ~n105 & ~n106;
  assign n108 = i1 & ~i14;
  assign n109 = n107 & ~n108;
  assign n110 = i2 & ~i13;
  assign n111 = n109 & ~n110;
  assign n112 = i3 & ~i12;
  assign i9 = n111 & ~n112;
  assign i11 = 1'b1;
endmodule


