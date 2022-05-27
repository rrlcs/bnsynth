// Benchmark "SKOLEMFORMULA" written by ABC on Tue May 24 12:04:49 2022

module SKOLEMFORMULA ( 
    i0, i1, i2, i3, i4,
    i5, i6, i7  );
  input  i0, i1, i2, i3, i4;
  output i5, i6, i7;
  wire n9, n10, n11, n12, n13, n14, n15, n16, n17, n18, n19, n20, n21, n22,
    n23, n24, n25, n26, n27, n29, n30, n31, n32, n33, n34, n35, n36, n37,
    n38, n39, n40, n41, n42, n43, n44, n45, n46, n47, n48, n49, n50, n51,
    n52, n53, n54, n55, n57, n58, n59, n60, n61, n62, n63, n64, n65, n66,
    n67, n68, n69, n70, n71, n72, n73, n74, n75, n76, n77, n78, n79, n80,
    n81, n82, n83, n84, n85, n86, n87, n88, n89, n90, n91, n92, n93, n94,
    n95, n96, n97, n98, n99, n100, n101, n102, n103, n104, n105, n106,
    n107, n108, n109, n110, n111, n112, n113;
  assign n9 = ~i3 & ~i4;
  assign n10 = ~i1 & n9;
  assign n11 = ~i2 & n10;
  assign n12 = i0 & n11;
  assign n13 = i1 & n9;
  assign n14 = ~i0 & n13;
  assign n15 = ~n12 & ~n14;
  assign n16 = i0 & n13;
  assign n17 = i2 & n16;
  assign n18 = n15 & ~n17;
  assign n19 = ~i3 & i4;
  assign n20 = ~i2 & n19;
  assign n21 = n18 & ~n20;
  assign n22 = i2 & n19;
  assign n23 = ~i0 & n22;
  assign n24 = n21 & ~n23;
  assign n25 = i0 & n22;
  assign n26 = ~i1 & n25;
  assign n27 = n24 & ~n26;
  assign i5 = i3 | ~n27;
  assign n29 = ~i1 & ~i2;
  assign n30 = i4 & n29;
  assign n31 = i0 & n30;
  assign n32 = ~i3 & n31;
  assign n33 = ~i1 & i2;
  assign n34 = ~i4 & n33;
  assign n35 = ~i3 & n34;
  assign n36 = i0 & n35;
  assign n37 = ~n32 & ~n36;
  assign n38 = i3 & n34;
  assign n39 = n37 & ~n38;
  assign n40 = i4 & n33;
  assign n41 = ~i3 & n40;
  assign n42 = ~i0 & n41;
  assign n43 = n39 & ~n42;
  assign n44 = i1 & ~i3;
  assign n45 = i5 & n44;
  assign n46 = ~i0 & n45;
  assign n47 = ~i2 & n46;
  assign n48 = ~i4 & n47;
  assign n49 = n43 & ~n48;
  assign n50 = i2 & n46;
  assign n51 = n49 & ~n50;
  assign n52 = i0 & n45;
  assign n53 = i4 & n52;
  assign n54 = n51 & ~n53;
  assign n55 = i1 & i3;
  assign i6 = ~n54 | n55;
  assign n57 = i0 & ~i1;
  assign n58 = ~i2 & n57;
  assign n59 = ~i3 & n58;
  assign n60 = i4 & n59;
  assign n61 = i5 & n60;
  assign n62 = i6 & n61;
  assign n63 = ~i0 & i1;
  assign n64 = ~i2 & n63;
  assign n65 = i3 & n64;
  assign n66 = i4 & n65;
  assign n67 = i5 & n66;
  assign n68 = i6 & n67;
  assign n69 = i0 & i1;
  assign n70 = i2 & n69;
  assign n71 = i3 & n70;
  assign n72 = ~i4 & n71;
  assign n73 = i5 & n72;
  assign n74 = i6 & n73;
  assign n75 = ~i2 & n69;
  assign n76 = i3 & n75;
  assign n77 = i4 & n76;
  assign n78 = i5 & n77;
  assign n79 = i6 & n78;
  assign n80 = ~i3 & ~i6;
  assign n81 = ~i0 & n80;
  assign n82 = ~i2 & n81;
  assign n83 = ~i1 & n82;
  assign n84 = i0 & n80;
  assign n85 = ~n83 & ~n84;
  assign n86 = i3 & ~i6;
  assign n87 = ~i0 & n86;
  assign n88 = ~i2 & n87;
  assign n89 = ~i4 & n88;
  assign n90 = n85 & ~n89;
  assign n91 = i2 & n87;
  assign n92 = n90 & ~n91;
  assign n93 = i0 & n86;
  assign n94 = ~i2 & n93;
  assign n95 = i4 & n94;
  assign n96 = n92 & ~n95;
  assign n97 = ~i0 & i6;
  assign n98 = ~i1 & n97;
  assign n99 = n96 & ~n98;
  assign n100 = i1 & n97;
  assign n101 = ~i3 & n100;
  assign n102 = i2 & n101;
  assign n103 = ~i4 & n102;
  assign n104 = n99 & ~n103;
  assign n105 = i3 & n100;
  assign n106 = ~i2 & n105;
  assign n107 = n104 & ~n106;
  assign n108 = i2 & n105;
  assign n109 = i4 & n108;
  assign n110 = n107 & ~n109;
  assign n111 = ~n62 & n110;
  assign n112 = ~n68 & ~n111;
  assign n113 = ~n74 & ~n112;
  assign i7 = n79 | ~n113;
endmodule


