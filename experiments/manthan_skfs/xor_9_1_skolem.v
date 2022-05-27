// Benchmark "SKOLEMFORMULA" written by ABC on Wed May 25 16:28:39 2022

module SKOLEMFORMULA ( 
    i0, i1, i2, i3, i4, i5, i6, i7, i8,
    i9  );
  input  i0, i1, i2, i3, i4, i5, i6, i7, i8;
  output i9;
  wire n11, n12, n13, n14, n15, n16, n17, n18, n19, n20, n21, n22, n23, n24,
    n25, n26, n27, n28, n29, n30, n31, n32, n33;
  assign n11 = ~i0 & i1;
  assign n12 = i0 & ~i1;
  assign n13 = ~n11 & ~n12;
  assign n14 = i2 & n13;
  assign n15 = ~i2 & ~n13;
  assign n16 = ~n14 & ~n15;
  assign n17 = i3 & n16;
  assign n18 = ~i3 & ~n16;
  assign n19 = ~n17 & ~n18;
  assign n20 = i4 & n19;
  assign n21 = ~i4 & ~n19;
  assign n22 = ~n20 & ~n21;
  assign n23 = i5 & n22;
  assign n24 = ~i5 & ~n22;
  assign n25 = ~n23 & ~n24;
  assign n26 = i6 & n25;
  assign n27 = ~i6 & ~n25;
  assign n28 = ~n26 & ~n27;
  assign n29 = i7 & n28;
  assign n30 = ~i7 & ~n28;
  assign n31 = ~n29 & ~n30;
  assign n32 = i8 & n31;
  assign n33 = ~i8 & ~n31;
  assign i9 = ~n32 & ~n33;
endmodule


