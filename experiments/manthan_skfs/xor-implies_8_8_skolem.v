// Benchmark "SKOLEMFORMULA" written by ABC on Sun May 22 05:35:39 2022

module SKOLEMFORMULA ( 
    i0, i1, i2, i3, i4, i5, i6, i7,
    i8, i9, i10, i11, i12, i13, i14, i15  );
  input  i0, i1, i2, i3, i4, i5, i6, i7;
  output i8, i9, i10, i11, i12, i13, i14, i15;
  wire n18, n19, n20, n21, n22, n23, n24, n25, n26, n27, n28, n29, n30, n31,
    n32, n33, n34, n35, n36, n37;
  assign n18 = ~i0 & i1;
  assign n19 = i0 & ~i1;
  assign n20 = ~n18 & ~n19;
  assign n21 = i2 & n20;
  assign n22 = ~i2 & ~n20;
  assign n23 = ~n21 & ~n22;
  assign n24 = i3 & n23;
  assign n25 = ~i3 & ~n23;
  assign n26 = ~n24 & ~n25;
  assign n27 = i4 & n26;
  assign n28 = ~i4 & ~n26;
  assign n29 = ~n27 & ~n28;
  assign n30 = i5 & n29;
  assign n31 = ~i5 & ~n29;
  assign n32 = ~n30 & ~n31;
  assign n33 = i6 & n32;
  assign n34 = ~i6 & ~n32;
  assign n35 = ~n33 & ~n34;
  assign n36 = i7 & n35;
  assign n37 = ~i7 & ~n35;
  assign i8 = n36 | n37;
  assign i9 = 1'b1;
  assign i10 = 1'b1;
  assign i11 = 1'b1;
  assign i12 = 1'b1;
  assign i13 = 1'b1;
  assign i14 = 1'b1;
  assign i15 = 1'b1;
endmodule


