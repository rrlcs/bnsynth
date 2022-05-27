// Benchmark "SKOLEMFORMULA" written by ABC on Wed May 25 11:17:36 2022

module SKOLEMFORMULA ( 
    i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15,
    i16, i17, i18, i19, i20, i21, i22, i23, i24, i25, i26, i27, i28, i29,
    i30, i31,
    i32  );
  input  i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14,
    i15, i16, i17, i18, i19, i20, i21, i22, i23, i24, i25, i26, i27, i28,
    i29, i30, i31;
  output i32;
  wire n34, n35, n36, n37, n38, n39, n40, n41, n42, n43, n44;
  assign n34 = ~i29 & ~i30;
  assign n35 = i27 & n34;
  assign n36 = i28 & n35;
  assign n37 = i29 & i30;
  assign n38 = ~i26 & n37;
  assign n39 = ~i27 & n38;
  assign n40 = i28 & n39;
  assign n41 = ~n36 & ~n40;
  assign n42 = i27 & n38;
  assign n43 = n41 & ~n42;
  assign n44 = i26 & n37;
  assign i32 = ~n43 | n44;
endmodule


