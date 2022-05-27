// Benchmark "SKOLEMFORMULA" written by ABC on Tue May 24 12:31:40 2022

module SKOLEMFORMULA ( 
    i0, i1, i2, i3, i4,
    i5, i6  );
  input  i0, i1, i2, i3, i4;
  output i5, i6;
  wire n9, n10, n11, n12, n13, n14;
  assign n9 = i1 & i2;
  assign n10 = ~i3 & n9;
  assign n11 = i4 & n10;
  assign n12 = i0 & n11;
  assign n13 = i1 & ~n12;
  assign n14 = i3 & n9;
  assign i5 = ~n13 | n14;
  assign i6 = 1'b1;
endmodule


