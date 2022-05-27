// Benchmark "SKOLEMFORMULA" written by ABC on Tue May 24 11:29:45 2022

module SKOLEMFORMULA ( 
    i0, i1,
    i2, i3, i4, i5  );
  input  i0, i1;
  output i2, i3, i4, i5;
  wire n8, n10, n11;
  assign n8 = ~i0 & ~i1;
  assign i2 = i0 | n8;
  assign n10 = ~i1 & i2;
  assign n11 = i0 & n10;
  assign i3 = ~i2 | n11;
  assign i5 = 1'b1;
  assign i4 = i2;
endmodule


