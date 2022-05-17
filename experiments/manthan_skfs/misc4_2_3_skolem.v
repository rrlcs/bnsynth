// Benchmark "SKOLEMFORMULA" written by ABC on Tue May 17 23:38:50 2022

module SKOLEMFORMULA ( 
    i0, i1,
    i2, i3, i4  );
  input  i0, i1;
  output i2, i3, i4;
  wire n8, n9, n10;
  assign i4 = ~i0 & ~i1;
  assign n8 = ~i0 & ~i4;
  assign n9 = i0 & ~i4;
  assign n10 = ~i1 & n9;
  assign i3 = n8 | n10;
  assign i2 = 1'b1;
endmodule


