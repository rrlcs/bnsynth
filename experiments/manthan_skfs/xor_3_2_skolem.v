// Benchmark "SKOLEMFORMULA" written by ABC on Fri May 20 20:27:16 2022

module SKOLEMFORMULA ( 
    i0, i1, i2,
    i3, i4  );
  input  i0, i1, i2;
  output i3, i4;
  wire n6, n8, n9, n10, n11, n12, n13;
  assign n6 = i1 & ~i2;
  assign i3 = i2 | n6;
  assign n8 = ~i0 & ~i1;
  assign n9 = ~i0 & i1;
  assign n10 = ~i2 & n9;
  assign n11 = ~n8 & ~n10;
  assign n12 = i0 & i2;
  assign n13 = i1 & n12;
  assign i4 = ~n11 | n13;
endmodule


