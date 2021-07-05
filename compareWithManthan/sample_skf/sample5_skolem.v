// Benchmark "SKOLEMFORMULA" written by ABC on Thu Jul  1 14:01:40 2021

module SKOLEMFORMULA ( 
    i0, i1, i2, i3,
    i4  );
  input  i0, i1, i2, i3;
  output i4;
  wire n6, n7, n8, n9, n10, n11, n12, n13, n14, n15, n16;
  assign n6 = ~i0 & ~i3;
  assign n7 = ~i1 & n6;
  assign n8 = ~i2 & i3;
  assign n9 = ~i0 & n8;
  assign n10 = ~i1 & n9;
  assign n11 = ~n7 & ~n10;
  assign n12 = i2 & i3;
  assign n13 = ~i1 & n12;
  assign n14 = i0 & n13;
  assign n15 = n11 & ~n14;
  assign n16 = i1 & n12;
  assign i4 = ~n15 | n16;
endmodule


