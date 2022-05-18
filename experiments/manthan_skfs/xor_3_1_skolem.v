// Benchmark "SKOLEMFORMULA" written by ABC on Wed May 18 01:04:39 2022

module SKOLEMFORMULA ( 
    i0, i1, i2,
    i3  );
  input  i0, i1, i2;
  output i3;
  wire n5, n6, n7, n8, n9, n10, n11, n12, n13, n14;
  assign n5 = i0 & ~i1;
  assign n6 = ~i2 & n5;
  assign n7 = ~i0 & i1;
  assign n8 = ~i2 & n7;
  assign n9 = ~i0 & ~i1;
  assign n10 = i2 & n9;
  assign n11 = i0 & i1;
  assign n12 = i2 & n11;
  assign n13 = ~n6 & ~n8;
  assign n14 = ~n10 & n13;
  assign i3 = ~n12 & n14;
endmodule


