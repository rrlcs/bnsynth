// Benchmark "SKOLEMFORMULA" written by ABC on Sun May 22 03:46:54 2022

module SKOLEMFORMULA ( 
    i0, i1,
    i2, i3  );
  input  i0, i1;
  output i2, i3;
  wire n5, n7;
  assign n5 = i0 & ~i1;
  assign i2 = ~i0 | n5;
  assign n7 = ~i0 & i1;
  assign i3 = i0 | n7;
endmodule


