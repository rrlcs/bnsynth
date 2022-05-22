// Benchmark "SKOLEMFORMULA" written by ABC on Sun May 22 03:45:05 2022

module SKOLEMFORMULA ( 
    i0, i1, i2,
    i3  );
  input  i0, i1, i2;
  output i3;
  wire n5, n6;
  assign n5 = i0 & i2;
  assign n6 = i1 & n5;
  assign i3 = ~i2 | n6;
endmodule


