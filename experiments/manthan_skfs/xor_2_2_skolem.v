// Benchmark "SKOLEMFORMULA" written by ABC on Fri May 20 19:56:40 2022

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


