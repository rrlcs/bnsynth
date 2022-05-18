// Benchmark "SKOLEMFORMULA" written by ABC on Wed May 18 01:05:08 2022

module SKOLEMFORMULA ( 
    i0, i1,
    i2, i3, i4  );
  input  i0, i1;
  output i2, i3, i4;
  wire n7, n9;
  assign n7 = ~i0 & ~i1;
  assign i2 = i0 | n7;
  assign n9 = i0 & i1;
  assign i4 = ~i0 | n9;
  assign i3 = 1'b1;
endmodule


