// Benchmark "SKOLEMFORMULA" written by ABC on Fri May 20 21:05:20 2022

module SKOLEMFORMULA ( 
    i1, i2,
    i0, i3, i4  );
  input  i1, i2;
  output i0, i3, i4;
  wire n7, n9, n10;
  assign n7 = i1 & i2;
  assign i4 = ~i1 | n7;
  assign n9 = ~i1 & i4;
  assign n10 = i2 & n9;
  assign i0 = ~i4 | n10;
  assign i3 = 1'b1;
endmodule


