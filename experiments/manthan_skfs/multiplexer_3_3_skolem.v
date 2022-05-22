// Benchmark "SKOLEMFORMULA" written by ABC on Sun May 22 05:52:15 2022

module SKOLEMFORMULA ( 
    i0, i1, i2,
    i3, i4, i5  );
  input  i0, i1, i2;
  output i3, i4, i5;
  assign i5 = ~i1 & i2;
  assign i3 = i0 & ~i5;
  assign i4 = ~i0 & i1;
endmodule


