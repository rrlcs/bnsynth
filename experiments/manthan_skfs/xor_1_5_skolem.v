// Benchmark "SKOLEMFORMULA" written by ABC on Wed May 18 01:46:26 2022

module SKOLEMFORMULA ( 
    i0,
    i1, i2, i3, i4, i5  );
  input  i0;
  output i1, i2, i3, i4, i5;
  assign i1 = 1'b1;
  assign i5 = 1'b1;
  assign i3 = ~i0;
  assign i2 = i0;
  assign i4 = i0;
endmodule


