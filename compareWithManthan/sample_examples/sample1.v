// Test Sample 1: xor of 3 variables 

module formula ( i_0,i_1,i_2,out);
input i_0, i_1, i_2;
output out;
wire w1;
assign out = w1 ^ i_2;
assign w1 = i_0 ^ i_1;

endmodule
