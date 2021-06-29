// Test Sample 1: xor of 3 variables 

module formula ( i_0,i_1, i_2, i_3,out);
input i_0, i_1, i_2, i_3;
output out;
wire w1, w2;
assign w1 = i_0 ^ i_1;
assign w2 = i_2 ^ i_3;
assign out = w1 ^ w2;
endmodule