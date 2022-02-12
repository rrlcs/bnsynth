// Generated using findDep.cpp 
module ex7 (v_4, v_1, v_3, o_1);
input v_4;
input v_1;
input v_3;
output o_1;
wire x_1;
wire x_2;
wire x_3;
wire x_4;
wire x_5;
wire x_6;
wire x_7;
wire x_8;
wire x_9;
assign v_5 = 1;
assign x_1 = v_2 | v_3 | ~v_4;
assign x_2 = ~v_2 | ~v_3;
assign x_3 = ~v_2 | v_4;
assign x_4 = v_1 | v_2;
assign x_5 = ~v_1 | ~v_2;
assign x_6 = x_1 & x_2;
assign x_7 = x_4 & x_5;
assign x_8 = x_3 & x_7;
assign x_9 = x_6 & x_8;
assign o_1 = x_9;
endmodule
