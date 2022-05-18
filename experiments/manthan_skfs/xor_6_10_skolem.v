// Benchmark "SKOLEMFORMULA" written by ABC on Wed May 18 00:55:59 2022

module SKOLEMFORMULA ( 
    i0, i1, i2, i3, i4, i5,
    i6, i7, i8, i9, i10, i11, i12, i13, i14, i15  );
  input  i0, i1, i2, i3, i4, i5;
  output i6, i7, i8, i9, i10, i11, i12, i13, i14, i15;
  wire n18, n19, n20, n21, n22, n23, n24, n25, n26, n27, n28, n29, n30, n31,
    n32, n33, n34, n35, n36, n37, n38, n39, n40, n41, n42, n43, n44, n45,
    n46, n47, n48, n49, n50, n52, n53, n54, n55, n56, n57, n58, n59, n60,
    n61, n62, n63, n64, n65, n66, n67, n68, n69, n70, n71, n72, n73, n74,
    n75, n76, n77, n78, n80, n81, n82, n83, n84, n85, n86, n87, n88, n89,
    n90, n91, n92, n93, n94, n95, n96, n97, n98, n99, n100, n101, n102,
    n103, n104, n105, n106, n108, n109, n110, n111, n112, n113, n114, n115,
    n116, n117, n118, n119, n120, n121, n122, n123, n124, n125, n126, n127,
    n128, n129, n130, n131, n132, n133, n134, n135, n136, n137, n138, n139,
    n140, n141, n142, n143, n144, n145, n146, n147, n148, n149, n150, n151,
    n152, n153, n154, n155, n156, n157, n158, n159, n160, n161, n162, n163,
    n164, n166, n167, n168, n169, n170, n171, n172, n173, n174, n175, n176,
    n177, n178, n179, n180, n181, n182, n183, n184, n185, n186, n187, n188,
    n189, n190, n191, n192, n193, n194, n195, n196, n197, n198, n199, n200,
    n201, n202, n203, n204, n205, n207, n208, n209, n210, n211, n212, n213,
    n214, n215, n216, n217, n218, n219, n220, n221, n222, n223, n224, n225,
    n226, n227, n228, n229, n230, n231, n232, n233, n234, n235, n236, n237,
    n238, n239, n240, n241, n242, n243, n244, n245, n246, n247, n249, n250,
    n251, n252, n253, n254, n255, n256, n257, n258, n259, n260, n261, n262,
    n263, n264, n265, n266, n267, n268, n269, n271, n272, n273, n274, n275,
    n276, n277, n278, n279, n280, n281, n282, n283, n284, n285, n286, n287,
    n288, n289, n290, n291, n292, n293, n294, n295, n296, n297, n298, n299,
    n300, n301, n302, n303, n304, n305;
  assign n18 = i3 & ~i4;
  assign n19 = ~i3 & ~i4;
  assign n20 = ~i1 & n19;
  assign n21 = ~i5 & n20;
  assign n22 = i5 & n20;
  assign n23 = i0 & n22;
  assign n24 = ~n21 & ~n23;
  assign n25 = i1 & n19;
  assign n26 = ~i2 & n25;
  assign n27 = n24 & ~n26;
  assign n28 = i2 & n25;
  assign n29 = ~i0 & n28;
  assign n30 = i5 & n29;
  assign n31 = n27 & ~n30;
  assign n32 = i0 & n28;
  assign n33 = n31 & ~n32;
  assign n34 = ~n18 & n33;
  assign n35 = ~i2 & i4;
  assign n36 = ~i5 & n35;
  assign n37 = i0 & n36;
  assign n38 = i3 & n37;
  assign n39 = n34 & ~n38;
  assign n40 = i5 & n35;
  assign n41 = n39 & ~n40;
  assign n42 = i2 & i4;
  assign n43 = ~i5 & n42;
  assign n44 = ~i0 & n43;
  assign n45 = ~i1 & n44;
  assign n46 = n41 & ~n45;
  assign n47 = i0 & n43;
  assign n48 = n46 & ~n47;
  assign n49 = i5 & n42;
  assign n50 = ~i0 & n49;
  assign i14 = ~n48 | n50;
  assign n52 = ~i2 & ~i4;
  assign n53 = ~i5 & n52;
  assign n54 = i14 & n53;
  assign n55 = ~i14 & n53;
  assign n56 = i0 & n55;
  assign n57 = ~n54 & ~n56;
  assign n58 = i5 & n52;
  assign n59 = ~i14 & n58;
  assign n60 = n57 & ~n59;
  assign n61 = i14 & n58;
  assign n62 = i1 & n61;
  assign n63 = n60 & ~n62;
  assign n64 = i2 & ~i4;
  assign n65 = ~i14 & n64;
  assign n66 = i3 & n65;
  assign n67 = n63 & ~n66;
  assign n68 = i14 & n64;
  assign n69 = ~i0 & n68;
  assign n70 = ~i5 & n69;
  assign n71 = n67 & ~n70;
  assign n72 = i5 & n69;
  assign n73 = ~i3 & n72;
  assign n74 = n71 & ~n73;
  assign n75 = i0 & n68;
  assign n76 = i5 & n75;
  assign n77 = ~i1 & n76;
  assign n78 = n74 & ~n77;
  assign i13 = i4 | ~n78;
  assign n80 = ~i0 & ~i4;
  assign n81 = ~i14 & n80;
  assign n82 = ~i13 & n81;
  assign n83 = i13 & n81;
  assign n84 = i3 & n83;
  assign n85 = ~n82 & ~n84;
  assign n86 = i14 & n80;
  assign n87 = ~i1 & n86;
  assign n88 = i2 & n87;
  assign n89 = ~i3 & n88;
  assign n90 = n85 & ~n89;
  assign n91 = i1 & n86;
  assign n92 = ~i2 & n91;
  assign n93 = i13 & n92;
  assign n94 = n90 & ~n93;
  assign n95 = i2 & n91;
  assign n96 = ~i13 & n95;
  assign n97 = n94 & ~n96;
  assign n98 = i0 & ~i4;
  assign n99 = ~i2 & n98;
  assign n100 = n97 & ~n99;
  assign n101 = i2 & n98;
  assign n102 = ~i1 & n101;
  assign n103 = i14 & n102;
  assign n104 = n100 & ~n103;
  assign n105 = ~i0 & n35;
  assign n106 = n104 & ~n105;
  assign i12 = n42 | ~n106;
  assign n108 = ~i1 & ~i14;
  assign n109 = ~i1 & i14;
  assign n110 = ~i5 & n109;
  assign n111 = ~i12 & n110;
  assign n112 = ~i2 & n111;
  assign n113 = ~i4 & n112;
  assign n114 = ~n108 & ~n113;
  assign n115 = i12 & n110;
  assign n116 = ~i2 & n115;
  assign n117 = i0 & n116;
  assign n118 = n114 & ~n117;
  assign n119 = i2 & n115;
  assign n120 = ~i13 & n119;
  assign n121 = ~i3 & n120;
  assign n122 = n118 & ~n121;
  assign n123 = i3 & n120;
  assign n124 = ~i0 & n123;
  assign n125 = n122 & ~n124;
  assign n126 = i13 & n119;
  assign n127 = n125 & ~n126;
  assign n128 = i5 & n109;
  assign n129 = ~i2 & n128;
  assign n130 = n127 & ~n129;
  assign n131 = i2 & n128;
  assign n132 = ~i4 & n131;
  assign n133 = n130 & ~n132;
  assign n134 = i1 & ~i3;
  assign n135 = ~i5 & n134;
  assign n136 = ~i12 & n135;
  assign n137 = ~i13 & n136;
  assign n138 = n133 & ~n137;
  assign n139 = i12 & n135;
  assign n140 = i4 & n139;
  assign n141 = ~i2 & n140;
  assign n142 = n138 & ~n141;
  assign n143 = i5 & n134;
  assign n144 = i12 & n143;
  assign n145 = ~i2 & n144;
  assign n146 = n142 & ~n145;
  assign n147 = i2 & n144;
  assign n148 = ~i14 & n147;
  assign n149 = n146 & ~n148;
  assign n150 = i1 & i3;
  assign n151 = ~i4 & n150;
  assign n152 = ~i0 & n151;
  assign n153 = n149 & ~n152;
  assign n154 = i0 & n151;
  assign n155 = i13 & n154;
  assign n156 = i5 & n155;
  assign n157 = n153 & ~n156;
  assign n158 = i4 & n150;
  assign n159 = i13 & n158;
  assign n160 = ~i5 & n159;
  assign n161 = ~i14 & n160;
  assign n162 = n157 & ~n161;
  assign n163 = i14 & n160;
  assign n164 = i2 & n163;
  assign i11 = ~n162 | n164;
  assign n166 = n52 & ~i13;
  assign n167 = n52 & i13;
  assign n168 = i14 & n167;
  assign n169 = ~i0 & n168;
  assign n170 = ~n166 & ~n169;
  assign n171 = i0 & n168;
  assign n172 = i1 & n171;
  assign n173 = ~i3 & n172;
  assign n174 = ~i11 & n173;
  assign n175 = n170 & ~n174;
  assign n176 = i3 & n172;
  assign n177 = n175 & ~n176;
  assign n178 = n35 & ~i14;
  assign n179 = ~i3 & n178;
  assign n180 = n177 & ~n179;
  assign n181 = i3 & n178;
  assign n182 = ~i13 & n181;
  assign n183 = n180 & ~n182;
  assign n184 = n35 & i14;
  assign n185 = ~i11 & n184;
  assign n186 = ~i1 & n185;
  assign n187 = n183 & ~n186;
  assign n188 = i11 & n184;
  assign n189 = ~i0 & n188;
  assign n190 = n187 & ~n189;
  assign n191 = ~i0 & i2;
  assign n192 = ~i3 & n191;
  assign n193 = i12 & n192;
  assign n194 = ~i4 & n193;
  assign n195 = i5 & n194;
  assign n196 = n190 & ~n195;
  assign n197 = i4 & n193;
  assign n198 = n196 & ~n197;
  assign n199 = i3 & n191;
  assign n200 = ~i13 & n199;
  assign n201 = i4 & n200;
  assign n202 = n198 & ~n201;
  assign n203 = i13 & n199;
  assign n204 = n202 & ~n203;
  assign n205 = i0 & i2;
  assign i10 = ~n204 | n205;
  assign n207 = ~i3 & ~i5;
  assign n208 = i2 & n207;
  assign n209 = ~i10 & n208;
  assign n210 = i10 & n208;
  assign n211 = i4 & n210;
  assign n212 = i0 & n211;
  assign n213 = ~n209 & ~n212;
  assign n214 = i3 & ~i5;
  assign n215 = ~i14 & n214;
  assign n216 = ~i0 & n215;
  assign n217 = n213 & ~n216;
  assign n218 = i0 & n215;
  assign n219 = i10 & n218;
  assign n220 = n217 & ~n219;
  assign n221 = i14 & n214;
  assign n222 = ~i1 & n221;
  assign n223 = ~i4 & n222;
  assign n224 = ~i2 & n223;
  assign n225 = n220 & ~n224;
  assign n226 = i2 & n223;
  assign n227 = ~i13 & n226;
  assign n228 = n225 & ~n227;
  assign n229 = i1 & n221;
  assign n230 = n228 & ~n229;
  assign n231 = i5 & ~i10;
  assign n232 = n230 & ~n231;
  assign n233 = i5 & i10;
  assign n234 = ~i2 & n233;
  assign n235 = i12 & n234;
  assign n236 = ~i1 & n235;
  assign n237 = n232 & ~n236;
  assign n238 = i1 & n235;
  assign n239 = n237 & ~n238;
  assign n240 = i2 & n233;
  assign n241 = ~i3 & n240;
  assign n242 = ~i0 & n241;
  assign n243 = n239 & ~n242;
  assign n244 = i0 & n241;
  assign n245 = ~i12 & n244;
  assign n246 = n243 & ~n245;
  assign n247 = i3 & n240;
  assign i9 = ~n246 | n247;
  assign n249 = ~i1 & ~i4;
  assign n250 = ~i2 & n249;
  assign n251 = ~i10 & n250;
  assign n252 = i10 & n250;
  assign n253 = i3 & n252;
  assign n254 = i14 & n253;
  assign n255 = ~n251 & ~n254;
  assign n256 = i2 & n249;
  assign n257 = i10 & n256;
  assign n258 = n255 & ~n257;
  assign n259 = i1 & ~i4;
  assign n260 = ~i5 & n259;
  assign n261 = ~i9 & n260;
  assign n262 = n258 & ~n261;
  assign n263 = ~n35 & n262;
  assign n264 = ~i0 & n42;
  assign n265 = i12 & n264;
  assign n266 = n263 & ~n265;
  assign n267 = i0 & n42;
  assign n268 = ~i1 & n267;
  assign n269 = i9 & n268;
  assign i8 = ~n266 | n269;
  assign n271 = ~i0 & i1;
  assign n272 = i0 & ~i1;
  assign n273 = ~n271 & ~n272;
  assign n274 = i2 & n273;
  assign n275 = ~i2 & ~n273;
  assign n276 = ~n274 & ~n275;
  assign n277 = i3 & n276;
  assign n278 = ~i3 & ~n276;
  assign n279 = ~n277 & ~n278;
  assign n280 = i4 & n279;
  assign n281 = ~i4 & ~n279;
  assign n282 = ~n280 & ~n281;
  assign n283 = i5 & n282;
  assign n284 = ~i5 & ~n282;
  assign n285 = ~n283 & ~n284;
  assign n286 = i8 & n285;
  assign n287 = ~i8 & ~n285;
  assign n288 = ~n286 & ~n287;
  assign n289 = i9 & n288;
  assign n290 = ~i9 & ~n288;
  assign n291 = ~n289 & ~n290;
  assign n292 = i10 & n291;
  assign n293 = ~i10 & ~n291;
  assign n294 = ~n292 & ~n293;
  assign n295 = i11 & n294;
  assign n296 = ~i11 & ~n294;
  assign n297 = ~n295 & ~n296;
  assign n298 = i12 & n297;
  assign n299 = ~i12 & ~n297;
  assign n300 = ~n298 & ~n299;
  assign n301 = i13 & n300;
  assign n302 = ~i13 & ~n300;
  assign n303 = ~n301 & ~n302;
  assign n304 = i14 & n303;
  assign n305 = ~i14 & ~n303;
  assign i6 = ~n304 & ~n305;
  assign i7 = 1'b1;
  assign i15 = 1'b1;
endmodule


