def F(XY_vars, util):
	v_1 = XY_vars[0, :]
	v_2 = XY_vars[1, :]
	v_3 = XY_vars[2, :]
	v_4 = XY_vars[3, :]
	v_5 = XY_vars[4, :]
	v_6 = XY_vars[5, :]
	v_7 = XY_vars[6, :]
	v_8 = XY_vars[7, :]
	v_9 = XY_vars[8, :]
	v_10 = XY_vars[9, :]
	v_11 = XY_vars[10, :]
	v_12 = XY_vars[11, :]
	v_13 = XY_vars[12, :]
	v_14 = XY_vars[13, :]
	v_15 = XY_vars[14, :]
	v_16 = XY_vars[15, :]
	v_17 = XY_vars[16, :]
	v_18 = XY_vars[17, :]
	v_19 = XY_vars[18, :]
	v_20 = XY_vars[19, :]
	v_21 = XY_vars[20, :]
	v_22 = XY_vars[21, :]
	v_23 = XY_vars[22, :]
	v_24 = XY_vars[23, :]
	v_25 = XY_vars[24, :]
	v_26 = XY_vars[25, :]
	v_27 = XY_vars[26, :]
	v_28 = XY_vars[27, :]
	v_29 = XY_vars[28, :]
	v_30 = XY_vars[29, :]
	v_31 = XY_vars[30, :]
	v_32 = XY_vars[31, :]
	v_33 = XY_vars[32, :]
	v_34 = XY_vars[33, :]
	v_35 = XY_vars[34, :]
	v_36 = XY_vars[35, :]
	v_37 = XY_vars[36, :]
	v_38 = XY_vars[37, :]
	v_39 = XY_vars[38, :]
	v_40 = XY_vars[39, :]
	v_41 = XY_vars[40, :]
	v_42 = XY_vars[41, :]
	v_43 = XY_vars[42, :]
	v_44 = XY_vars[43, :]
	v_45 = XY_vars[44, :]
	v_46 = XY_vars[45, :]
	v_47 = XY_vars[46, :]
	v_48 = XY_vars[47, :]
	v_49 = XY_vars[48, :]
	v_50 = XY_vars[49, :]
	v_51 = XY_vars[50, :]
	v_52 = XY_vars[51, :]
	v_53 = XY_vars[52, :]
	v_54 = XY_vars[53, :]
	v_55 = XY_vars[54, :]
	v_56 = XY_vars[55, :]
	v_261 = v_44
	v_241 = v_41
	v_217 = v_36
	v_197 = v_33
	v_169 = v_20
	v_149 = v_17
	v_125 = v_12
	v_105 = v_9
	v_81 = v_4
	v_61 = v_1
	v_263 = util.tnorm_vectorized((v_45),(v_261))
	v_243 = util.tnorm_vectorized((v_42),(v_241))
	v_219 = util.tnorm_vectorized((v_37),(v_217))
	v_199 = util.tnorm_vectorized((v_34),(v_197))
	v_171 = util.tnorm_vectorized((v_21),(v_169))
	v_151 = util.tnorm_vectorized((v_18),(v_149))
	v_127 = util.tnorm_vectorized((v_13),(v_125))
	v_107 = util.tnorm_vectorized((v_10),(v_105))
	v_83 = util.tnorm_vectorized((v_5),(v_81))
	v_63 = util.tnorm_vectorized((v_2),(v_61))
	v_264 = v_263
	v_244 = v_243
	v_220 = v_219
	v_200 = v_199
	v_172 = v_171
	v_152 = v_151
	v_128 = v_127
	v_108 = v_107
	v_84 = v_83
	v_64 = v_63
	v_265 = util.continuous_xor((v_264),(v_46))
	v_240 = util.continuous_xor((v_48),(v_47))
	v_262 = util.continuous_xor((v_261),(v_45))
	v_245 = util.continuous_xor((v_244),(v_43))
	v_242 = util.continuous_xor((v_241),(v_42))
	v_221 = util.continuous_xor((v_220),(v_38))
	v_196 = util.continuous_xor((v_40),(v_39))
	v_218 = util.continuous_xor((v_217),(v_37))
	v_201 = util.continuous_xor((v_200),(v_35))
	v_198 = util.continuous_xor((v_197),(v_34))
	v_173 = util.continuous_xor((v_172),(v_22))
	v_148 = util.continuous_xor((v_24),(v_23))
	v_170 = util.continuous_xor((v_169),(v_21))
	v_153 = util.continuous_xor((v_152),(v_19))
	v_150 = util.continuous_xor((v_149),(v_18))
	v_129 = util.continuous_xor((v_128),(v_14))
	v_104 = util.continuous_xor((v_16),(v_15))
	v_126 = util.continuous_xor((v_125),(v_13))
	v_109 = util.continuous_xor((v_108),(v_11))
	v_106 = util.continuous_xor((v_105),(v_10))
	v_85 = util.continuous_xor((v_84),(v_6))
	v_60 = util.continuous_xor((v_8),(v_7))
	v_82 = util.continuous_xor((v_81),(v_5))
	v_65 = util.continuous_xor((v_64),(v_3))
	v_62 = util.continuous_xor((v_61),(v_2))
	v_275 = util.tnorm_vectorized((v_240),(v_265))
	v_274 = util.tnorm_vectorized(util.negation(v_240),(v_46))
	v_272 = util.tnorm_vectorized((v_240),(v_262))
	v_271 = util.tnorm_vectorized(util.negation(v_240),(v_45))
	v_269 = util.tnorm_vectorized(util.negation(v_44),(v_240))
	v_268 = util.tnorm_vectorized(util.negation(v_240),(v_44))
	v_255 = util.tnorm_vectorized((v_240),(v_43))
	v_254 = util.tnorm_vectorized(util.negation(v_240),(v_245))
	v_252 = util.tnorm_vectorized((v_240),(v_42))
	v_251 = util.tnorm_vectorized(util.negation(v_240),(v_242))
	v_249 = util.tnorm_vectorized((v_240),(v_41))
	v_248 = util.tnorm_vectorized(util.negation(v_240),util.negation(v_41))
	v_231 = util.tnorm_vectorized((v_196),(v_221))
	v_230 = util.tnorm_vectorized(util.negation(v_196),(v_38))
	v_228 = util.tnorm_vectorized((v_196),(v_218))
	v_227 = util.tnorm_vectorized(util.negation(v_196),(v_37))
	v_225 = util.tnorm_vectorized(util.negation(v_36),(v_196))
	v_224 = util.tnorm_vectorized(util.negation(v_196),(v_36))
	v_211 = util.tnorm_vectorized((v_196),(v_35))
	v_210 = util.tnorm_vectorized(util.negation(v_196),(v_201))
	v_208 = util.tnorm_vectorized((v_196),(v_34))
	v_207 = util.tnorm_vectorized(util.negation(v_196),(v_198))
	v_205 = util.tnorm_vectorized((v_196),(v_33))
	v_204 = util.tnorm_vectorized(util.negation(v_196),util.negation(v_33))
	v_183 = util.tnorm_vectorized((v_148),(v_173))
	v_182 = util.tnorm_vectorized(util.negation(v_148),(v_22))
	v_180 = util.tnorm_vectorized((v_148),(v_170))
	v_179 = util.tnorm_vectorized(util.negation(v_148),(v_21))
	v_177 = util.tnorm_vectorized(util.negation(v_20),(v_148))
	v_176 = util.tnorm_vectorized(util.negation(v_148),(v_20))
	v_163 = util.tnorm_vectorized((v_148),(v_19))
	v_162 = util.tnorm_vectorized(util.negation(v_148),(v_153))
	v_160 = util.tnorm_vectorized((v_148),(v_18))
	v_159 = util.tnorm_vectorized(util.negation(v_148),(v_150))
	v_157 = util.tnorm_vectorized((v_148),(v_17))
	v_156 = util.tnorm_vectorized(util.negation(v_148),util.negation(v_17))
	v_139 = util.tnorm_vectorized((v_104),(v_129))
	v_138 = util.tnorm_vectorized(util.negation(v_104),(v_14))
	v_136 = util.tnorm_vectorized((v_104),(v_126))
	v_135 = util.tnorm_vectorized(util.negation(v_104),(v_13))
	v_133 = util.tnorm_vectorized(util.negation(v_12),(v_104))
	v_132 = util.tnorm_vectorized(util.negation(v_104),(v_12))
	v_119 = util.tnorm_vectorized((v_104),(v_11))
	v_118 = util.tnorm_vectorized(util.negation(v_104),(v_109))
	v_116 = util.tnorm_vectorized((v_104),(v_10))
	v_115 = util.tnorm_vectorized(util.negation(v_104),(v_106))
	v_113 = util.tnorm_vectorized((v_104),(v_9))
	v_112 = util.tnorm_vectorized(util.negation(v_104),util.negation(v_9))
	v_95 = util.tnorm_vectorized((v_60),(v_85))
	v_94 = util.tnorm_vectorized(util.negation(v_60),(v_6))
	v_92 = util.tnorm_vectorized((v_60),(v_82))
	v_91 = util.tnorm_vectorized(util.negation(v_60),(v_5))
	v_89 = util.tnorm_vectorized(util.negation(v_4),(v_60))
	v_88 = util.tnorm_vectorized(util.negation(v_60),(v_4))
	v_75 = util.tnorm_vectorized((v_60),(v_3))
	v_74 = util.tnorm_vectorized(util.negation(v_60),(v_65))
	v_72 = util.tnorm_vectorized((v_60),(v_2))
	v_71 = util.tnorm_vectorized(util.negation(v_60),(v_62))
	v_69 = util.tnorm_vectorized((v_60),(v_1))
	v_68 = util.tnorm_vectorized(util.negation(v_60),util.negation(v_1))
	v_276 = util.tconorm_vectorized((v_274),(v_275))
	v_273 = util.tconorm_vectorized((v_271),(v_272))
	v_270 = util.tconorm_vectorized((v_268),(v_269))
	v_256 = util.tconorm_vectorized((v_254),(v_255))
	v_253 = util.tconorm_vectorized((v_251),(v_252))
	v_250 = util.tconorm_vectorized((v_248),(v_249))
	v_232 = util.tconorm_vectorized((v_230),(v_231))
	v_229 = util.tconorm_vectorized((v_227),(v_228))
	v_226 = util.tconorm_vectorized((v_224),(v_225))
	v_212 = util.tconorm_vectorized((v_210),(v_211))
	v_209 = util.tconorm_vectorized((v_207),(v_208))
	v_206 = util.tconorm_vectorized((v_204),(v_205))
	v_184 = util.tconorm_vectorized((v_182),(v_183))
	v_181 = util.tconorm_vectorized((v_179),(v_180))
	v_178 = util.tconorm_vectorized((v_176),(v_177))
	v_164 = util.tconorm_vectorized((v_162),(v_163))
	v_161 = util.tconorm_vectorized((v_159),(v_160))
	v_158 = util.tconorm_vectorized((v_156),(v_157))
	v_140 = util.tconorm_vectorized((v_138),(v_139))
	v_137 = util.tconorm_vectorized((v_135),(v_136))
	v_134 = util.tconorm_vectorized((v_132),(v_133))
	v_120 = util.tconorm_vectorized((v_118),(v_119))
	v_117 = util.tconorm_vectorized((v_115),(v_116))
	v_114 = util.tconorm_vectorized((v_112),(v_113))
	v_96 = util.tconorm_vectorized((v_94),(v_95))
	v_93 = util.tconorm_vectorized((v_91),(v_92))
	v_90 = util.tconorm_vectorized((v_88),(v_89))
	v_76 = util.tconorm_vectorized((v_74),(v_75))
	v_73 = util.tconorm_vectorized((v_71),(v_72))
	v_70 = util.tconorm_vectorized((v_68),(v_69))
	v_313 = util.continuous_xor((v_54),(v_30))
	v_312 = util.continuous_xor((v_53),(v_29))
	v_311 = util.continuous_xor((v_52),(v_28))
	v_309 = util.continuous_xor((v_51),(v_27))
	v_308 = util.continuous_xor((v_50),(v_26))
	v_307 = util.continuous_xor((v_49),(v_25))
	v_302 = util.continuous_xor((v_46),(v_30))
	v_301 = util.continuous_xor((v_45),(v_29))
	v_300 = util.continuous_xor((v_44),(v_28))
	v_298 = util.continuous_xor((v_43),(v_27))
	v_297 = util.continuous_xor((v_42),(v_26))
	v_296 = util.continuous_xor((v_41),(v_25))
	v_291 = util.continuous_xor((v_38),(v_30))
	v_290 = util.continuous_xor((v_37),(v_29))
	v_289 = util.continuous_xor((v_36),(v_28))
	v_287 = util.continuous_xor((v_35),(v_27))
	v_286 = util.continuous_xor((v_34),(v_26))
	v_285 = util.continuous_xor((v_33),(v_25))
	v_279 = util.continuous_xor((v_276),(v_54))
	v_278 = util.continuous_xor((v_273),(v_53))
	v_277 = util.continuous_xor((v_270),(v_52))
	v_259 = util.continuous_xor((v_256),(v_51))
	v_258 = util.continuous_xor((v_253),(v_50))
	v_257 = util.continuous_xor((v_250),(v_49))
	v_235 = util.continuous_xor((v_232),(v_46))
	v_234 = util.continuous_xor((v_229),(v_45))
	v_233 = util.continuous_xor((v_226),(v_44))
	v_215 = util.continuous_xor((v_212),(v_43))
	v_214 = util.continuous_xor((v_209),(v_42))
	v_213 = util.continuous_xor((v_206),(v_41))
	v_187 = util.continuous_xor((v_184),(v_30))
	v_186 = util.continuous_xor((v_181),(v_29))
	v_185 = util.continuous_xor((v_178),(v_28))
	v_167 = util.continuous_xor((v_164),(v_27))
	v_166 = util.continuous_xor((v_161),(v_26))
	v_165 = util.continuous_xor((v_158),(v_25))
	v_143 = util.continuous_xor((v_140),(v_22))
	v_142 = util.continuous_xor((v_137),(v_21))
	v_141 = util.continuous_xor((v_134),(v_20))
	v_123 = util.continuous_xor((v_120),(v_19))
	v_122 = util.continuous_xor((v_117),(v_18))
	v_121 = util.continuous_xor((v_114),(v_17))
	v_99 = util.continuous_xor((v_96),(v_14))
	v_98 = util.continuous_xor((v_93),(v_13))
	v_97 = util.continuous_xor((v_90),(v_12))
	v_79 = util.continuous_xor((v_76),(v_11))
	v_78 = util.continuous_xor((v_73),(v_10))
	v_77 = util.continuous_xor((v_70),(v_9))
	v_314 = util.tnorm_vectorized((util.tnorm_vectorized(util.negation(v_311),util.negation(v_312))),util.negation(v_313))
	v_310 = util.tnorm_vectorized((util.tnorm_vectorized(util.negation(v_307),util.negation(v_308))),util.negation(v_309))
	v_316 = util.continuous_xor((v_56),(v_32))
	v_315 = util.continuous_xor((v_55),(v_31))
	v_303 = util.tnorm_vectorized((util.tnorm_vectorized(util.negation(v_300),util.negation(v_301))),util.negation(v_302))
	v_299 = util.tnorm_vectorized((util.tnorm_vectorized(util.negation(v_296),util.negation(v_297))),util.negation(v_298))
	v_305 = util.continuous_xor((v_48),(v_32))
	v_304 = util.continuous_xor((v_47),(v_31))
	v_292 = util.tnorm_vectorized((util.tnorm_vectorized(util.negation(v_289),util.negation(v_290))),util.negation(v_291))
	v_288 = util.tnorm_vectorized((util.tnorm_vectorized(util.negation(v_285),util.negation(v_286))),util.negation(v_287))
	v_294 = util.continuous_xor((v_40),(v_32))
	v_293 = util.continuous_xor((v_39),(v_31))
	v_280 = util.tnorm_vectorized((util.tnorm_vectorized(util.negation(v_277),util.negation(v_278))),util.negation(v_279))
	v_260 = util.tnorm_vectorized((util.tnorm_vectorized(util.negation(v_257),util.negation(v_258))),util.negation(v_259))
	v_282 = util.continuous_xor((v_47),(v_56))
	v_281 = util.continuous_xor(util.negation(v_55),(v_48))
	v_236 = util.tnorm_vectorized((util.tnorm_vectorized(util.negation(v_233),util.negation(v_234))),util.negation(v_235))
	v_216 = util.tnorm_vectorized((util.tnorm_vectorized(util.negation(v_213),util.negation(v_214))),util.negation(v_215))
	v_238 = util.continuous_xor((v_39),(v_48))
	v_237 = util.continuous_xor(util.negation(v_47),(v_40))
	v_194 = util.tnorm_vectorized((util.tnorm_vectorized(util.negation(v_36),util.negation(v_37))),util.negation(v_38))
	v_193 = util.tnorm_vectorized((util.tnorm_vectorized(util.negation(v_33),util.negation(v_34))),util.negation(v_35))
	v_188 = util.tnorm_vectorized((util.tnorm_vectorized(util.negation(v_185),util.negation(v_186))),util.negation(v_187))
	v_168 = util.tnorm_vectorized((util.tnorm_vectorized(util.negation(v_165),util.negation(v_166))),util.negation(v_167))
	v_190 = util.continuous_xor((v_23),(v_32))
	v_189 = util.continuous_xor(util.negation(v_31),(v_24))
	v_144 = util.tnorm_vectorized((util.tnorm_vectorized(util.negation(v_141),util.negation(v_142))),util.negation(v_143))
	v_124 = util.tnorm_vectorized((util.tnorm_vectorized(util.negation(v_121),util.negation(v_122))),util.negation(v_123))
	v_146 = util.continuous_xor((v_15),(v_24))
	v_145 = util.continuous_xor(util.negation(v_23),(v_16))
	v_100 = util.tnorm_vectorized((util.tnorm_vectorized(util.negation(v_97),util.negation(v_98))),util.negation(v_99))
	v_80 = util.tnorm_vectorized((util.tnorm_vectorized(util.negation(v_77),util.negation(v_78))),util.negation(v_79))
	v_102 = util.continuous_xor((v_7),(v_16))
	v_101 = util.continuous_xor(util.negation(v_15),(v_8))
	v_58 = util.tnorm_vectorized((util.tnorm_vectorized(util.negation(v_4),util.negation(v_5))),util.negation(v_6))
	v_57 = util.tnorm_vectorized((util.tnorm_vectorized(util.negation(v_1),util.negation(v_2))),util.negation(v_3))
	v_317 = util.tnorm_vectorized((util.tnorm_vectorized((util.tnorm_vectorized(util.negation(v_315),util.negation(v_316))),(v_310))),(v_314))
	v_306 = util.tnorm_vectorized((util.tnorm_vectorized((util.tnorm_vectorized(util.negation(v_304),util.negation(v_305))),(v_299))),(v_303))
	v_295 = util.tnorm_vectorized((util.tnorm_vectorized((util.tnorm_vectorized(util.negation(v_293),util.negation(v_294))),(v_288))),(v_292))
	v_283 = util.tnorm_vectorized((util.tnorm_vectorized((util.tnorm_vectorized(util.negation(v_281),util.negation(v_282))),(v_260))),(v_280))
	v_239 = util.tnorm_vectorized((util.tnorm_vectorized((util.tnorm_vectorized(util.negation(v_237),util.negation(v_238))),(v_216))),(v_236))
	v_195 = util.tnorm_vectorized((util.tnorm_vectorized((util.tnorm_vectorized(util.negation(v_39),util.negation(v_40))),(v_193))),(v_194))
	v_191 = util.tnorm_vectorized((util.tnorm_vectorized((util.tnorm_vectorized(util.negation(v_189),util.negation(v_190))),(v_168))),(v_188))
	v_147 = util.tnorm_vectorized((util.tnorm_vectorized((util.tnorm_vectorized(util.negation(v_145),util.negation(v_146))),(v_124))),(v_144))
	v_103 = util.tnorm_vectorized((util.tnorm_vectorized((util.tnorm_vectorized(util.negation(v_101),util.negation(v_102))),(v_80))),(v_100))
	v_59 = util.tnorm_vectorized((util.tnorm_vectorized((util.tnorm_vectorized(util.negation(v_7),util.negation(v_8))),(v_57))),(v_58))
	v_318 = util.tconorm_vectorized((util.tconorm_vectorized((v_295),(v_306))),(v_317))
	v_284 = util.tnorm_vectorized((util.tnorm_vectorized((v_195),(v_239))),(v_283))
	v_192 = util.tnorm_vectorized((util.tnorm_vectorized((util.tnorm_vectorized((v_59),(v_103))),(v_147))),(v_191))
	v_319 = util.tnorm_vectorized((v_284),(v_318))
	v_266 = util.tnorm_vectorized((v_46),(v_264))
	v_246 = util.tnorm_vectorized((v_43),(v_244))
	v_222 = util.tnorm_vectorized((v_38),(v_220))
	v_202 = util.tnorm_vectorized((v_35),(v_200))
	v_174 = util.tnorm_vectorized((v_22),(v_172))
	v_154 = util.tnorm_vectorized((v_19),(v_152))
	v_130 = util.tnorm_vectorized((v_14),(v_128))
	v_110 = util.tnorm_vectorized((v_11),(v_108))
	v_86 = util.tnorm_vectorized((v_6),(v_84))
	v_66 = util.tnorm_vectorized((v_3),(v_64))
	x_1 = util.tconorm_vectorized((v_319),util.negation(v_192))
	v_267 = v_266
	v_247 = v_246
	v_223 = v_222
	v_203 = v_202
	v_175 = v_174
	v_155 = v_154
	v_131 = v_130
	v_111 = v_110
	v_87 = v_86
	v_67 = v_66
	o_1 = x_1
	return o_1