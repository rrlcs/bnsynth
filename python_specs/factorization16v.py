def F(XY_vars, util):
	i2 = XY_vars[0, :]
	i2 = XY_vars[1, :]
	i2 = XY_vars[2, :]
	i2 = XY_vars[3, :]
	i2 = XY_vars[4, :]
	i2 = XY_vars[5, :]
	i2 = XY_vars[6, :]
	i2 = XY_vars[7, :]
	i1 = XY_vars[8, :]
	i1 = XY_vars[9, :]
	i1 = XY_vars[10, :]
	i1 = XY_vars[11, :]
	i1 = XY_vars[12, :]
	i1 = XY_vars[13, :]
	i1 = XY_vars[14, :]
	i1 = XY_vars[15, :]
	a = XY_vars[16, :]
	a = XY_vars[17, :]
	a = XY_vars[18, :]
	a = XY_vars[19, :]
	a = XY_vars[20, :]
	a = XY_vars[21, :]
	a = XY_vars[22, :]
	a = XY_vars[23, :]
	a = XY_vars[24, :]
	a = XY_vars[25, :]
	a = XY_vars[26, :]
	a = XY_vars[27, :]
	a = XY_vars[28, :]
	a = XY_vars[29, :]
	a = XY_vars[30, :]
	a = XY_vars[31, :]
	n_20 = (i2[12] & i1[15])
	n_26 = (i2[13] & i1[15])
	n_461 = (~i2[13] & ~i2[12])
	n_19 = (i2[13] & i1[14])
	n_16 = (i2[12] & i1[14])
	n_467 = (i1[15] & ~i1[14])
	n_25 = (i2[14] & i1[14])
	n_22 = (i2[14] & i1[15])
	n_23 = (i2[15] & i1[14])
	n_460 = (i2[15] & ~i2[14])
	n_449 = (i2[15] & i1[15])
	n_17 = (i2[11] & i1[15])
	n_13 = (i2[11] & i1[14])
	n_14 = (i2[10] & i1[15])
	n_10 = (i2[10] & i1[14])
	n_457 = (~i2[11] & ~i2[10])
	n_74 = (i2[14] & i1[13])
	n_76 = (i2[15] & i1[13])
	n_70 = (i2[13] & i1[13])
	n_66 = (i2[12] & i1[13])
	n_62 = (i2[11] & i1[13])
	n_58 = (i2[10] & i1[13])
	n_11 = (i2[9] & i1[15])
	n_7 = (i2[9] & i1[14])
	n_52 = (i2[9] & i1[13])
	n_8 = (i2[8] & i1[15])
	n_49 = (i2[8] & i1[14])
	n_6 = (i2[8] & i1[13])
	n_458 = (~i2[9] & ~i2[8])
	n_133 = (i2[14] & i1[12])
	n_137 = (i2[15] & i1[12])
	n_129 = (i2[13] & i1[12])
	n_125 = (i2[12] & i1[12])
	n_121 = (i2[11] & i1[12])
	n_115 = (i2[10] & i1[12])
	n_111 = (i2[9] & i1[12])
	n_468 = (~i1[13] & ~i1[12])
	n_5 = (i2[8] & i1[12])
	n_190 = (i2[14] & i1[11])
	n_194 = (i2[15] & i1[11])
	n_186 = (i2[13] & i1[11])
	n_182 = (i2[12] & i1[11])
	n_176 = (i2[11] & i1[11])
	n_172 = (i2[10] & i1[11])
	n_168 = (i2[9] & i1[11])
	n_4 = (i2[8] & i1[11])
	n_304 = (i2[14] & i1[9])
	n_308 = (i2[15] & i1[9])
	n_298 = (i2[13] & i1[9])
	n_294 = (i2[12] & i1[9])
	n_290 = (i2[11] & i1[9])
	n_286 = (i2[10] & i1[9])
	n_282 = (i2[9] & i1[9])
	n_2 = (i2[8] & i1[9])
	n_465 = (~i1[9] & ~i1[8])
	n_365 = (i2[15] & i1[8])
	n_359 = (i2[14] & i1[8])
	n_355 = (i2[13] & i1[8])
	n_351 = (i2[12] & i1[8])
	n_347 = (i2[11] & i1[8])
	n_343 = (i2[10] & i1[8])
	n_339 = (i2[9] & i1[8])
	n_1 = (i2[8] & i1[8])
	n_464 = (~i1[11] & ~i1[10])
	n_247 = (i2[14] & i1[10])
	n_251 = (i2[15] & i1[10])
	n_243 = (i2[13] & i1[10])
	n_237 = (i2[12] & i1[10])
	n_233 = (i2[11] & i1[10])
	n_229 = (i2[10] & i1[10])
	n_225 = (i2[9] & i1[10])
	n_3 = (i2[8] & i1[10])
	n_21 = (n_19 & n_20)
	n_29 = (~n_19 & ~n_20)
	n_27 = (n_25 & n_26)
	n_79 = (~n_25 & ~n_26)
	n_24 = (n_22 & n_23)
	n_453 = (~n_22 & ~n_23)
	n_462 = (n_460 & n_461)
	n_451 = (a[15] & n_449)
	n_450 = (~a[15] & ~n_449)
	n_18 = (n_16 & n_17)
	n_33 = (~n_16 & ~n_17)
	n_15 = (n_13 & n_14)
	n_37 = (~n_13 & ~n_14)
	n_12 = (n_10 & n_11)
	n_41 = (~n_10 & ~n_11)
	n_9 = (n_7 & n_8)
	n_45 = (~n_7 & ~n_8)
	n_459 = (n_457 & n_458)
	n_469 = (n_467 & n_468)
	n_466 = (n_464 & n_465)
	n_30 = (~n_29 & ~n_21)
	n_28 = (~n_24 & ~n_27)
	n_77 = (n_24 & n_27)
	n_454 = (~n_24 & ~n_453)
	n_452 = (~n_450 & ~n_451)
	n_34 = (~n_33 & ~n_18)
	n_38 = (~n_37 & ~n_15)
	n_42 = (~n_41 & ~n_12)
	n_46 = (~n_45 & ~n_9)
	n_463 = (n_459 & n_462)
	n_470 = (n_466 & n_469)
	n_31 = (~n_28 & n_30)
	n_72 = (n_28 & ~n_30)
	n_78 = (~n_28 & ~n_77)
	n_471 = (~a[14] & n_454)
	n_455 = (a[14] & ~n_454)
	n_32 = (~n_21 & ~n_31)
	n_73 = (~n_31 & ~n_72)
	n_80 = (~n_78 & ~n_79)
	n_472 = (~n_470 & ~n_471)
	n_456 = (~n_452 & ~n_455)
	n_35 = (~n_32 & n_34)
	n_68 = (n_32 & ~n_34)
	n_82 = (~n_73 & ~n_74)
	n_75 = (n_73 & n_74)
	n_81 = (n_76 & n_80)
	n_445 = (~n_76 & ~n_80)
	n_473 = (~n_463 & n_472)
	n_36 = (~n_18 & ~n_35)
	n_69 = (~n_35 & ~n_68)
	n_83 = (~n_75 & ~n_82)
	n_446 = (~n_81 & ~n_445)
	n_474 = (n_456 & n_473)
	n_39 = (~n_36 & n_38)
	n_64 = (n_36 & ~n_38)
	n_86 = (~n_69 & ~n_70)
	n_71 = (n_69 & n_70)
	n_84 = (n_81 & n_83)
	n_135 = (~n_81 & ~n_83)
	n_448 = (a[13] & ~n_446)
	n_447 = (~a[13] & n_446)
	n_40 = (~n_15 & ~n_39)
	n_65 = (~n_39 & ~n_64)
	n_87 = (~n_71 & ~n_86)
	n_85 = (~n_75 & ~n_84)
	n_136 = (~n_84 & ~n_135)
	n_475 = (~n_448 & n_474)
	n_43 = (~n_40 & n_42)
	n_60 = (n_40 & ~n_42)
	n_90 = (~n_65 & ~n_66)
	n_67 = (n_65 & n_66)
	n_88 = (~n_85 & n_87)
	n_131 = (n_85 & ~n_87)
	n_138 = (n_136 & n_137)
	n_441 = (~n_136 & ~n_137)
	n_476 = (~n_447 & n_475)
	n_44 = (~n_12 & ~n_43)
	n_61 = (~n_43 & ~n_60)
	n_91 = (~n_67 & ~n_90)
	n_89 = (~n_71 & ~n_88)
	n_132 = (~n_88 & ~n_131)
	n_442 = (~n_138 & ~n_441)
	n_47 = (~n_44 & n_46)
	n_56 = (n_44 & ~n_46)
	n_94 = (~n_61 & ~n_62)
	n_63 = (n_61 & n_62)
	n_92 = (~n_89 & n_91)
	n_127 = (n_89 & ~n_91)
	n_139 = (~n_132 & ~n_133)
	n_134 = (n_132 & n_133)
	n_444 = (a[12] & ~n_442)
	n_443 = (~a[12] & n_442)
	n_48 = (~n_9 & ~n_47)
	n_57 = (~n_47 & ~n_56)
	n_95 = (~n_63 & ~n_94)
	n_93 = (~n_67 & ~n_92)
	n_128 = (~n_92 & ~n_127)
	n_140 = (~n_134 & ~n_139)
	n_477 = (~n_444 & n_476)
	n_53 = (n_48 & ~n_49)
	n_50 = (~n_48 & n_49)
	n_98 = (~n_57 & ~n_58)
	n_59 = (n_57 & n_58)
	n_96 = (~n_93 & n_95)
	n_123 = (n_93 & ~n_95)
	n_143 = (~n_128 & ~n_129)
	n_130 = (n_128 & n_129)
	n_141 = (n_138 & n_140)
	n_192 = (~n_138 & ~n_140)
	n_478 = (~n_443 & n_477)
	n_54 = (~n_50 & ~n_53)
	n_106 = (~n_6 & ~n_50)
	n_51 = (n_6 & n_50)
	n_99 = (~n_59 & ~n_98)
	n_97 = (~n_63 & ~n_96)
	n_124 = (~n_96 & ~n_123)
	n_144 = (~n_130 & ~n_143)
	n_142 = (~n_134 & ~n_141)
	n_193 = (~n_141 & ~n_192)
	n_102 = (~n_52 & ~n_54)
	n_55 = (n_52 & n_54)
	n_107 = (~n_51 & ~n_106)
	n_100 = (~n_97 & n_99)
	n_119 = (n_97 & ~n_99)
	n_147 = (~n_124 & ~n_125)
	n_126 = (n_124 & n_125)
	n_145 = (~n_142 & n_144)
	n_188 = (n_142 & ~n_144)
	n_195 = (n_193 & n_194)
	n_437 = (~n_193 & ~n_194)
	n_103 = (~n_55 & ~n_102)
	n_101 = (~n_59 & ~n_100)
	n_120 = (~n_100 & ~n_119)
	n_148 = (~n_126 & ~n_147)
	n_146 = (~n_130 & ~n_145)
	n_189 = (~n_145 & ~n_188)
	n_438 = (~n_195 & ~n_437)
	n_116 = (n_101 & ~n_103)
	n_104 = (~n_101 & n_103)
	n_151 = (~n_120 & ~n_121)
	n_122 = (n_120 & n_121)
	n_149 = (~n_146 & n_148)
	n_184 = (n_146 & ~n_148)
	n_196 = (~n_189 & ~n_190)
	n_191 = (n_189 & n_190)
	n_440 = (a[11] & ~n_438)
	n_439 = (~a[11] & n_438)
	n_117 = (~n_104 & ~n_116)
	n_105 = (~n_55 & ~n_104)
	n_152 = (~n_122 & ~n_151)
	n_150 = (~n_126 & ~n_149)
	n_185 = (~n_149 & ~n_184)
	n_197 = (~n_191 & ~n_196)
	n_479 = (~n_440 & n_478)
	n_155 = (~n_115 & ~n_117)
	n_118 = (n_115 & n_117)
	n_112 = (n_105 & ~n_107)
	n_108 = (~n_105 & n_107)
	n_153 = (~n_150 & n_152)
	n_180 = (n_150 & ~n_152)
	n_200 = (~n_185 & ~n_186)
	n_187 = (n_185 & n_186)
	n_198 = (n_195 & n_197)
	n_249 = (~n_195 & ~n_197)
	n_480 = (~n_439 & n_479)
	n_156 = (~n_118 & ~n_155)
	n_113 = (~n_108 & ~n_112)
	n_109 = (~n_51 & ~n_108)
	n_154 = (~n_122 & ~n_153)
	n_181 = (~n_153 & ~n_180)
	n_201 = (~n_187 & ~n_200)
	n_199 = (~n_191 & ~n_198)
	n_250 = (~n_198 & ~n_249)
	n_159 = (~n_111 & ~n_113)
	n_114 = (n_111 & n_113)
	n_163 = (~n_5 & n_109)
	n_110 = (n_5 & ~n_109)
	n_177 = (n_154 & ~n_156)
	n_157 = (~n_154 & n_156)
	n_204 = (~n_181 & ~n_182)
	n_183 = (n_181 & n_182)
	n_202 = (~n_199 & n_201)
	n_245 = (n_199 & ~n_201)
	n_252 = (n_250 & n_251)
	n_433 = (~n_250 & ~n_251)
	n_160 = (~n_114 & ~n_159)
	n_164 = (~n_110 & ~n_163)
	n_178 = (~n_157 & ~n_177)
	n_158 = (~n_118 & ~n_157)
	n_205 = (~n_183 & ~n_204)
	n_203 = (~n_187 & ~n_202)
	n_246 = (~n_202 & ~n_245)
	n_434 = (~n_252 & ~n_433)
	n_208 = (~n_176 & ~n_178)
	n_179 = (n_176 & n_178)
	n_173 = (n_158 & ~n_160)
	n_161 = (~n_158 & n_160)
	n_206 = (~n_203 & n_205)
	n_241 = (n_203 & ~n_205)
	n_253 = (~n_246 & ~n_247)
	n_248 = (n_246 & n_247)
	n_436 = (a[10] & ~n_434)
	n_435 = (~a[10] & n_434)
	n_209 = (~n_179 & ~n_208)
	n_174 = (~n_161 & ~n_173)
	n_162 = (~n_114 & ~n_161)
	n_207 = (~n_183 & ~n_206)
	n_242 = (~n_206 & ~n_241)
	n_254 = (~n_248 & ~n_253)
	n_481 = (~n_436 & n_480)
	n_212 = (~n_172 & ~n_174)
	n_175 = (n_172 & n_174)
	n_169 = (n_162 & ~n_164)
	n_165 = (~n_162 & n_164)
	n_238 = (n_207 & ~n_209)
	n_210 = (~n_207 & n_209)
	n_257 = (~n_242 & ~n_243)
	n_244 = (n_242 & n_243)
	n_255 = (n_252 & n_254)
	n_306 = (~n_252 & ~n_254)
	n_482 = (~n_435 & n_481)
	n_213 = (~n_175 & ~n_212)
	n_170 = (~n_165 & ~n_169)
	n_166 = (~n_110 & ~n_165)
	n_239 = (~n_210 & ~n_238)
	n_211 = (~n_179 & ~n_210)
	n_258 = (~n_244 & ~n_257)
	n_256 = (~n_248 & ~n_255)
	n_307 = (~n_255 & ~n_306)
	n_216 = (~n_168 & ~n_170)
	n_171 = (n_168 & n_170)
	n_220 = (~n_4 & n_166)
	n_167 = (n_4 & ~n_166)
	n_261 = (~n_237 & ~n_239)
	n_240 = (n_237 & n_239)
	n_234 = (n_211 & ~n_213)
	n_214 = (~n_211 & n_213)
	n_259 = (~n_256 & n_258)
	n_302 = (n_256 & ~n_258)
	n_309 = (n_307 & n_308)
	n_429 = (~n_307 & ~n_308)
	n_217 = (~n_171 & ~n_216)
	n_221 = (~n_167 & ~n_220)
	n_262 = (~n_240 & ~n_261)
	n_235 = (~n_214 & ~n_234)
	n_215 = (~n_175 & ~n_214)
	n_260 = (~n_244 & ~n_259)
	n_303 = (~n_259 & ~n_302)
	n_430 = (~n_309 & ~n_429)
	n_265 = (~n_233 & ~n_235)
	n_236 = (n_233 & n_235)
	n_230 = (n_215 & ~n_217)
	n_218 = (~n_215 & n_217)
	n_299 = (n_260 & ~n_262)
	n_263 = (~n_260 & n_262)
	n_310 = (~n_303 & ~n_304)
	n_305 = (n_303 & n_304)
	n_432 = (a[9] & ~n_430)
	n_431 = (~a[9] & n_430)
	n_266 = (~n_236 & ~n_265)
	n_231 = (~n_218 & ~n_230)
	n_219 = (~n_171 & ~n_218)
	n_300 = (~n_263 & ~n_299)
	n_264 = (~n_240 & ~n_263)
	n_311 = (~n_305 & ~n_310)
	n_483 = (~n_432 & n_482)
	n_269 = (~n_229 & ~n_231)
	n_232 = (n_229 & n_231)
	n_226 = (n_219 & ~n_221)
	n_222 = (~n_219 & n_221)
	n_314 = (~n_298 & ~n_300)
	n_301 = (n_298 & n_300)
	n_295 = (n_264 & ~n_266)
	n_267 = (~n_264 & n_266)
	n_312 = (n_309 & n_311)
	n_363 = (~n_309 & ~n_311)
	n_484 = (~n_431 & n_483)
	n_270 = (~n_232 & ~n_269)
	n_227 = (~n_222 & ~n_226)
	n_223 = (~n_167 & ~n_222)
	n_315 = (~n_301 & ~n_314)
	n_296 = (~n_267 & ~n_295)
	n_268 = (~n_236 & ~n_267)
	n_313 = (~n_305 & ~n_312)
	n_364 = (~n_312 & ~n_363)
	n_273 = (~n_225 & ~n_227)
	n_228 = (n_225 & n_227)
	n_277 = (~n_3 & n_223)
	n_224 = (n_3 & ~n_223)
	n_318 = (~n_294 & ~n_296)
	n_297 = (n_294 & n_296)
	n_291 = (n_268 & ~n_270)
	n_271 = (~n_268 & n_270)
	n_360 = (n_313 & ~n_315)
	n_316 = (~n_313 & n_315)
	n_425 = (~n_364 & ~n_365)
	n_366 = (n_364 & n_365)
	n_274 = (~n_228 & ~n_273)
	n_278 = (~n_224 & ~n_277)
	n_319 = (~n_297 & ~n_318)
	n_292 = (~n_271 & ~n_291)
	n_272 = (~n_232 & ~n_271)
	n_361 = (~n_316 & ~n_360)
	n_317 = (~n_301 & ~n_316)
	n_426 = (~n_366 & ~n_425)
	n_322 = (~n_290 & ~n_292)
	n_293 = (n_290 & n_292)
	n_287 = (n_272 & ~n_274)
	n_275 = (~n_272 & n_274)
	n_367 = (~n_359 & ~n_361)
	n_362 = (n_359 & n_361)
	n_356 = (n_317 & ~n_319)
	n_320 = (~n_317 & n_319)
	n_428 = (a[8] & ~n_426)
	n_427 = (~a[8] & n_426)
	n_323 = (~n_293 & ~n_322)
	n_288 = (~n_275 & ~n_287)
	n_276 = (~n_228 & ~n_275)
	n_368 = (~n_362 & ~n_367)
	n_357 = (~n_320 & ~n_356)
	n_321 = (~n_297 & ~n_320)
	n_485 = (~n_428 & n_484)
	n_326 = (~n_286 & ~n_288)
	n_289 = (n_286 & n_288)
	n_283 = (n_276 & ~n_278)
	n_279 = (~n_276 & n_278)
	n_421 = (~n_366 & ~n_368)
	n_369 = (n_366 & n_368)
	n_371 = (~n_355 & ~n_357)
	n_358 = (n_355 & n_357)
	n_352 = (n_321 & ~n_323)
	n_324 = (~n_321 & n_323)
	n_486 = (~n_427 & n_485)
	n_327 = (~n_289 & ~n_326)
	n_284 = (~n_279 & ~n_283)
	n_280 = (~n_224 & ~n_279)
	n_422 = (~n_369 & ~n_421)
	n_370 = (~n_362 & ~n_369)
	n_372 = (~n_358 & ~n_371)
	n_353 = (~n_324 & ~n_352)
	n_325 = (~n_293 & ~n_324)
	n_330 = (~n_282 & ~n_284)
	n_285 = (n_282 & n_284)
	n_334 = (~n_2 & n_280)
	n_281 = (n_2 & ~n_280)
	n_424 = (a[7] & ~n_422)
	n_423 = (~a[7] & n_422)
	n_417 = (n_370 & ~n_372)
	n_373 = (~n_370 & n_372)
	n_375 = (~n_351 & ~n_353)
	n_354 = (n_351 & n_353)
	n_348 = (n_325 & ~n_327)
	n_328 = (~n_325 & n_327)
	n_331 = (~n_285 & ~n_330)
	n_335 = (~n_281 & ~n_334)
	n_487 = (~n_424 & n_486)
	n_418 = (~n_373 & ~n_417)
	n_374 = (~n_358 & ~n_373)
	n_376 = (~n_354 & ~n_375)
	n_349 = (~n_328 & ~n_348)
	n_329 = (~n_289 & ~n_328)
	n_488 = (~n_423 & n_487)
	n_420 = (a[6] & ~n_418)
	n_419 = (~a[6] & n_418)
	n_413 = (n_374 & ~n_376)
	n_377 = (~n_374 & n_376)
	n_379 = (~n_347 & ~n_349)
	n_350 = (n_347 & n_349)
	n_344 = (n_329 & ~n_331)
	n_332 = (~n_329 & n_331)
	n_489 = (~n_420 & n_488)
	n_414 = (~n_377 & ~n_413)
	n_378 = (~n_354 & ~n_377)
	n_380 = (~n_350 & ~n_379)
	n_345 = (~n_332 & ~n_344)
	n_333 = (~n_285 & ~n_332)
	n_490 = (~n_419 & n_489)
	n_416 = (a[5] & ~n_414)
	n_415 = (~a[5] & n_414)
	n_409 = (n_378 & ~n_380)
	n_381 = (~n_378 & n_380)
	n_383 = (~n_343 & ~n_345)
	n_346 = (n_343 & n_345)
	n_340 = (n_333 & ~n_335)
	n_336 = (~n_333 & n_335)
	n_491 = (~n_416 & n_490)
	n_410 = (~n_381 & ~n_409)
	n_382 = (~n_350 & ~n_381)
	n_384 = (~n_346 & ~n_383)
	n_341 = (~n_336 & ~n_340)
	n_337 = (~n_281 & ~n_336)
	n_492 = (~n_415 & n_491)
	n_412 = (a[4] & ~n_410)
	n_411 = (~a[4] & n_410)
	n_405 = (n_382 & ~n_384)
	n_385 = (~n_382 & n_384)
	n_387 = (~n_339 & ~n_341)
	n_342 = (n_339 & n_341)
	n_391 = (~n_1 & n_337)
	n_338 = (n_1 & ~n_337)
	n_493 = (~n_412 & n_492)
	n_406 = (~n_385 & ~n_405)
	n_386 = (~n_346 & ~n_385)
	n_388 = (~n_342 & ~n_387)
	n_392 = (~n_338 & ~n_391)
	n_494 = (~n_411 & n_493)
	n_408 = (a[3] & ~n_406)
	n_407 = (~a[3] & n_406)
	n_401 = (n_386 & ~n_388)
	n_389 = (~n_386 & n_388)
	n_495 = (~n_408 & n_494)
	n_402 = (~n_389 & ~n_401)
	n_390 = (~n_342 & ~n_389)
	n_496 = (~n_407 & n_495)
	n_404 = (a[2] & ~n_402)
	n_403 = (~a[2] & n_402)
	n_397 = (n_390 & ~n_392)
	n_393 = (~n_390 & n_392)
	n_497 = (~n_404 & n_496)
	n_398 = (~n_393 & ~n_397)
	n_394 = (~n_338 & ~n_393)
	n_498 = (~n_403 & n_497)
	n_400 = (a[1] & ~n_398)
	n_399 = (~a[1] & n_398)
	n_396 = (~a[0] & ~n_394)
	n_395 = (a[0] & n_394)
	n_499 = (~n_400 & n_498)
	n_500 = (~n_399 & n_499)
	n_501 = (~n_396 & n_500)
	n_502 = (~n_395 & n_501)
	o_1 = n_502
	return o_1