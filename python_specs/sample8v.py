def F(XY_vars, util):
	v_14 = XY_vars[0, :]
	v_15 = util.tnorm_vectorized((v_13),(v_14))
	v_20 = util.tconorm_vectorized(util.negation(v_13),util.negation(v_14))
	v_30 = util.tnorm_vectorized(util.negation(v_13),util.negation(v_14))
	v_34 = util.tconorm_vectorized((v_13),(v_14))
	v_52 = util.tconorm_vectorized((v_8),(v_9))
	v_44 = util.tnorm_vectorized(util.negation(v_8),util.negation(v_9))
	v_54 = util.tnorm_vectorized((v_8),(v_9))
	v_46 = util.tconorm_vectorized(util.negation(v_8),util.negation(v_9))
	v_18 = util.tnorm_vectorized((v_8),util.negation(v_9))
	v_17 = util.tnorm_vectorized(util.negation(v_8),(v_9))
	v_11 = util.tconorm_vectorized(util.negation(v_8),(v_9))
	v_10 = util.tconorm_vectorized((v_8),util.negation(v_9))
	v_79 = util.tconorm_vectorized((v_39),(v_40))
	v_71 = util.tnorm_vectorized(util.negation(v_39),util.negation(v_40))
	v_81 = util.tnorm_vectorized((v_39),(v_40))
	v_73 = util.tconorm_vectorized(util.negation(v_39),util.negation(v_40))
	v_50 = util.tnorm_vectorized((v_39),util.negation(v_40))
	v_49 = util.tnorm_vectorized(util.negation(v_39),(v_40))
	v_42 = util.tconorm_vectorized(util.negation(v_39),(v_40))
	v_41 = util.tconorm_vectorized((v_39),util.negation(v_40))
	v_106 = util.tconorm_vectorized((v_66),(v_67))
	v_98 = util.tnorm_vectorized(util.negation(v_66),util.negation(v_67))
	v_108 = util.tnorm_vectorized((v_66),(v_67))
	v_100 = util.tconorm_vectorized(util.negation(v_66),util.negation(v_67))
	v_77 = util.tnorm_vectorized((v_66),util.negation(v_67))
	v_76 = util.tnorm_vectorized(util.negation(v_66),(v_67))
	v_69 = util.tconorm_vectorized(util.negation(v_66),(v_67))
	v_68 = util.tconorm_vectorized((v_66),util.negation(v_67))
	v_133 = util.tconorm_vectorized((v_93),(v_94))
	v_125 = util.tnorm_vectorized(util.negation(v_93),util.negation(v_94))
	v_135 = util.tnorm_vectorized((v_93),(v_94))
	v_127 = util.tconorm_vectorized(util.negation(v_93),util.negation(v_94))
	v_104 = util.tnorm_vectorized((v_93),util.negation(v_94))
	v_103 = util.tnorm_vectorized(util.negation(v_93),(v_94))
	v_96 = util.tconorm_vectorized(util.negation(v_93),(v_94))
	v_95 = util.tconorm_vectorized((v_93),util.negation(v_94))
	v_160 = util.tconorm_vectorized((v_120),(v_121))
	v_152 = util.tnorm_vectorized(util.negation(v_120),util.negation(v_121))
	v_162 = util.tnorm_vectorized((v_120),(v_121))
	v_154 = util.tconorm_vectorized(util.negation(v_120),util.negation(v_121))
	v_131 = util.tnorm_vectorized((v_120),util.negation(v_121))
	v_130 = util.tnorm_vectorized(util.negation(v_120),(v_121))
	v_123 = util.tconorm_vectorized(util.negation(v_120),(v_121))
	v_122 = util.tconorm_vectorized((v_120),util.negation(v_121))
	v_187 = util.tconorm_vectorized((v_147),(v_148))
	v_179 = util.tnorm_vectorized(util.negation(v_147),util.negation(v_148))
	v_189 = util.tnorm_vectorized((v_147),(v_148))
	v_181 = util.tconorm_vectorized(util.negation(v_147),util.negation(v_148))
	v_158 = util.tnorm_vectorized((v_147),util.negation(v_148))
	v_157 = util.tnorm_vectorized(util.negation(v_147),(v_148))
	v_150 = util.tconorm_vectorized(util.negation(v_147),(v_148))
	v_149 = util.tconorm_vectorized((v_147),util.negation(v_148))
	v_214 = util.tconorm_vectorized((v_174),(v_175))
	v_206 = util.tnorm_vectorized(util.negation(v_174),util.negation(v_175))
	v_216 = util.tnorm_vectorized((v_174),(v_175))
	v_208 = util.tconorm_vectorized(util.negation(v_174),util.negation(v_175))
	v_185 = util.tnorm_vectorized((v_174),util.negation(v_175))
	v_184 = util.tnorm_vectorized(util.negation(v_174),(v_175))
	v_177 = util.tconorm_vectorized(util.negation(v_174),(v_175))
	v_176 = util.tconorm_vectorized((v_174),util.negation(v_175))
	v_241 = util.tconorm_vectorized((v_201),(v_202))
	v_233 = util.tnorm_vectorized(util.negation(v_201),util.negation(v_202))
	v_243 = util.tnorm_vectorized((v_201),(v_202))
	v_235 = util.tconorm_vectorized(util.negation(v_201),util.negation(v_202))
	v_212 = util.tnorm_vectorized((v_201),util.negation(v_202))
	v_211 = util.tnorm_vectorized(util.negation(v_201),(v_202))
	v_204 = util.tconorm_vectorized(util.negation(v_201),(v_202))
	v_203 = util.tconorm_vectorized((v_201),util.negation(v_202))
	v_268 = util.tconorm_vectorized((v_228),(v_229))
	v_260 = util.tnorm_vectorized(util.negation(v_228),util.negation(v_229))
	v_270 = util.tnorm_vectorized((v_228),(v_229))
	v_262 = util.tconorm_vectorized(util.negation(v_228),util.negation(v_229))
	v_239 = util.tnorm_vectorized((v_228),util.negation(v_229))
	v_238 = util.tnorm_vectorized(util.negation(v_228),(v_229))
	v_231 = util.tconorm_vectorized(util.negation(v_228),(v_229))
	v_230 = util.tconorm_vectorized((v_228),util.negation(v_229))
	v_295 = util.tconorm_vectorized((v_255),(v_256))
	v_287 = util.tnorm_vectorized(util.negation(v_255),util.negation(v_256))
	v_297 = util.tnorm_vectorized((v_255),(v_256))
	v_289 = util.tconorm_vectorized(util.negation(v_255),util.negation(v_256))
	v_266 = util.tnorm_vectorized((v_255),util.negation(v_256))
	v_265 = util.tnorm_vectorized(util.negation(v_255),(v_256))
	v_258 = util.tconorm_vectorized(util.negation(v_255),(v_256))
	v_257 = util.tconorm_vectorized((v_255),util.negation(v_256))
	v_322 = util.tconorm_vectorized((v_282),(v_283))
	v_314 = util.tnorm_vectorized(util.negation(v_282),util.negation(v_283))
	v_324 = util.tnorm_vectorized((v_282),(v_283))
	v_316 = util.tconorm_vectorized(util.negation(v_282),util.negation(v_283))
	v_293 = util.tnorm_vectorized((v_282),util.negation(v_283))
	v_292 = util.tnorm_vectorized(util.negation(v_282),(v_283))
	v_285 = util.tconorm_vectorized(util.negation(v_282),(v_283))
	v_284 = util.tconorm_vectorized((v_282),util.negation(v_283))
	v_349 = util.tconorm_vectorized((v_309),(v_310))
	v_341 = util.tnorm_vectorized(util.negation(v_309),util.negation(v_310))
	v_351 = util.tnorm_vectorized((v_309),(v_310))
	v_343 = util.tconorm_vectorized(util.negation(v_309),util.negation(v_310))
	v_320 = util.tnorm_vectorized((v_309),util.negation(v_310))
	v_319 = util.tnorm_vectorized(util.negation(v_309),(v_310))
	v_312 = util.tconorm_vectorized(util.negation(v_309),(v_310))
	v_311 = util.tconorm_vectorized((v_309),util.negation(v_310))
	v_376 = util.tconorm_vectorized((v_336),(v_337))
	v_368 = util.tnorm_vectorized(util.negation(v_336),util.negation(v_337))
	v_378 = util.tnorm_vectorized((v_336),(v_337))
	v_370 = util.tconorm_vectorized(util.negation(v_336),util.negation(v_337))
	v_347 = util.tnorm_vectorized((v_336),util.negation(v_337))
	v_346 = util.tnorm_vectorized(util.negation(v_336),(v_337))
	v_339 = util.tconorm_vectorized(util.negation(v_336),(v_337))
	v_338 = util.tconorm_vectorized((v_336),util.negation(v_337))
	v_403 = util.tconorm_vectorized((v_363),(v_364))
	v_395 = util.tnorm_vectorized(util.negation(v_363),util.negation(v_364))
	v_405 = util.tnorm_vectorized((v_363),(v_364))
	v_397 = util.tconorm_vectorized(util.negation(v_363),util.negation(v_364))
	v_374 = util.tnorm_vectorized((v_363),util.negation(v_364))
	v_373 = util.tnorm_vectorized(util.negation(v_363),(v_364))
	v_366 = util.tconorm_vectorized(util.negation(v_363),(v_364))
	v_365 = util.tconorm_vectorized((v_363),util.negation(v_364))
	v_430 = util.tconorm_vectorized((v_390),(v_391))
	v_422 = util.tnorm_vectorized(util.negation(v_390),util.negation(v_391))
	v_432 = util.tnorm_vectorized((v_390),(v_391))
	v_424 = util.tconorm_vectorized(util.negation(v_390),util.negation(v_391))
	v_401 = util.tnorm_vectorized((v_390),util.negation(v_391))
	v_400 = util.tnorm_vectorized(util.negation(v_390),(v_391))
	v_393 = util.tconorm_vectorized(util.negation(v_390),(v_391))
	v_392 = util.tconorm_vectorized((v_390),util.negation(v_391))
	v_457 = util.tconorm_vectorized((v_417),(v_418))
	v_449 = util.tnorm_vectorized(util.negation(v_417),util.negation(v_418))
	v_459 = util.tnorm_vectorized((v_417),(v_418))
	v_451 = util.tconorm_vectorized(util.negation(v_417),util.negation(v_418))
	v_428 = util.tnorm_vectorized((v_417),util.negation(v_418))
	v_427 = util.tnorm_vectorized(util.negation(v_417),(v_418))
	v_420 = util.tconorm_vectorized(util.negation(v_417),(v_418))
	v_419 = util.tconorm_vectorized((v_417),util.negation(v_418))
	v_484 = util.tconorm_vectorized((v_444),(v_445))
	v_476 = util.tnorm_vectorized(util.negation(v_444),util.negation(v_445))
	v_486 = util.tnorm_vectorized((v_444),(v_445))
	v_478 = util.tconorm_vectorized(util.negation(v_444),util.negation(v_445))
	v_455 = util.tnorm_vectorized((v_444),util.negation(v_445))
	v_454 = util.tnorm_vectorized(util.negation(v_444),(v_445))
	v_447 = util.tconorm_vectorized(util.negation(v_444),(v_445))
	v_446 = util.tconorm_vectorized((v_444),util.negation(v_445))
	v_498 = util.tconorm_vectorized((v_471),(v_472))
	v_504 = util.tnorm_vectorized(util.negation(v_471),util.negation(v_472))
	v_500 = util.tnorm_vectorized((v_471),(v_472))
	v_506 = util.tconorm_vectorized(util.negation(v_471),util.negation(v_472))
	v_482 = util.tnorm_vectorized((v_471),util.negation(v_472))
	v_481 = util.tnorm_vectorized(util.negation(v_471),(v_472))
	v_474 = util.tconorm_vectorized(util.negation(v_471),(v_472))
	v_473 = util.tconorm_vectorized((v_471),util.negation(v_472))
	v_610 = util.tconorm_vectorized(util.negation(v_1),util.negation(v_607))
	v_608 = util.tnorm_vectorized((v_1),(v_607))
	v_2 = v
	v_4 = v
	v_612 = util.negation(v)
	o_1 = x
	v_31 = util.tconorm_vectorized((v_15),(v_30))
	v_35 = util.tnorm_vectorized((v_20),(v_34))
	v_53 = util.tnorm_vectorized((v_15),(v_52))
	v_45 = util.tconorm_vectorized((v_20),(v_44))
	v_19 = util.tconorm_vectorized((v_17),(v_18))
	v_12 = util.tnorm_vectorized((v_10),(v_11))
	v_51 = util.tconorm_vectorized((v_49),(v_50))
	v_43 = util.tnorm_vectorized((v_41),(v_42))
	v_78 = util.tconorm_vectorized((v_76),(v_77))
	v_70 = util.tnorm_vectorized((v_68),(v_69))
	v_105 = util.tconorm_vectorized((v_103),(v_104))
	v_97 = util.tnorm_vectorized((v_95),(v_96))
	v_132 = util.tconorm_vectorized((v_130),(v_131))
	v_124 = util.tnorm_vectorized((v_122),(v_123))
	v_159 = util.tconorm_vectorized((v_157),(v_158))
	v_151 = util.tnorm_vectorized((v_149),(v_150))
	v_186 = util.tconorm_vectorized((v_184),(v_185))
	v_178 = util.tnorm_vectorized((v_176),(v_177))
	v_213 = util.tconorm_vectorized((v_211),(v_212))
	v_205 = util.tnorm_vectorized((v_203),(v_204))
	v_240 = util.tconorm_vectorized((v_238),(v_239))
	v_232 = util.tnorm_vectorized((v_230),(v_231))
	v_267 = util.tconorm_vectorized((v_265),(v_266))
	v_259 = util.tnorm_vectorized((v_257),(v_258))
	v_294 = util.tconorm_vectorized((v_292),(v_293))
	v_286 = util.tnorm_vectorized((v_284),(v_285))
	v_321 = util.tconorm_vectorized((v_319),(v_320))
	v_313 = util.tnorm_vectorized((v_311),(v_312))
	v_348 = util.tconorm_vectorized((v_346),(v_347))
	v_340 = util.tnorm_vectorized((v_338),(v_339))
	v_375 = util.tconorm_vectorized((v_373),(v_374))
	v_367 = util.tnorm_vectorized((v_365),(v_366))
	v_402 = util.tconorm_vectorized((v_400),(v_401))
	v_394 = util.tnorm_vectorized((v_392),(v_393))
	v_429 = util.tconorm_vectorized((v_427),(v_428))
	v_421 = util.tnorm_vectorized((v_419),(v_420))
	v_456 = util.tconorm_vectorized((v_454),(v_455))
	v_448 = util.tnorm_vectorized((v_446),(v_447))
	v_483 = util.tconorm_vectorized((v_481),(v_482))
	v_475 = util.tnorm_vectorized((v_473),(v_474))
	v_611 = util.tconorm_vectorized((v_606),(v_610))
	v_609 = util.tconorm_vectorized(util.negation(v_606),(v_608))
	v_5 = util.tconorm_vectorized((v_2),(v_4))
	v_530 = util.tconorm_vectorized((v_31),(v_32))
	v_33 = util.tnorm_vectorized((v_31),(v_32))
	v_529 = util.tconorm_vectorized(util.negation(v_32),(v_35))
	v_36 = util.tnorm_vectorized(util.negation(v_32),(v_35))
	v_55 = util.tconorm_vectorized((v_53),(v_54))
	v_47 = util.tnorm_vectorized((v_45),(v_46))
	v_21 = util.tconorm_vectorized((v_19),(v_20))
	v_25 = util.tnorm_vectorized((v_19),(v_20))
	v_16 = util.tconorm_vectorized((v_12),(v_15))
	v_26 = util.tnorm_vectorized((v_12),(v_15))
	v_531 = util.tnorm_vectorized((v_529),(v_530))
	v_37 = util.tconorm_vectorized((v_33),(v_36))
	v_80 = util.tnorm_vectorized((v_55),(v_79))
	v_56 = util.tnorm_vectorized((v_51),(v_55))
	v_60 = util.tconorm_vectorized((v_51),(v_55))
	v_72 = util.tconorm_vectorized((v_47),(v_71))
	v_48 = util.tnorm_vectorized((v_43),(v_47))
	v_61 = util.tconorm_vectorized((v_43),(v_47))
	v_22 = util.tnorm_vectorized((v_16),(v_21))
	v_27 = util.tconorm_vectorized((v_25),(v_26))
	v_82 = util.tconorm_vectorized((v_80),(v_81))
	v_613 = util.negation(v_6)
	v_74 = util.tnorm_vectorized((v_72),(v_73))
	v_57 = util.tconorm_vectorized((v_48),(v_56))
	v_618 = v_6
	v_62 = util.tnorm_vectorized((v_60),(v_61))
	v_527 = util.tconorm_vectorized((v_22),(v_23))
	v_24 = util.tnorm_vectorized((v_22),(v_23))
	v_526 = util.tconorm_vectorized(util.negation(v_23),(v_27))
	v_28 = util.tnorm_vectorized(util.negation(v_23),(v_27))
	v_107 = util.tnorm_vectorized((v_82),(v_106))
	v_83 = util.tnorm_vectorized((v_78),(v_82))
	v_87 = util.tconorm_vectorized((v_78),(v_82))
	v_614 = util.tnorm_vectorized((v_612),(v_613))
	v_99 = util.tconorm_vectorized((v_74),(v_98))
	v_75 = util.tnorm_vectorized((v_70),(v_74))
	v_88 = util.tconorm_vectorized((v_70),(v_74))
	v_534 = util.tconorm_vectorized((v_57),(v_58))
	v_59 = util.tnorm_vectorized((v_57),(v_58))
	v_533 = util.tconorm_vectorized(util.negation(v_58),(v_62))
	v_63 = util.tnorm_vectorized(util.negation(v_58),(v_62))
	v_528 = util.tnorm_vectorized((v_526),(v_527))
	v_29 = util.tconorm_vectorized((v_24),(v_28))
	v_109 = util.tconorm_vectorized((v_107),(v_108))
	v_101 = util.tnorm_vectorized((v_99),(v_100))
	v_84 = util.tconorm_vectorized((v_75),(v_83))
	v_89 = util.tnorm_vectorized((v_87),(v_88))
	v_535 = util.tnorm_vectorized((v_533),(v_534))
	v_64 = util.tconorm_vectorized((v_59),(v_63))
	v_532 = util.tnorm_vectorized((v_528),(v_531))
	v_38 = util.tconorm_vectorized((v_29),(v_37))
	v_134 = util.tnorm_vectorized((v_109),(v_133))
	v_110 = util.tnorm_vectorized((v_105),(v_109))
	v_114 = util.tconorm_vectorized((v_105),(v_109))
	v_126 = util.tconorm_vectorized((v_101),(v_125))
	v_102 = util.tnorm_vectorized((v_97),(v_101))
	v_115 = util.tconorm_vectorized((v_97),(v_101))
	v_538 = util.tconorm_vectorized((v_84),(v_85))
	v_86 = util.tnorm_vectorized((v_84),(v_85))
	v_537 = util.tconorm_vectorized(util.negation(v_85),(v_89))
	v_90 = util.tnorm_vectorized(util.negation(v_85),(v_89))
	v_536 = util.tnorm_vectorized((v_532),(v_535))
	v_65 = util.tconorm_vectorized((v_38),(v_64))
	v_136 = util.tconorm_vectorized((v_134),(v_135))
	v_128 = util.tnorm_vectorized((v_126),(v_127))
	v_111 = util.tconorm_vectorized((v_102),(v_110))
	v_116 = util.tnorm_vectorized((v_114),(v_115))
	v_539 = util.tnorm_vectorized((v_537),(v_538))
	v_91 = util.tconorm_vectorized((v_86),(v_90))
	v_161 = util.tnorm_vectorized((v_136),(v_160))
	v_137 = util.tnorm_vectorized((v_132),(v_136))
	v_141 = util.tconorm_vectorized((v_132),(v_136))
	v_153 = util.tconorm_vectorized((v_128),(v_152))
	v_129 = util.tnorm_vectorized((v_124),(v_128))
	v_142 = util.tconorm_vectorized((v_124),(v_128))
	v_542 = util.tconorm_vectorized((v_111),(v_112))
	v_113 = util.tnorm_vectorized((v_111),(v_112))
	v_541 = util.tconorm_vectorized(util.negation(v_112),(v_116))
	v_117 = util.tnorm_vectorized(util.negation(v_112),(v_116))
	v_540 = util.tnorm_vectorized((v_536),(v_539))
	v_92 = util.tconorm_vectorized((v_65),(v_91))
	v_163 = util.tconorm_vectorized((v_161),(v_162))
	v_155 = util.tnorm_vectorized((v_153),(v_154))
	v_138 = util.tconorm_vectorized((v_129),(v_137))
	v_143 = util.tnorm_vectorized((v_141),(v_142))
	v_543 = util.tnorm_vectorized((v_541),(v_542))
	v_118 = util.tconorm_vectorized((v_113),(v_117))
	v_188 = util.tnorm_vectorized((v_163),(v_187))
	v_164 = util.tnorm_vectorized((v_159),(v_163))
	v_168 = util.tconorm_vectorized((v_159),(v_163))
	v_180 = util.tconorm_vectorized((v_155),(v_179))
	v_156 = util.tnorm_vectorized((v_151),(v_155))
	v_169 = util.tconorm_vectorized((v_151),(v_155))
	v_546 = util.tconorm_vectorized((v_138),(v_139))
	v_140 = util.tnorm_vectorized((v_138),(v_139))
	v_545 = util.tconorm_vectorized(util.negation(v_139),(v_143))
	v_144 = util.tnorm_vectorized(util.negation(v_139),(v_143))
	v_544 = util.tnorm_vectorized((v_540),(v_543))
	v_119 = util.tconorm_vectorized((v_92),(v_118))
	v_190 = util.tconorm_vectorized((v_188),(v_189))
	v_182 = util.tnorm_vectorized((v_180),(v_181))
	v_165 = util.tconorm_vectorized((v_156),(v_164))
	v_170 = util.tnorm_vectorized((v_168),(v_169))
	v_547 = util.tnorm_vectorized((v_545),(v_546))
	v_145 = util.tconorm_vectorized((v_140),(v_144))
	v_215 = util.tnorm_vectorized((v_190),(v_214))
	v_191 = util.tnorm_vectorized((v_186),(v_190))
	v_195 = util.tconorm_vectorized((v_186),(v_190))
	v_207 = util.tconorm_vectorized((v_182),(v_206))
	v_183 = util.tnorm_vectorized((v_178),(v_182))
	v_196 = util.tconorm_vectorized((v_178),(v_182))
	v_550 = util.tconorm_vectorized((v_165),(v_166))
	v_167 = util.tnorm_vectorized((v_165),(v_166))
	v_549 = util.tconorm_vectorized(util.negation(v_166),(v_170))
	v_171 = util.tnorm_vectorized(util.negation(v_166),(v_170))
	v_548 = util.tnorm_vectorized((v_544),(v_547))
	v_146 = util.tconorm_vectorized((v_119),(v_145))
	v_217 = util.tconorm_vectorized((v_215),(v_216))
	v_209 = util.tnorm_vectorized((v_207),(v_208))
	v_192 = util.tconorm_vectorized((v_183),(v_191))
	v_197 = util.tnorm_vectorized((v_195),(v_196))
	v_551 = util.tnorm_vectorized((v_549),(v_550))
	v_172 = util.tconorm_vectorized((v_167),(v_171))
	v_242 = util.tnorm_vectorized((v_217),(v_241))
	v_218 = util.tnorm_vectorized((v_213),(v_217))
	v_222 = util.tconorm_vectorized((v_213),(v_217))
	v_234 = util.tconorm_vectorized((v_209),(v_233))
	v_210 = util.tnorm_vectorized((v_205),(v_209))
	v_223 = util.tconorm_vectorized((v_205),(v_209))
	v_554 = util.tconorm_vectorized((v_192),(v_193))
	v_194 = util.tnorm_vectorized((v_192),(v_193))
	v_553 = util.tconorm_vectorized(util.negation(v_193),(v_197))
	v_198 = util.tnorm_vectorized(util.negation(v_193),(v_197))
	v_552 = util.tnorm_vectorized((v_548),(v_551))
	v_173 = util.tconorm_vectorized((v_146),(v_172))
	v_244 = util.tconorm_vectorized((v_242),(v_243))
	v_236 = util.tnorm_vectorized((v_234),(v_235))
	v_219 = util.tconorm_vectorized((v_210),(v_218))
	v_224 = util.tnorm_vectorized((v_222),(v_223))
	v_555 = util.tnorm_vectorized((v_553),(v_554))
	v_199 = util.tconorm_vectorized((v_194),(v_198))
	v_269 = util.tnorm_vectorized((v_244),(v_268))
	v_245 = util.tnorm_vectorized((v_240),(v_244))
	v_249 = util.tconorm_vectorized((v_240),(v_244))
	v_261 = util.tconorm_vectorized((v_236),(v_260))
	v_237 = util.tnorm_vectorized((v_232),(v_236))
	v_250 = util.tconorm_vectorized((v_232),(v_236))
	v_558 = util.tconorm_vectorized((v_219),(v_220))
	v_221 = util.tnorm_vectorized((v_219),(v_220))
	v_557 = util.tconorm_vectorized(util.negation(v_220),(v_224))
	v_225 = util.tnorm_vectorized(util.negation(v_220),(v_224))
	v_556 = util.tnorm_vectorized((v_552),(v_555))
	v_200 = util.tconorm_vectorized((v_173),(v_199))
	v_271 = util.tconorm_vectorized((v_269),(v_270))
	v_263 = util.tnorm_vectorized((v_261),(v_262))
	v_246 = util.tconorm_vectorized((v_237),(v_245))
	v_251 = util.tnorm_vectorized((v_249),(v_250))
	v_559 = util.tnorm_vectorized((v_557),(v_558))
	v_226 = util.tconorm_vectorized((v_221),(v_225))
	v_296 = util.tnorm_vectorized((v_271),(v_295))
	v_272 = util.tnorm_vectorized((v_267),(v_271))
	v_276 = util.tconorm_vectorized((v_267),(v_271))
	v_288 = util.tconorm_vectorized((v_263),(v_287))
	v_264 = util.tnorm_vectorized((v_259),(v_263))
	v_277 = util.tconorm_vectorized((v_259),(v_263))
	v_562 = util.tconorm_vectorized((v_246),(v_247))
	v_248 = util.tnorm_vectorized((v_246),(v_247))
	v_561 = util.tconorm_vectorized(util.negation(v_247),(v_251))
	v_252 = util.tnorm_vectorized(util.negation(v_247),(v_251))
	v_560 = util.tnorm_vectorized((v_556),(v_559))
	v_227 = util.tconorm_vectorized((v_200),(v_226))
	v_298 = util.tconorm_vectorized((v_296),(v_297))
	v_290 = util.tnorm_vectorized((v_288),(v_289))
	v_273 = util.tconorm_vectorized((v_264),(v_272))
	v_278 = util.tnorm_vectorized((v_276),(v_277))
	v_563 = util.tnorm_vectorized((v_561),(v_562))
	v_253 = util.tconorm_vectorized((v_248),(v_252))
	v_323 = util.tnorm_vectorized((v_298),(v_322))
	v_299 = util.tnorm_vectorized((v_294),(v_298))
	v_303 = util.tconorm_vectorized((v_294),(v_298))
	v_315 = util.tconorm_vectorized((v_290),(v_314))
	v_291 = util.tnorm_vectorized((v_286),(v_290))
	v_304 = util.tconorm_vectorized((v_286),(v_290))
	v_566 = util.tconorm_vectorized((v_273),(v_274))
	v_275 = util.tnorm_vectorized((v_273),(v_274))
	v_565 = util.tconorm_vectorized(util.negation(v_274),(v_278))
	v_279 = util.tnorm_vectorized(util.negation(v_274),(v_278))
	v_564 = util.tnorm_vectorized((v_560),(v_563))
	v_254 = util.tconorm_vectorized((v_227),(v_253))
	v_325 = util.tconorm_vectorized((v_323),(v_324))
	v_317 = util.tnorm_vectorized((v_315),(v_316))
	v_300 = util.tconorm_vectorized((v_291),(v_299))
	v_305 = util.tnorm_vectorized((v_303),(v_304))
	v_567 = util.tnorm_vectorized((v_565),(v_566))
	v_280 = util.tconorm_vectorized((v_275),(v_279))
	v_350 = util.tnorm_vectorized((v_325),(v_349))
	v_326 = util.tnorm_vectorized((v_321),(v_325))
	v_330 = util.tconorm_vectorized((v_321),(v_325))
	v_342 = util.tconorm_vectorized((v_317),(v_341))
	v_318 = util.tnorm_vectorized((v_313),(v_317))
	v_331 = util.tconorm_vectorized((v_313),(v_317))
	v_570 = util.tconorm_vectorized((v_300),(v_301))
	v_302 = util.tnorm_vectorized((v_300),(v_301))
	v_569 = util.tconorm_vectorized(util.negation(v_301),(v_305))
	v_306 = util.tnorm_vectorized(util.negation(v_301),(v_305))
	v_568 = util.tnorm_vectorized((v_564),(v_567))
	v_281 = util.tconorm_vectorized((v_254),(v_280))
	v_352 = util.tconorm_vectorized((v_350),(v_351))
	v_344 = util.tnorm_vectorized((v_342),(v_343))
	v_327 = util.tconorm_vectorized((v_318),(v_326))
	v_332 = util.tnorm_vectorized((v_330),(v_331))
	v_571 = util.tnorm_vectorized((v_569),(v_570))
	v_307 = util.tconorm_vectorized((v_302),(v_306))
	v_377 = util.tnorm_vectorized((v_352),(v_376))
	v_353 = util.tnorm_vectorized((v_348),(v_352))
	v_357 = util.tconorm_vectorized((v_348),(v_352))
	v_369 = util.tconorm_vectorized((v_344),(v_368))
	v_345 = util.tnorm_vectorized((v_340),(v_344))
	v_358 = util.tconorm_vectorized((v_340),(v_344))
	v_574 = util.tconorm_vectorized((v_327),(v_328))
	v_329 = util.tnorm_vectorized((v_327),(v_328))
	v_573 = util.tconorm_vectorized(util.negation(v_328),(v_332))
	v_333 = util.tnorm_vectorized(util.negation(v_328),(v_332))
	v_572 = util.tnorm_vectorized((v_568),(v_571))
	v_308 = util.tconorm_vectorized((v_281),(v_307))
	v_379 = util.tconorm_vectorized((v_377),(v_378))
	v_371 = util.tnorm_vectorized((v_369),(v_370))
	v_354 = util.tconorm_vectorized((v_345),(v_353))
	v_359 = util.tnorm_vectorized((v_357),(v_358))
	v_575 = util.tnorm_vectorized((v_573),(v_574))
	v_334 = util.tconorm_vectorized((v_329),(v_333))
	v_404 = util.tnorm_vectorized((v_379),(v_403))
	v_380 = util.tnorm_vectorized((v_375),(v_379))
	v_384 = util.tconorm_vectorized((v_375),(v_379))
	v_396 = util.tconorm_vectorized((v_371),(v_395))
	v_372 = util.tnorm_vectorized((v_367),(v_371))
	v_385 = util.tconorm_vectorized((v_367),(v_371))
	v_578 = util.tconorm_vectorized((v_354),(v_355))
	v_356 = util.tnorm_vectorized((v_354),(v_355))
	v_577 = util.tconorm_vectorized(util.negation(v_355),(v_359))
	v_360 = util.tnorm_vectorized(util.negation(v_355),(v_359))
	v_576 = util.tnorm_vectorized((v_572),(v_575))
	v_335 = util.tconorm_vectorized((v_308),(v_334))
	v_406 = util.tconorm_vectorized((v_404),(v_405))
	v_398 = util.tnorm_vectorized((v_396),(v_397))
	v_381 = util.tconorm_vectorized((v_372),(v_380))
	v_386 = util.tnorm_vectorized((v_384),(v_385))
	v_579 = util.tnorm_vectorized((v_577),(v_578))
	v_361 = util.tconorm_vectorized((v_356),(v_360))
	v_431 = util.tnorm_vectorized((v_406),(v_430))
	v_407 = util.tnorm_vectorized((v_402),(v_406))
	v_411 = util.tconorm_vectorized((v_402),(v_406))
	v_423 = util.tconorm_vectorized((v_398),(v_422))
	v_399 = util.tnorm_vectorized((v_394),(v_398))
	v_412 = util.tconorm_vectorized((v_394),(v_398))
	v_582 = util.tconorm_vectorized((v_381),(v_382))
	v_383 = util.tnorm_vectorized((v_381),(v_382))
	v_581 = util.tconorm_vectorized(util.negation(v_382),(v_386))
	v_387 = util.tnorm_vectorized(util.negation(v_382),(v_386))
	v_580 = util.tnorm_vectorized((v_576),(v_579))
	v_362 = util.tconorm_vectorized((v_335),(v_361))
	v_433 = util.tconorm_vectorized((v_431),(v_432))
	v_425 = util.tnorm_vectorized((v_423),(v_424))
	v_408 = util.tconorm_vectorized((v_399),(v_407))
	v_413 = util.tnorm_vectorized((v_411),(v_412))
	v_583 = util.tnorm_vectorized((v_581),(v_582))
	v_388 = util.tconorm_vectorized((v_383),(v_387))
	v_458 = util.tnorm_vectorized((v_433),(v_457))
	v_434 = util.tnorm_vectorized((v_429),(v_433))
	v_438 = util.tconorm_vectorized((v_429),(v_433))
	v_450 = util.tconorm_vectorized((v_425),(v_449))
	v_426 = util.tnorm_vectorized((v_421),(v_425))
	v_439 = util.tconorm_vectorized((v_421),(v_425))
	v_586 = util.tconorm_vectorized((v_408),(v_409))
	v_410 = util.tnorm_vectorized((v_408),(v_409))
	v_585 = util.tconorm_vectorized(util.negation(v_409),(v_413))
	v_414 = util.tnorm_vectorized(util.negation(v_409),(v_413))
	v_584 = util.tnorm_vectorized((v_580),(v_583))
	v_389 = util.tconorm_vectorized((v_362),(v_388))
	v_460 = util.tconorm_vectorized((v_458),(v_459))
	v_452 = util.tnorm_vectorized((v_450),(v_451))
	v_435 = util.tconorm_vectorized((v_426),(v_434))
	v_440 = util.tnorm_vectorized((v_438),(v_439))
	v_587 = util.tnorm_vectorized((v_585),(v_586))
	v_415 = util.tconorm_vectorized((v_410),(v_414))
	v_485 = util.tnorm_vectorized((v_460),(v_484))
	v_461 = util.tnorm_vectorized((v_456),(v_460))
	v_465 = util.tconorm_vectorized((v_456),(v_460))
	v_477 = util.tconorm_vectorized((v_452),(v_476))
	v_453 = util.tnorm_vectorized((v_448),(v_452))
	v_466 = util.tconorm_vectorized((v_448),(v_452))
	v_590 = util.tconorm_vectorized((v_435),(v_436))
	v_437 = util.tnorm_vectorized((v_435),(v_436))
	v_589 = util.tconorm_vectorized(util.negation(v_436),(v_440))
	v_441 = util.tnorm_vectorized(util.negation(v_436),(v_440))
	v_588 = util.tnorm_vectorized((v_584),(v_587))
	v_416 = util.tconorm_vectorized((v_389),(v_415))
	v_487 = util.tconorm_vectorized((v_485),(v_486))
	v_479 = util.tnorm_vectorized((v_477),(v_478))
	v_462 = util.tconorm_vectorized((v_453),(v_461))
	v_467 = util.tnorm_vectorized((v_465),(v_466))
	v_591 = util.tnorm_vectorized((v_589),(v_590))
	v_442 = util.tconorm_vectorized((v_437),(v_441))
	v_499 = util.tnorm_vectorized((v_487),(v_498))
	v_488 = util.tnorm_vectorized((v_483),(v_487))
	v_492 = util.tconorm_vectorized((v_483),(v_487))
	v_505 = util.tconorm_vectorized((v_479),(v_504))
	v_480 = util.tnorm_vectorized((v_475),(v_479))
	v_493 = util.tconorm_vectorized((v_475),(v_479))
	v_594 = util.tconorm_vectorized((v_462),(v_463))
	v_464 = util.tnorm_vectorized((v_462),(v_463))
	v_593 = util.tconorm_vectorized(util.negation(v_463),(v_467))
	v_468 = util.tnorm_vectorized(util.negation(v_463),(v_467))
	v_592 = util.tnorm_vectorized((v_588),(v_591))
	v_443 = util.tconorm_vectorized((v_416),(v_442))
	v_501 = util.tconorm_vectorized((v_499),(v_500))
	v_507 = util.tnorm_vectorized((v_505),(v_506))
	v_489 = util.tconorm_vectorized((v_480),(v_488))
	v_494 = util.tnorm_vectorized((v_492),(v_493))
	v_595 = util.tnorm_vectorized((v_593),(v_594))
	v_469 = util.tconorm_vectorized((v_464),(v_468))
	v_513 = util.tnorm_vectorized((v_501),(v_502))
	v_503 = util.tconorm_vectorized((v_501),(v_502))
	v_512 = util.tnorm_vectorized(util.negation(v_502),(v_507))
	v_508 = util.tconorm_vectorized(util.negation(v_502),(v_507))
	v_598 = util.tconorm_vectorized((v_489),(v_490))
	v_491 = util.tnorm_vectorized((v_489),(v_490))
	v_597 = util.tconorm_vectorized(util.negation(v_490),(v_494))
	v_495 = util.tnorm_vectorized(util.negation(v_490),(v_494))
	v_596 = util.tnorm_vectorized((v_592),(v_595))
	v_470 = util.tconorm_vectorized((v_443),(v_469))
	v_514 = util.tconorm_vectorized((v_512),(v_513))
	v_509 = util.tnorm_vectorized((v_503),(v_508))
	v_599 = util.tnorm_vectorized((v_597),(v_598))
	v_496 = util.tconorm_vectorized((v_491),(v_495))
	v_515 = util.tnorm_vectorized(util.negation(v_510),(v_514))
	v_519 = util.tconorm_vectorized(util.negation(v_510),(v_514))
	v_511 = util.tnorm_vectorized((v_509),(v_510))
	v_520 = util.tconorm_vectorized((v_509),(v_510))
	v_600 = util.tnorm_vectorized((v_596),(v_599))
	v_497 = util.tconorm_vectorized((v_470),(v_496))
	v_516 = util.tconorm_vectorized((v_511),(v_515))
	v_521 = util.tnorm_vectorized((v_519),(v_520))
	v_602 = util.tconorm_vectorized((v_516),(v_517))
	v_518 = util.tnorm_vectorized((v_516),(v_517))
	v_601 = util.tconorm_vectorized(util.negation(v_517),(v_521))
	v_522 = util.tnorm_vectorized(util.negation(v_517),(v_521))
	v_603 = util.tnorm_vectorized((v_601),(v_602))
	v_523 = util.tconorm_vectorized((v_518),(v_522))
	v_604 = util.tnorm_vectorized((v_600),(v_603))
	v_524 = util.tconorm_vectorized((v_497),(v_523))
	v_605 = util.tconorm_vectorized((v_7),(v_604))
	v_525 = util.tconorm_vectorized(util.negation(v_7),(v_524))
	v_617 = util.tnorm_vectorized(((util.tnorm_vectorized(((util.tnorm_vectorized(((util.tnorm_vectorized((v_6),(v_525)))),(v_605)))),(v_609)))),(v_611))
	v_615 = util.tnorm_vectorized((v_617),(v_618))
	x_1 = util.tconorm_vectorized((v_5),(v_615))
	return o_1