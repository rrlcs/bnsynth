def F(XY_vars, util):
	i2 = XY_vars[0, :]
	i2 = XY_vars[1, :]
	i2 = XY_vars[2, :]
	i2 = XY_vars[3, :]
	i2 = XY_vars[4, :]
	i1 = XY_vars[5, :]
	i1 = XY_vars[6, :]
	i1 = XY_vars[7, :]
	i1 = XY_vars[8, :]
	i1 = XY_vars[9, :]
	a = XY_vars[10, :]
	a = XY_vars[11, :]
	a = XY_vars[12, :]
	a = XY_vars[13, :]
	a = XY_vars[14, :]
	a = XY_vars[15, :]
	a = XY_vars[16, :]
	a = XY_vars[17, :]
	a = XY_vars[18, :]
	a = XY_vars[19, :]
	n_8 = (i2[6] & i1[9])
	n_14 = (i2[7] & i1[9])
	n_173 = (~i2[7] & ~i2[6])
	n_7 = (i2[7] & i1[8])
	n_4 = (i2[6] & i1[8])
	n_13 = (i2[8] & i1[8])
	n_10 = (i2[8] & i1[9])
	n_11 = (i2[9] & i1[8])
	n_172 = (i2[9] & ~i2[8])
	n_170 = (i2[9] & i1[9])
	n_5 = (i2[5] & i1[9])
	n_25 = (i2[5] & i1[8])
	n_38 = (i2[8] & i1[7])
	n_40 = (i2[9] & i1[7])
	n_34 = (i2[7] & i1[7])
	n_28 = (i2[6] & i1[7])
	n_3 = (i2[5] & i1[7])
	n_160 = (~i1[8] & ~i1[7])
	n_73 = (i2[8] & i1[6])
	n_77 = (i2[9] & i1[6])
	n_67 = (i2[7] & i1[6])
	n_63 = (i2[6] & i1[6])
	n_2 = (i2[5] & i1[6])
	n_161 = (~i1[6] & ~i1[5])
	n_110 = (i2[9] & i1[5])
	n_104 = (i2[8] & i1[5])
	n_100 = (i2[7] & i1[5])
	n_96 = (i2[6] & i1[5])
	n_1 = (i2[5] & i1[5])
	n_159 = (i2[9] & ~a[9])
	n_174 = (~i2[5] & n_173)
	n_9 = (n_7 & n_8)
	n_17 = (~n_7 & ~n_8)
	n_15 = (n_13 & n_14)
	n_43 = (~n_13 & ~n_14)
	n_12 = (n_10 & n_11)
	n_165 = (~n_10 & ~n_11)
	n_171 = (a[9] & ~n_170)
	n_6 = (n_4 & n_5)
	n_21 = (~n_4 & ~n_5)
	n_162 = (n_160 & n_161)
	n_175 = (n_172 & n_174)
	n_18 = (~n_17 & ~n_9)
	n_16 = (~n_12 & ~n_15)
	n_41 = (n_12 & n_15)
	n_166 = (~n_12 & ~n_165)
	n_22 = (~n_21 & ~n_6)
	n_163 = (~n_159 & ~n_162)
	n_176 = (~n_171 & ~n_175)
	n_19 = (~n_16 & n_18)
	n_36 = (n_16 & ~n_18)
	n_42 = (~n_16 & ~n_41)
	n_168 = (a[8] & n_166)
	n_167 = (~a[8] & ~n_166)
	n_164 = (i1[9] & ~n_163)
	n_20 = (~n_9 & ~n_19)
	n_37 = (~n_19 & ~n_36)
	n_44 = (~n_42 & ~n_43)
	n_169 = (~n_167 & ~n_168)
	n_23 = (~n_20 & n_22)
	n_32 = (n_20 & ~n_22)
	n_46 = (~n_37 & ~n_38)
	n_39 = (n_37 & n_38)
	n_45 = (n_40 & n_44)
	n_155 = (~n_40 & ~n_44)
	n_177 = (~n_169 & n_176)
	n_24 = (~n_6 & ~n_23)
	n_33 = (~n_23 & ~n_32)
	n_47 = (~n_39 & ~n_46)
	n_156 = (~n_45 & ~n_155)
	n_178 = (~n_164 & n_177)
	n_29 = (n_24 & ~n_25)
	n_26 = (~n_24 & n_25)
	n_50 = (~n_33 & ~n_34)
	n_35 = (n_33 & n_34)
	n_48 = (n_45 & n_47)
	n_75 = (~n_45 & ~n_47)
	n_158 = (a[7] & ~n_156)
	n_157 = (~a[7] & n_156)
	n_30 = (~n_26 & ~n_29)
	n_58 = (~n_3 & ~n_26)
	n_27 = (n_3 & n_26)
	n_51 = (~n_35 & ~n_50)
	n_49 = (~n_39 & ~n_48)
	n_76 = (~n_48 & ~n_75)
	n_179 = (~n_158 & n_178)
	n_54 = (~n_28 & ~n_30)
	n_31 = (n_28 & n_30)
	n_59 = (~n_27 & ~n_58)
	n_52 = (~n_49 & n_51)
	n_71 = (n_49 & ~n_51)
	n_78 = (n_76 & n_77)
	n_151 = (~n_76 & ~n_77)
	n_180 = (~n_157 & n_179)
	n_55 = (~n_31 & ~n_54)
	n_53 = (~n_35 & ~n_52)
	n_72 = (~n_52 & ~n_71)
	n_152 = (~n_78 & ~n_151)
	n_68 = (n_53 & ~n_55)
	n_56 = (~n_53 & n_55)
	n_79 = (~n_72 & ~n_73)
	n_74 = (n_72 & n_73)
	n_154 = (a[6] & ~n_152)
	n_153 = (~a[6] & n_152)
	n_69 = (~n_56 & ~n_68)
	n_57 = (~n_31 & ~n_56)
	n_80 = (~n_74 & ~n_79)
	n_181 = (~n_154 & n_180)
	n_83 = (~n_67 & ~n_69)
	n_70 = (n_67 & n_69)
	n_64 = (n_57 & ~n_59)
	n_60 = (~n_57 & n_59)
	n_81 = (n_78 & n_80)
	n_108 = (~n_78 & ~n_80)
	n_182 = (~n_153 & n_181)
	n_84 = (~n_70 & ~n_83)
	n_65 = (~n_60 & ~n_64)
	n_61 = (~n_27 & ~n_60)
	n_82 = (~n_74 & ~n_81)
	n_109 = (~n_81 & ~n_108)
	n_87 = (~n_63 & ~n_65)
	n_66 = (n_63 & n_65)
	n_91 = (~n_2 & n_61)
	n_62 = (n_2 & ~n_61)
	n_105 = (n_82 & ~n_84)
	n_85 = (~n_82 & n_84)
	n_147 = (~n_109 & ~n_110)
	n_111 = (n_109 & n_110)
	n_88 = (~n_66 & ~n_87)
	n_92 = (~n_62 & ~n_91)
	n_106 = (~n_85 & ~n_105)
	n_86 = (~n_70 & ~n_85)
	n_148 = (~n_111 & ~n_147)
	n_112 = (~n_104 & ~n_106)
	n_107 = (n_104 & n_106)
	n_101 = (n_86 & ~n_88)
	n_89 = (~n_86 & n_88)
	n_150 = (a[5] & ~n_148)
	n_149 = (~a[5] & n_148)
	n_113 = (~n_107 & ~n_112)
	n_102 = (~n_89 & ~n_101)
	n_90 = (~n_66 & ~n_89)
	n_183 = (~n_150 & n_182)
	n_143 = (~n_111 & ~n_113)
	n_114 = (n_111 & n_113)
	n_116 = (~n_100 & ~n_102)
	n_103 = (n_100 & n_102)
	n_97 = (n_90 & ~n_92)
	n_93 = (~n_90 & n_92)
	n_184 = (~n_149 & n_183)
	n_144 = (~n_114 & ~n_143)
	n_115 = (~n_107 & ~n_114)
	n_117 = (~n_103 & ~n_116)
	n_98 = (~n_93 & ~n_97)
	n_94 = (~n_62 & ~n_93)
	n_146 = (a[4] & ~n_144)
	n_145 = (~a[4] & n_144)
	n_139 = (n_115 & ~n_117)
	n_118 = (~n_115 & n_117)
	n_120 = (~n_96 & ~n_98)
	n_99 = (n_96 & n_98)
	n_124 = (~n_1 & n_94)
	n_95 = (n_1 & ~n_94)
	n_185 = (~n_146 & n_184)
	n_140 = (~n_118 & ~n_139)
	n_119 = (~n_103 & ~n_118)
	n_121 = (~n_99 & ~n_120)
	n_125 = (~n_95 & ~n_124)
	n_186 = (~n_145 & n_185)
	n_142 = (a[3] & ~n_140)
	n_141 = (~a[3] & n_140)
	n_135 = (n_119 & ~n_121)
	n_122 = (~n_119 & n_121)
	n_187 = (~n_142 & n_186)
	n_136 = (~n_122 & ~n_135)
	n_123 = (~n_99 & ~n_122)
	n_188 = (~n_141 & n_187)
	n_138 = (a[2] & ~n_136)
	n_137 = (~a[2] & n_136)
	n_131 = (n_123 & ~n_125)
	n_126 = (~n_123 & n_125)
	n_189 = (~n_138 & n_188)
	n_132 = (~n_126 & ~n_131)
	n_127 = (~n_95 & ~n_126)
	n_190 = (~n_137 & n_189)
	n_134 = (a[1] & ~n_132)
	n_133 = (~a[1] & n_132)
	n_129 = (a[0] & ~n_127)
	n_128 = (~a[0] & n_127)
	n_191 = (~n_134 & n_190)
	n_130 = (~n_128 & ~n_129)
	n_192 = (~n_133 & n_191)
	n_193 = (~n_130 & n_192)
	o_1 = n_193
	return o_1