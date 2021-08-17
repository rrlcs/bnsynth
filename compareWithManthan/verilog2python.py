import os
import antlr4
import argparse
from rectify_spec_file import replace_preref
from compareWithManthan.verilogToPythonVisitor import verilogVisitor
from compareWithManthan.Verilog2001Lexer import Verilog2001Lexer
from compareWithManthan.Verilog2001Parser import Verilog2001Parser

# Add proper comment
def build_spec(verilog_spec):
	'''
	Input: verilog file
	Output: Python Function
	Functionality: Converts verilog (decalrative) to python (imperative)
	'''
	
	filename = verilog_spec.split(".v")[0]+"_preprocessed.v"
	f = open("compareWithManthan/verilog/"+filename, "r")
	data = f.read()
	data = data.replace(".", "")
	inputStream = antlr4.InputStream(data)
	lexer = Verilog2001Lexer(inputStream)
	tokenStream = antlr4.CommonTokenStream(lexer)
	parser = Verilog2001Parser(tokenStream)
	tree = parser.module_declaration()
	visitor = verilogVisitor(verilog_spec)
	F, num_out_vars, num_of_vars, output_var_idx, io_dict = visitor.visit(tree)
	F = replace_preref(F)
	f = open("func_spec.py", "w")
	f.write(F)
	f.close()

	return F, num_of_vars, num_out_vars, output_var_idx, io_dict
