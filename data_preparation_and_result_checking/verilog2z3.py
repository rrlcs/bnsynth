import sys

import antlr4

from data_preparation_and_result_checking.Verilog2001Lexer import \
    Verilog2001Lexer
from data_preparation_and_result_checking.Verilog2001Parser import \
    Verilog2001Parser
from data_preparation_and_result_checking.verilogToZ3Visitor import \
    verilogVisitor


def preparez3(verilog_spec, verilog_spec_location, num_of_ouputs):
	'''
	Input: verilog file
	Output: z3py equivalent of verilog file
	Functionality: Parses the verilog file and converts it to z3py format.
					It does the same for the NN output as well.
	'''

	filename = verilog_spec
	f = open("data_preparation_and_result_checking/"+verilog_spec_location+"/"+filename, "r")
	data = f.read()
	inputStream = antlr4.InputStream(data)
	lexer = Verilog2001Lexer(inputStream)
	tokenStream = antlr4.CommonTokenStream(lexer)
	parser = Verilog2001Parser(tokenStream)
	tree = parser.module_declaration()
	visitor = verilogVisitor(verilog_spec, verilog_spec_location, num_of_ouputs)
	z3filecontent = visitor.visit(tree)
	with open('data_preparation_and_result_checking/templateZ3Checker.py', 'r') as file :
		filedata = file.read()
		file.close()
	filedata = filedata.replace('#*#', z3filecontent)
	with open('data_preparation_and_result_checking/z3ValidityChecker.py', 'w') as file:
		file.write(filedata)
		file.close()

	# Parse the NN output and Generate Z3Py Format
	f = open("nn_output", "r")
	data = f.read()
	data = data.split("\n")
	sys.setrecursionlimit(2000)
	for i in range(len(data)):
		inputStream = antlr4.InputStream(data[i])
		lexer = Verilog2001Lexer(inputStream)
		tokenStream = antlr4.CommonTokenStream(lexer)
		parser = Verilog2001Parser(tokenStream)
		tree = parser.mintypmax_expression()
		visitor = verilogVisitor(verilog_spec, verilog_spec_location, num_of_ouputs)
		nnOut = visitor.visit(tree)
		with open('data_preparation_and_result_checking/z3ValidityChecker.py', 'r') as file :
			filedata = file.read()
			file.close()
		text = "$$"+str(i)
		print("text {}, nnout {} ".format(text, nnOut))
		filedata = filedata.replace(text, nnOut)
		# filedata = replace_preref(filedata)
		with open('data_preparation_and_result_checking/z3ValidityChecker.py', 'w') as file:
			file.write(filedata)
			file.close()
	
