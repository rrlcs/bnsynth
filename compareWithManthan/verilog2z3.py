import os
import antlr4
import argparse
from compareWithManthan.verilogToZ3Visitor import verilogVisitor
from compareWithManthan.Verilog2001Lexer import Verilog2001Lexer
from compareWithManthan.Verilog2001Parser import Verilog2001Parser

def preparez3(verilog_spec, output_var_idx):
	'''
	Input: verilog file
	Output: z3py equivalent of verilog file
	Functionality: Parses the verilog file and converts it to z3py format.
					It does the same for the NN output as well.
	'''

	filename = verilog_spec+"_preprocessed.v"
	f = open("compareWithManthan/verilog/"+filename, "r")
	data = f.read()
	inputStream = antlr4.InputStream(data)
	lexer = Verilog2001Lexer(inputStream)
	tokenStream = antlr4.CommonTokenStream(lexer)
	parser = Verilog2001Parser(tokenStream)
	tree = parser.module_declaration()
	visitor = verilogVisitor(verilog_spec)
	z3filecontent = visitor.visit(tree)
	with open('compareWithManthan/templateZ3Checker.py', 'r') as file :
		filedata = file.read()
	filedata = filedata.replace('#*#', z3filecontent)
	with open('compareWithManthan/z3ValidityChecker.py', 'w') as file:
		file.write(filedata)

	# Parse the NN output and Generate Z3Py Format
	f = open("nn_output", "r")
	data = f.read()
	data = data.split("\n")
	for i in range(len(data)):
		inputStream = antlr4.InputStream(data[i])
		lexer = Verilog2001Lexer(inputStream)
		tokenStream = antlr4.CommonTokenStream(lexer)
		parser = Verilog2001Parser(tokenStream)
		tree = parser.expression()
		visitor = verilogVisitor(verilog_spec)
		nnOut = visitor.visit(tree)
		with open('compareWithManthan/z3ValidityChecker.py', 'r') as file :
			filedata = file.read()
		filedata = filedata.replace('$$'+str(i), nnOut)
		with open('compareWithManthan/z3ValidityChecker.py', 'w') as file:
			file.write(filedata)
	
