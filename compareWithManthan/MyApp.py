import os
import antlr4
import argparse
from compareWithManthan.newVerilogVisitor import verilogVisitor
from compareWithManthan.Verilog2001Lexer import Verilog2001Lexer
from compareWithManthan.Verilog2001Parser import Verilog2001Parser

def preparez3(spec, output_var_idx):
	# Select spec file to parse
	# parser = argparse.ArgumentParser()
	# parser.add_argument("--spec", type=int, default=1, help="Enter values from 1 to 5")
	# args = parser.parse_args()
	filename = "sample"+str(spec)+".v"

	# Parse Skolem Function from Manthan and
	# Generate Z3Py Format
	f = open("compareWithManthan/sample_examples/"+filename, "r")
	data = f.read()
	inputStream = antlr4.InputStream(data)
	lexer = Verilog2001Lexer(inputStream)
	tokenStream = antlr4.CommonTokenStream(lexer)
	parser = Verilog2001Parser(tokenStream)
	tree = parser.module_declaration()
	visitor = verilogVisitor(spec)
	z3filecontent = visitor.visit(tree)
	with open('compareWithManthan/templateZ3Checker.py', 'r') as file :
		filedata = file.read()
	filedata = filedata.replace('#*#', z3filecontent)
	with open('compareWithManthan/z3ValidityChecker.py', 'w') as file:
		file.write(filedata)

	# Parse the NN output and Generate Z3Py Format
	f = open("nn_output", "r")
	data = f.read()
	# print("data: ", data)
	data = data.split("\n")
	# print(data)
	for i in range(len(data)):
		inputStream = antlr4.InputStream(data[i])
		lexer = Verilog2001Lexer(inputStream)
		tokenStream = antlr4.CommonTokenStream(lexer)
		parser = Verilog2001Parser(tokenStream)
		tree = parser.expression()
		visitor = verilogVisitor(spec)
		nnOut = visitor.visit(tree)
		with open('compareWithManthan/z3ValidityChecker.py', 'r') as file :
			filedata = file.read()
		filedata = filedata.replace('$$'+str(i), nnOut)
		with open('compareWithManthan/z3ValidityChecker.py', 'w') as file:
			file.write(filedata)
	
