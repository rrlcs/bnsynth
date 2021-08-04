import os
import antlr4
import argparse
from compareWithManthan.newVerilogVisitor2 import verilogVisitor
from compareWithManthan.Verilog2001Lexer import Verilog2001Lexer
from compareWithManthan.Verilog2001Parser import Verilog2001Parser

def build_spec(spec):
	filename = "sample"+str(spec)+".v"

	# Parse Skolem Function from Manthan and
	# Generate Z3Py Format
	f = open("compareWithManthan/sample_examples/"+filename, "r")
	data = f.read()
	data = data.replace(".", "")
	inputStream = antlr4.InputStream(data)
	lexer = Verilog2001Lexer(inputStream)
	tokenStream = antlr4.CommonTokenStream(lexer)
	parser = Verilog2001Parser(tokenStream)
	tree = parser.module_declaration()
	visitor = verilogVisitor(spec)
	F, num_out_vars, num_of_vars = visitor.visit(tree)
	# print(num_of_vars, num_out_vars)
	f = open("func_spec.py", "w")
	f.write(F)
	f.close()
	return F, num_of_vars, num_out_vars


# if __name__ == "__main__":
# 	# Select spec file to parse
# 	parser = argparse.ArgumentParser()
# 	parser.add_argument("--spec", type=int, default=1, help="Enter values from 1 to 5")
# 	args = parser.parse_args()
# 	F, num_of_vars, num_out_vars = build_spec(args.spec)