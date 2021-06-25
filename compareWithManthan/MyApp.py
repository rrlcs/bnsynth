import os
import antlr4

from verilogVisitor import verilogVisitor
from expressionVisitor import expressionVisitor
from Verilog2001Lexer import Verilog2001Lexer
from Verilog2001Parser import Verilog2001Parser
from Verilog2001Visitor import Verilog2001Visitor

if __name__ == "__main__":
	f = open("./compareWithManthan/3xor_skolem.v", "r")
	data = f.read()
	inputStream = antlr4.InputStream(data)
	lexer = Verilog2001Lexer(inputStream)
	tokenStream = antlr4.CommonTokenStream(lexer)
	parser = Verilog2001Parser(tokenStream)
	tree = parser.module_declaration()
	visitor = verilogVisitor()
	z3filecontent = visitor.visit(tree)
	# print(z3filecontent)

	with open('./compareWithManthan/templateZ3Checker.py', 'r') as file :
		filedata = file.read()
	filedata = filedata.replace('#*#', z3filecontent)
	with open('./compareWithManthan/z3ValidityChecker.py', 'w') as file:
		file.write(filedata)

	f = open("./compareWithManthan/test", "r")
	data = f.read()
	inputStream = antlr4.InputStream(data)
	lexer = Verilog2001Lexer(inputStream)
	tokenStream = antlr4.CommonTokenStream(lexer)
	parser = Verilog2001Parser(tokenStream)
	tree = parser.expression()
	visitor = expressionVisitor()
	nnOut = visitor.visit(tree)

	with open('./compareWithManthan/z3ValidityChecker.py', 'r') as file :
		filedata = file.read()
	filedata = filedata.replace('$$', nnOut)
	with open('./compareWithManthan/z3ValidityChecker.py', 'w') as file:
		file.write(filedata)
	
	os.system("python ./compareWithManthan/z3ValidityChecker.py")
	# # print("==== nn out ====", nnOut)
