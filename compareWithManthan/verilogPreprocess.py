import antlr4
from rectify_spec_file import replace_preref
from compareWithManthan.verilogPreprocessVisitor import verilogVisitor
from compareWithManthan.Verilog2001Lexer import Verilog2001Lexer
from compareWithManthan.Verilog2001Parser import Verilog2001Parser

def verilog_preprocess(verilog_spec):
	'''
	Input: verilog file
	Output: preprocessed verilog file
	Functionality: Converts verilog into 2 argument format
	'''
	
	filename = verilog_spec
	f = open("compareWithManthan/verilog/"+filename, "r")
	data = f.read()
	f.close()
	data = data.replace(".", "")
	inputStream = antlr4.InputStream(data)
	lexer = Verilog2001Lexer(inputStream)
	tokenStream = antlr4.CommonTokenStream(lexer)
	parser = Verilog2001Parser(tokenStream)
	tree = parser.module_declaration()
	visitor = verilogVisitor(verilog_spec)
	preprocessed_verilog = visitor.visit(tree)
	f = open("compareWithManthan/verilog/"+verilog_spec.split(".v")[0]+"_preprocessed.v", "w")
	f.write(preprocessed_verilog)
	f.close()
