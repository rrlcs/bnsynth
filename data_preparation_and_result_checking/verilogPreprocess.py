import antlr4
from data_preparation_and_result_checking.verilogPreprocessVisitor import verilogVisitor
from data_preparation_and_result_checking.Verilog2001Lexer import Verilog2001Lexer
from data_preparation_and_result_checking.Verilog2001Parser import Verilog2001Parser

def verilog_preprocess(verilog_spec, verilog_spec_location):
	'''
	Input: verilog file
	Output: preprocessed verilog file
	Functionality: Converts verilog into 2 argument format
	'''
	
	filename = verilog_spec
	f = open("data_preparation_and_result_checking/"+verilog_spec_location+"/"+filename, "r")
	data = f.read()
	f.close()
	data = data.replace(".", "")
	inputStream = antlr4.InputStream(data)
	lexer = Verilog2001Lexer(inputStream)
	tokenStream = antlr4.CommonTokenStream(lexer)
	parser = Verilog2001Parser(tokenStream)
	tree = parser.module_declaration()
	visitor = verilogVisitor()
	preprocessed_verilog = visitor.visit(tree)
	f = open("data_preparation_and_result_checking/"+verilog_spec_location+"/"+verilog_spec.split(".v")[0]+"_preprocessed.v", "w")
	f.write(preprocessed_verilog)
	f.close()
