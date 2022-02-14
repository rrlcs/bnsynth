import antlr4

from benchmarks.Verilog2001Lexer import Verilog2001Lexer
from benchmarks.Verilog2001Parser import Verilog2001Parser
from benchmarks.verilogToPythonVisitor import verilogVisitor


# Add proper comment
def build_spec(verilog_spec, verilog_spec_location):
	'''
	Input: verilog file
	Output: Python Function
	Functionality: Converts verilog (decalrative) to python (imperative)
	'''
	
	filename = verilog_spec
	f = open("benchmarks/"+verilog_spec_location+"/"+filename, "r")
	data = f.read()
	data = data.replace(".", "")
	inputStream = antlr4.InputStream(data)
	lexer = Verilog2001Lexer(inputStream)
	tokenStream = antlr4.CommonTokenStream(lexer)
	parser = Verilog2001Parser(tokenStream)
	tree = parser.module_declaration()
	visitor = verilogVisitor()
	F = visitor.visit(tree)
	filename = filename.replace(".", "")
	f = open("python_specs/"+filename+".py", "w")
	f.write(F)
	f.close()
	return
