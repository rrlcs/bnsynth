import antlr4

from benchmarks.Verilog2001Lexer import Verilog2001Lexer
from benchmarks.Verilog2001Parser import Verilog2001Parser
from benchmarks.verilogPreprocessVisitor import verilogVisitor


def preprocess(verilog_spec, verilog_spec_location):
	'''
	Input: verilog file
	Output: verilog file with resolved dependencies
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
	visitor = verilogVisitor(verilog_spec, verilog_spec_location)
	orderedVerilog, pyfilecontent, num_out_vars, num_of_vars, output_var_idx, io_dict, num_of_eqns = visitor.visit(tree)
	f = open("data_preparation_and_result_checking/"+verilog_spec_location+"/"+verilog_spec, "w")
	f.write(orderedVerilog)
	f.close()

	filename = filename.replace(".", "")
	# f = open("python_specs/"+filename+".py", "w")
	# f.write(pyfilecontent)
	# f.close()

	return num_of_vars, num_out_vars, output_var_idx, io_dict, num_of_eqns, filename
