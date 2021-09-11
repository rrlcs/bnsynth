import antlr4
from data_preparation_and_result_checking.verilogToPythonVisitor import verilogVisitor
from data_preparation_and_result_checking.Verilog2001Lexer import Verilog2001Lexer
from data_preparation_and_result_checking.Verilog2001Parser import Verilog2001Parser

# Add proper comment
def build_spec(verilog_spec, verilog_spec_location):
	'''
	Input: verilog file
	Output: Python Function
	Functionality: Converts verilog (decalrative) to python (imperative)
	'''
	
	filename = verilog_spec
	f = open("data_preparation_and_result_checking/"+verilog_spec_location+"/"+filename, "r")
	data = f.read()
	data = data.replace(".", "")
	inputStream = antlr4.InputStream(data)
	lexer = Verilog2001Lexer(inputStream)
	tokenStream = antlr4.CommonTokenStream(lexer)
	parser = Verilog2001Parser(tokenStream)
	tree = parser.module_declaration()
	visitor = verilogVisitor(verilog_spec, verilog_spec_location)
	F, num_out_vars, num_of_vars, output_var_idx, io_dict, num_of_eqns = visitor.visit(tree)
	# f = open("data_preparation_and_result_checking/"+verilog_spec_location+"/"+filename+".py", "w")
	filename = filename.replace(".", "")
	f = open("python_specs/"+filename+".py", "w")
	f.write(F)
	f.close()

	return F, num_of_vars, num_out_vars, output_var_idx, io_dict, num_of_eqns, filename
