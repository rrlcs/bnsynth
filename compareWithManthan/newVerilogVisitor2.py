from antlr4 import *
from compareWithManthan.Verilog2001Parser import Verilog2001Parser
from compareWithManthan.Verilog2001Visitor import Verilog2001Visitor

class verilogVisitor(Verilog2001Visitor):
	def __init__(self, spec) -> None:
		super().__init__()
		self.spec = spec
	def visitModule_declaration(self, ctx: Verilog2001Parser.Module_declarationContext):
		filename = "sample"+str(self.spec)+"_varstoelim.txt"
		f = open("compareWithManthan/sample_examples/Yvarlist/"+filename, "r")
		output = f.read()
		output_vars = output.split("\n")[:-1]
		num_out_vars = len(output_vars)
		# print(num_out_vars)
		self.visit(ctx.module_identifier())
		self.visit(ctx.list_of_ports())
		z3filecontent = ""
		inp = aux = var_dec = var_out = ""
		eq = ""
		eqn = []
		for i in range(len(ctx.module_item())):
			if ctx.module_item()[i].port_declaration():
				if ctx.module_item()[i].port_declaration().input_declaration():
					inp += self.visit(ctx.module_item()[i])
					if i < len(ctx.module_item()) - 1:
						inp += ", "
			if ctx.module_item()[i].port_declaration():
				if ctx.module_item()[i].port_declaration().output_declaration():
					var_out += self.visit(ctx.module_item()[i])
			if ctx.module_item()[i].module_or_generate_item():
				if ctx.module_item()[i].module_or_generate_item().module_or_generate_item_declaration():
					aux += self.visit(ctx.module_item()[i]) +"\n"
			if inp and var_out:
				rinp = inp.split(",")[:-1]
				io_vars = rinp
				num_of_vars = len(rinp)
				rinp = " ".join(rinp)
				rinp = rinp.replace("  ", " ")
				input_dec = inp[:-2] + " = Bools('" + (rinp[:-1]) + "')"
				var_dec = input_dec+"\n"+aux+"\n"+var_out
			if ctx.module_item()[i].module_or_generate_item():
				if ctx.module_item()[i].module_or_generate_item().continuous_assign():
					const = self.visit(ctx.module_item()[i])[1:-2]+"\n"
					eqn.append(const)
		# print(io_vars)
		io_dict = {}
		# i_0 = inp_vars[]
		var_defs = []
		for index, value in enumerate(io_vars):
			io_dict[index] = value
			value  = "	"+str(value)+" = "+"inp_vars["+str(index)+", :]"
			var_defs.append(value)

		var_defs = "\n".join(var_defs)
		# print(var_defs)
		# print(io_dict)
		eqn = ["	"+i[:-3]+"name)" for i in eqn]
		# print(eqn)
		# print(input_dec)
		eq = '\n'.join(eqn)
		# print("eq",eq)
		inp = inp[:-2]
		func_def = "def F(inp_vars, name, util):\n"
		z3filecontent = func_def+var_defs+"\n"+eq+"\n"+"	return out"
		return z3filecontent, num_out_vars, num_of_vars

	def visitModule_identifier(self, ctx: Verilog2001Parser.Module_identifierContext):
		return

	def visitList_of_ports(self, ctx: Verilog2001Parser.List_of_portsContext):
		return

	def visitModule_item(self, ctx: Verilog2001Parser.Module_itemContext):
		if ctx.port_declaration():
			return self.visit(ctx.port_declaration())
		elif ctx.module_or_generate_item():
			return self.visit(ctx.module_or_generate_item())

	def visitPort_declaration(self, ctx: Verilog2001Parser.Port_declarationContext):
		if ctx.input_declaration():
			return self.visit(ctx.input_declaration())
		elif ctx.output_declaration():
			return self.visit(ctx.output_declaration())
	
	def visitModule_or_generate_item(self, ctx: Verilog2001Parser.Module_or_generate_itemContext):
		if ctx.module_or_generate_item_declaration():
			return self.visit(ctx.module_or_generate_item_declaration())
		elif ctx.continuous_assign():
			return self.visit(ctx.continuous_assign())
	
	def visitContinuous_assign(self, ctx: Verilog2001Parser.Continuous_assignContext):
		return self.visit(ctx.list_of_net_assignments())

	def visitList_of_net_assignments(self, ctx: Verilog2001Parser.List_of_net_assignmentsContext):
		return self.visit(ctx.net_assignment()[0])

	def visitNet_assignment(self, ctx: Verilog2001Parser.Net_assignmentContext):
		lv = self.visit(ctx.net_lvalue())
		rv = self.visit(ctx.expression())
		return "(" + lv + " = " + rv + ")" + ", "

	def visitNet_lvalue(self, ctx: Verilog2001Parser.Net_lvalueContext):
		return str(ctx.getText())
	
	def visitExpression(self, ctx: Verilog2001Parser.ExpressionContext):
		exp = ""
		if ctx.binary_operator():

			exp += self.visit(ctx.binary_operator()[0])+"("
			for i in range(len(ctx.term())):
				exp += self.visit(ctx.term()[i])
				if i < len(ctx.term()):
					exp += ","
			exp += ")"
		if len(ctx.term()) == 1:
			exp = self.visit(ctx.term()[0])
		return exp
	
	def visitTerm(self, ctx: Verilog2001Parser.TermContext):
		term = ""
		if ctx.unary_operator():
			term = self.visit(ctx.unary_operator())
		term += "("+self.visit(ctx.primary())+")"
		return term
	
	def visitMintypmax_expression(self, ctx: Verilog2001Parser.Mintypmax_expressionContext):
		minexp = ""
		for i in range(len(ctx.expression())):
			minexp = self.visit(ctx.expression()[i])
		return minexp

	def visitPrimary(self, ctx: Verilog2001Parser.PrimaryContext):
		if ctx.mintypmax_expression():
			return self.visit(ctx.mintypmax_expression())
		elif ctx.number():
			return "Int("+ctx.getText()+")"
		return str(ctx.getText())
	
	def visitUnary_operator(self, ctx: Verilog2001Parser.Unary_operatorContext):
		if str(ctx.getText()) == "~":
			op = "Not"
		else:
			op = ""
		return op
	
	def visitBinary_operator(self, ctx: Verilog2001Parser.Binary_operatorContext):
		if str(ctx.getText()) == "&":
			op = "util.tnorm_vectorized"
		elif str(ctx.getText()) == "|":
			op = "util.tconorm_vectorized"
		elif str(ctx.getText()) == "^":
			op = "util.continuous_xor"
		return op
	
	def visitModule_or_generate_item_declaration(self, ctx: Verilog2001Parser.Module_or_generate_item_declarationContext):
		return self.visit(ctx.net_declaration())

	def visitNet_declaration(self, ctx: Verilog2001Parser.Net_declarationContext):
		return self.visit(ctx.list_of_net_identifiers())
	
	def visitList_of_net_identifiers(self, ctx: Verilog2001Parser.List_of_net_identifiersContext):
		lst = str(ctx.getText()).split(",")
		if len(lst) > 1:
			return str(ctx.getText()) + " = Bools('" + str(ctx.getText()).replace(",", " ") + "')"
		else:
			return str(ctx.getText()) + " = Bool('" + str(ctx.getText()).replace(",", " ") + "')"

	def visitInput_declaration(self, ctx: Verilog2001Parser.Input_declarationContext):
		inps = self.visit(ctx.list_of_port_identifiers())
		lv = ""
		for i in inps:
			lv += i
			if i == " ":
				lv += ","
		return lv[:-2]
	
	def visitOutput_declaration(self, ctx: Verilog2001Parser.Output_declarationContext):
		outs = self.visit(ctx.list_of_port_identifiers())
		lv = ""
		for i in outs:
			lv += i
			if i == " ":
				lv += ","
		return lv[:-2]+"" + " = " + "Bool('" + outs[:-1]+"')"

	def visitList_of_port_identifiers(self, ctx: Verilog2001Parser.List_of_port_identifiersContext):
		ids = ""
		for i in range(len(ctx.port_identifier())):
			ids += self.visit(ctx.port_identifier()[i]) + " "
		return ids
		
	def visitPort_identifier(self, ctx: Verilog2001Parser.Port_identifierContext):
		return str(ctx.getText())