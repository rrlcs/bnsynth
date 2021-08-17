from antlr4 import *
from Verilog2001Parser import Verilog2001Parser
from Verilog2001Visitor import Verilog2001Visitor

# global outvar
class verilogVisitor(Verilog2001Visitor):
	def visitModule_declaration(self, ctx: Verilog2001Parser.Module_declarationContext):
		# print("visitModule_decalaration: ", ctx.module_keyword().getText())
		# print("output variable", outvar)
		self.visit(ctx.module_identifier())
		self.visit(ctx.list_of_ports())
		z3filecontent = ""
		inp = out2 = aux = var_dec = ""
		eq = ""
		for i in range(len(ctx.module_item())):
			if ctx.module_item()[i].port_declaration():
				if ctx.module_item()[i].port_declaration().input_declaration():
					inp = self.visit(ctx.module_item()[i])
			if ctx.module_item()[i].port_declaration():
				if ctx.module_item()[i].port_declaration().output_declaration():
					out2 = self.visit(ctx.module_item()[i])
			if ctx.module_item()[i].module_or_generate_item():
				if ctx.module_item()[i].module_or_generate_item().module_or_generate_item_declaration():
					aux = self.visit(ctx.module_item()[i])
			if inp and out2 and aux:
				var_dec = inp+"\n"+out2+"\n"+aux
			# 	print(var_dec)
			if ctx.module_item()[i].module_or_generate_item():
				if ctx.module_item()[i].module_or_generate_item().continuous_assign():
					# print("haa hai", ctx.module_item()[3].module_or_generate_item().continuous_assign().getText())
					eq += self.visit(ctx.module_item()[i])
					# print("constraints: ", eq[:-2])
		eq = eq[:-2]
		outvar2 = out2.split("=")[0]
		outvar = outvar2[:-3]
		outvar1 = outvar2[:-2]+"1"
		z3constraint1 = "A = "+outvar1+" == $$"
		z3constraint2 = "B = "+"(And("+eq+"))"
		out1 = out2.replace(outvar2[:-1], outvar1)
		z3constraint2 = z3constraint2.replace(outvar, outvar2)
		var_dec += "\n"+out1
		z3filecontent = var_dec+"\n"+z3constraint1+"\n"+z3constraint2
		formula = "formula = Implies(And(A,B), "+outvar1+" == "+outvar2+")"
		z3filecontent += "\n"+formula+"\nvalid(formula)"
		return z3filecontent

	def visitModule_identifier(self, ctx: Verilog2001Parser.Module_identifierContext):
		# print("visitModule_identifier: ", ctx.getText())
		return

	def visitList_of_ports(self, ctx: Verilog2001Parser.List_of_portsContext):
		# print("visitList_of_ports: ", ctx.getText())
		return

	def visitModule_item(self, ctx: Verilog2001Parser.Module_itemContext):
		# print("visitModule_item: ", ctx.getText())
		if ctx.port_declaration():
			return self.visit(ctx.port_declaration())
		elif ctx.module_or_generate_item():
			return self.visit(ctx.module_or_generate_item())

	def visitPort_declaration(self, ctx: Verilog2001Parser.Port_declarationContext):
		# print("visitPort_decalaration: ", ctx.getText())
		if ctx.input_declaration():
			return self.visit(ctx.input_declaration())
		elif ctx.output_declaration():
			return self.visit(ctx.output_declaration())
	
	def visitModule_or_generate_item(self, ctx: Verilog2001Parser.Module_or_generate_itemContext):
		# print("visitModule_or_generate_item: ", ctx.getText())
		if ctx.module_or_generate_item_declaration():
			return self.visit(ctx.module_or_generate_item_declaration())
		elif ctx.continuous_assign():
			return self.visit(ctx.continuous_assign())
	
	def visitContinuous_assign(self, ctx: Verilog2001Parser.Continuous_assignContext):
		# print("visitContinuous_assign: ", ctx.getText()[0:6])
		return self.visit(ctx.list_of_net_assignments())

	def visitList_of_net_assignments(self, ctx: Verilog2001Parser.List_of_net_assignmentsContext):
		# print("visitList_of_net_assignments: ", ctx.getText())
		return self.visit(ctx.net_assignment()[0])

	def visitNet_assignment(self, ctx: Verilog2001Parser.Net_assignmentContext):
		# print("visitNet_assignment: ", ctx.getText())
		lv = self.visit(ctx.net_lvalue())
		rv = self.visit(ctx.expression())
		return "(" + lv + " == " + rv + ")" + ", "

	def visitNet_lvalue(self, ctx: Verilog2001Parser.Net_lvalueContext):
		# print("visitNet_lvalue:, ", ctx.getText())
		return str(ctx.getText())
	
	def visitExpression(self, ctx: Verilog2001Parser.ExpressionContext):
		# print("visitExpression:, ", ctx.getText())
		exp = ""
		if len(ctx.binary_operator()):
			# print("no bin op: ", len(ctx.binary_operator()))
			# print("no terma: ", len(ctx.term()))
			# for i in range(len(ctx.binary_operator())):
			exp += self.visit(ctx.binary_operator()[0])+"("
			# print("exp: ", exp)
			for i in range(len(ctx.term())):
				exp += self.visit(ctx.term()[i])
				if i < len(ctx.term()):
					exp += ","
			exp += ")"
			# exp += self.visit(ctx.term()[1]) + ")"
			# if len(ctx.term()) > 2:
			# 	print("ctx term 2", ctx.term()[2])
			# if len(ctx.binary_operator()) > 1:
			# 	print("ctx bin op 2: ", ctx.binary_operator()[2])
		else:
			print("ERROR: No Binary Operator Found")
		# print("exp: ", exp)
		return exp
	
	def visitTerm(self, ctx: Verilog2001Parser.TermContext):
		# print("visitTerem: ", ctx.getText())
		term = ""
		if ctx.unary_operator():
			term = self.visit(ctx.unary_operator())
		term += "("+self.visit(ctx.primary())+")"
		return term

	def visitPrimary(self, ctx: Verilog2001Parser.PrimaryContext):
		# print("visitPrimary: ", ctx.getText())
		return str(ctx.getText())
	
	def visitUnary_operator(self, ctx: Verilog2001Parser.Unary_operatorContext):
		# print("visitUnary_operator: ", ctx.getText())
		if str(ctx.getText()) == "~":
			op = "Not"
		return op
	
	def visitBinary_operator(self, ctx: Verilog2001Parser.Binary_operatorContext):
		# print("visitBinary_operator: ", ctx.getText())
		if str(ctx.getText()) == "&":
			op = "And"
		elif str(ctx.getText()) == "|":
			op = "Or"
		return op
	
	def visitModule_or_generate_item_declaration(self, ctx: Verilog2001Parser.Module_or_generate_item_declarationContext):
		# print("visitModule_or_generate_item_declaration: ", ctx.getText())
		return self.visit(ctx.net_declaration())

	def visitNet_declaration(self, ctx: Verilog2001Parser.Net_declarationContext):
		# print("visitNet_decalaration: ", ctx.getText())
		return self.visit(ctx.list_of_net_identifiers())
	
	def visitList_of_net_identifiers(self, ctx: Verilog2001Parser.List_of_net_identifiersContext):
		# print("visitList_of_net_identifiers: ", ctx.getText())
		return str(ctx.getText()) + " = Bools('" + str(ctx.getText()).replace(",", " ") + "')"

	def visitInput_declaration(self, ctx: Verilog2001Parser.Input_declarationContext):
		# print("visitInput_declaration: ", ctx.getText()[0:5])
		inps = self.visit(ctx.list_of_port_identifiers())
		# print("inps: ---=-=-=--", inps)
		lv = ""
		for i in inps:
			lv += i
			if i == " ":
				lv += ","
		# print("lv---------------", lv[:-1])
		return lv[:-2] + " = " + "Bools('" + inps[:-1]+"')"
	
	def visitOutput_declaration(self, ctx: Verilog2001Parser.Output_declarationContext):
		# print("visitOutput_decalaration: ", ctx.getText()[0:6])
		outs = self.visit(ctx.list_of_port_identifiers())
		# print("outs: ---=-=-=--", outs[:-1])
		# outvar = outs[:-1]
		lv = ""
		for i in outs:
			lv += i
			if i == " ":
				lv += ","
		# print("lv---------------", lv[:-1])
		return lv[:-2]+"_2" + " = " + "Bool('" + outs[:-1]+"_2')"

	def visitList_of_port_identifiers(self, ctx: Verilog2001Parser.List_of_port_identifiersContext):
		# print("visitList_of_port_identifiers: ", ctx.getText())
		ids = ""
		for i in range(len(ctx.port_identifier())):
			ids += self.visit(ctx.port_identifier()[i]) + " "
		# print("ids: ", ids)
		return ids
		
	def visitPort_identifier(self, ctx: Verilog2001Parser.Port_identifierContext):
		# print("visitPort_identifier: ", ctx.getText())
		return str(ctx.getText())