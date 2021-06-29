from antlr4 import *
from Verilog2001Parser import Verilog2001Parser
from Verilog2001Visitor import Verilog2001Visitor

class verilogVisitor(Verilog2001Visitor):
	def visitModule_declaration(self, ctx: Verilog2001Parser.Module_declarationContext):
		# print("visitModule_decalaration: ", ctx.getText())
		self.visit(ctx.module_identifier())
		self.visit(ctx.list_of_ports())
		z3filecontent = ""
		inp = out2 = aux = var_dec = var_out = ""
		eq = ""
		for i in range(len(ctx.module_item())):
			if ctx.module_item()[i].port_declaration():
				if ctx.module_item()[i].port_declaration().input_declaration():
					inp += self.visit(ctx.module_item()[i])+", "
			if ctx.module_item()[i].port_declaration():
				if ctx.module_item()[i].port_declaration().output_declaration():
					var_out += self.visit(ctx.module_item()[i])
					# print("ouuuuuuuuuuuuut: ",var_out)
			if ctx.module_item()[i].module_or_generate_item():
				if ctx.module_item()[i].module_or_generate_item().module_or_generate_item_declaration():
					aux += self.visit(ctx.module_item()[i]) +"\n"
			if inp and var_out and aux:
				# print("var out **************: ", var_out)
				rinp = inp.split(",")
				out = rinp[-2].replace(" ", "")
				out2 = var_out.split(" = ")[0]
				out1 = out+"_1"
				# print("out: ", out2)
				rinp = " ".join(rinp)
				rinp = rinp.replace("  ", " ")
				# print("rinp: ", rinp)
				# print("rinp joined: ", " ".join(rinp))
				# print("inp: ", inp)
				input_dec = inp[:-2] + " = Bools('" + (rinp[:-1]) + "')"
				# input_dec = input_dec.replace(out, out2)
				output_dec = out2+" = Bool('" + out2 + "')"
				var_dec = input_dec+"\n"+output_dec+"\n"+aux+"\n"+var_out
				# print(var_dec)
			if ctx.module_item()[i].module_or_generate_item():
				if ctx.module_item()[i].module_or_generate_item().continuous_assign():
					# print("haa hai", ctx.module_item()[3].module_or_generate_item().continuous_assign().getText())
					eq += self.visit(ctx.module_item()[i])
					# print("constraints: ", eq[:-2])
		inp = inp[:-2]
		# print("----------------INPUT------------------", inp)
		eq = eq[:-2]
		z3constraint1 = "A = "+out1+" == $$"
		z3constraint2 = "B = "+"(And("+eq+"))"
		z3constraint2 = z3constraint2.replace("("+out+")", "("+out2+")")
		output_dec1 = out1+" = Bool('" + out1 + "')"
		var_dec += "\n"+output_dec1
		z3filecontent = var_dec+"\n"+z3constraint1+"\n"+z3constraint2
		formula = "formula = Implies(And(A,B), "+out1+" == "+out2+")"
		z3filecontent += "\n"+formula+"\nvalid(formula)"
		# print(z3constraint2)
		return z3filecontent, out1, out

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
		# print("visitExpression: ", ctx.getText())
		exp = ""
		# print(ctx.term())
		# for i in range(len(ctx.term()) - 1):
		if ctx.binary_operator():
			# exp += self.visit(ctx.binary_operator()[i]) + "("
			# exp += self.visit(ctx.term()[i]) + ", "
			# exp += self.visit(ctx.term()[i+1]) + ")"
			exp += self.visit(ctx.binary_operator()[0])+"("
			# print("exp: ", exp)
			for i in range(len(ctx.term())):
				exp += self.visit(ctx.term()[i])
				if i < len(ctx.term()):
					exp += ","
			exp += ")"
		if len(ctx.term()) == 1:
			exp = self.visit(ctx.term()[0])
		# else:
		# 	print("ERROR: No Binary Operator Found")
		# print("exp: ", exp)
		return exp
	
	def visitTerm(self, ctx: Verilog2001Parser.TermContext):
		# print("visitTerm: ", ctx.getText())
		# if ctx.primary().mintypmax_expression():
		# 	return self.visit(ctx.primary().mintypmax_expression())
		# else:
		term = ""
		if ctx.unary_operator():
			term = self.visit(ctx.unary_operator())
		term += "("+self.visit(ctx.primary())+")"
		return term
	
	def visitMintypmax_expression(self, ctx: Verilog2001Parser.Mintypmax_expressionContext):
		# print("visitMintypmax_expression: ", ctx.getText())
		# print(len(ctx.expression()))
		minexp = ""
		for i in range(len(ctx.expression())):
			minexp = self.visit(ctx.expression()[i])
		return minexp

	def visitPrimary(self, ctx: Verilog2001Parser.PrimaryContext):
		# print("visitPrimary: ", ctx.getText())
		if ctx.mintypmax_expression():
			return self.visit(ctx.mintypmax_expression())
		elif ctx.number():
			return "Int("+ctx.getText()+")"
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
		elif str(ctx.getText()) == "^":
			op = "Xor"
		return op
	
	def visitModule_or_generate_item_declaration(self, ctx: Verilog2001Parser.Module_or_generate_item_declarationContext):
		# print("visitModule_or_generate_item_declaration: ", ctx.getText())
		return self.visit(ctx.net_declaration())

	def visitNet_declaration(self, ctx: Verilog2001Parser.Net_declarationContext):
		# print("visitNet_decalaration: ", ctx.getText())
		return self.visit(ctx.list_of_net_identifiers())
	
	def visitList_of_net_identifiers(self, ctx: Verilog2001Parser.List_of_net_identifiersContext):
		# print("visitList_of_net_identifiers: ", ctx.getText())
		lst = str(ctx.getText()).split(",")
		if len(lst) > 1:
			return str(ctx.getText()) + " = Bools('" + str(ctx.getText()).replace(",", " ") + "')"
		else:
			return str(ctx.getText()) + " = Bool('" + str(ctx.getText()).replace(",", " ") + "')"

	def visitInput_declaration(self, ctx: Verilog2001Parser.Input_declarationContext):
		# print("visitInput_declaration: ", ctx.getText()[0:5])
		# inps = ""
		inps = self.visit(ctx.list_of_port_identifiers())
		# print("inps: ---=-=-=--", inps)
		lv = ""
		for i in inps:
			lv += i
			if i == " ":
				lv += ","
		# print("lv---------------", lv[:-1])
		# return lv[:-2] + " = " + "Bools('" + inps[:-1]+"')"
		return lv[:-2]
	
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
		return lv[:-2]+"" + " = " + "Bool('" + outs[:-1]+"')"

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