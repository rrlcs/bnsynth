from antlr4 import *
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import OrderedDict
from typing import OrderedDict
from data_preparation_and_result_checking.Verilog2001Parser import Verilog2001Parser
from data_preparation_and_result_checking.Verilog2001Visitor import Verilog2001Visitor

class verilogVisitor(Verilog2001Visitor):
	def __init__(self, verilog_spec, verilog_spec_location):
		self.verilog_spec = verilog_spec
		self.verilog_spec_location = verilog_spec_location
		self.G = nx.DiGraph()
		self.eqn_dict = {}
		self.source = ""
		self.input_vars = []
	def visitModule_declaration(self, ctx: Verilog2001Parser.Module_declarationContext):
		
		# For Python Spec
		if "preprocessed" in self.verilog_spec:
			filename = self.verilog_spec.split("_preprocessed.v")[0]+"_varstoelim.txt"
		else:
			filename = self.verilog_spec.split(".v")[0]+"_varstoelim.txt"
		f = open("data_preparation_and_result_checking/"+self.verilog_spec_location+"/Yvarlist/"+filename, "r")
		output = f.read()
		output_vars = output.split("\n")[:-1]
		num_out_vars = len(output_vars)

		# For ordered verilog
		module = "module "
		module += self.visit(ctx.module_identifier())
		io = self.visit(ctx.list_of_ports())+";"

		inp = wires = var_out = ""
		eqn = []
		for i in range(len(ctx.module_item())):
			if ctx.module_item()[i].port_declaration():
				if ctx.module_item()[i].port_declaration().input_declaration():
					inp += self.visit(ctx.module_item()[i])
					if i < len(ctx.module_item()) - 1:
						inp += ", "
			if ctx.module_item()[i].port_declaration():
				if ctx.module_item()[i].port_declaration().output_declaration():
					var_out = self.visit(ctx.module_item()[i])
			if ctx.module_item()[i].module_or_generate_item():
				if ctx.module_item()[i].module_or_generate_item().module_or_generate_item_declaration():
					wires += self.visit(ctx.module_item()[i]) +"\n"
			if inp:
				inps = inp.split(", ")[:-1]
				rinp = inp.split(",")[:-1]
				io_vars = rinp
				num_of_vars = len(rinp)
			if ctx.module_item()[i].module_or_generate_item():
				if ctx.module_item()[i].module_or_generate_item().continuous_assign():
					constr = self.visit(ctx.module_item()[i])[:]+"\n"
					eqn.append(constr)

		# Plot the DAG
		nx.draw(self.G, with_labels=True, node_size=700)
		plt.savefig("dag.png")
		plt.show()
		plt.close()

		# Constructing Ordered Verilog
		wires = wires.split("\n")[:-1]
		out = "output "+var_out
		self.input_vars = list(np.array(self.input_vars).flatten())
		self.input_vars = list(filter(''.__ne__, self.input_vars))
		wires = ["	wire "+a+";" for a in wires]
		wires = "\n".join(wires)
		topological_sort = list(nx.topological_sort(self.G))
		lvalues = [x for x in topological_sort if x not in self.input_vars]
		ordered_eqns = [self.eqn_dict[lv] for lv in lvalues if lv in self.eqn_dict]
		ordered_eqns1 = '\n'.join(["	"+"assign "+i[:-1]+");" for i in ordered_eqns])
		inps = "\n".join(["	input "+i+";" for i in inps])
		orderedVerilog = module+io+"\n"+inps+"\n"+wires+"\n"+"	"+out+";\n"+ordered_eqns1+"\n"+"endmodule"+"\n"
		
		# Constructing Python Spec
		io_vars = [s.lstrip() for s in io_vars]
		io_vars = [s.rstrip() for s in io_vars]
		io_dict = {}
		var_defs = []
		for index, value in enumerate(io_vars):
			io_dict[index] = value
			value  = "	"+str(value)+" = "+"XY_vars["+str(index)+", :]"
			var_defs.append(value)
		var_defs = "\n".join(var_defs)
		io_dict = OrderedDict(io_dict)
		output_var_idx = [list(io_dict.values()).index(output_vars[i]) for i in range(len(output_vars)) if output_vars[i] in io_dict.values()]
		ordered_eqns = ["	"+i for i in ordered_eqns]
		eq = '\n'.join(ordered_eqns)
		func_def = "def F(XY_vars, util):\n"
		pyfilecontent = func_def+var_defs+"\n"+eq+"\n"+"	return "+var_out.split(" = ")[0]

		return orderedVerilog, pyfilecontent, num_out_vars, num_of_vars, output_var_idx, io_dict, len(ordered_eqns)

	def visitModule_identifier(self, ctx: Verilog2001Parser.Module_identifierContext):
		return str(ctx.getText())

	def visitList_of_ports(self, ctx: Verilog2001Parser.List_of_portsContext):
		return str(ctx.getText())

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
		self.source = lv
		rv = self.visit(ctx.expression())
		if rv[0] == "(":
			rv = rv[1:-1]
		self.eqn_dict[lv] = lv + " = " + rv
		return lv + " = " + rv + " "

	def visitNet_lvalue(self, ctx: Verilog2001Parser.Net_lvalueContext):
		return str(ctx.getText())
	
	def visitExpression(self, ctx: Verilog2001Parser.ExpressionContext):
		exp = ""
		if ctx.binary_operator():
			left_par = ""
			for i in range(len(ctx.term())):
				left_par += "("
			exp += left_par
			for i in range(len(ctx.term())):
				exp1 = self.visit(ctx.term()[i])
				exp += exp1
				if i >= 1:
					exp += ")"
				if i < len(ctx.term()) - 1:
					exp += " "+self.visit(ctx.binary_operator()[0])+" "
			exp += ")"
		if len(ctx.term()) == 1:
			exp = self.visit(ctx.term()[0])
		
		return exp
	
	def visitTerm(self, ctx: Verilog2001Parser.TermContext):
		term = ""
		if ctx.unary_operator():
			term = self.visit(ctx.unary_operator())
		term1 = self.visit(ctx.primary())
		term += term1
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
			return "("+ctx.getText()+")"
		else:
			self.G.add_edge(str(ctx.getText()), self.source)
			return str(ctx.getText())
	
	def visitUnary_operator(self, ctx: Verilog2001Parser.Unary_operatorContext):
		return str(ctx.getText())
	
	def visitBinary_operator(self, ctx: Verilog2001Parser.Binary_operatorContext):
		return str(ctx.getText())
	
	def visitModule_or_generate_item_declaration(self, ctx: Verilog2001Parser.Module_or_generate_item_declarationContext):
		return self.visit(ctx.net_declaration())

	def visitNet_declaration(self, ctx: Verilog2001Parser.Net_declarationContext):
		return self.visit(ctx.list_of_net_identifiers())
	
	def visitList_of_net_identifiers(self, ctx: Verilog2001Parser.List_of_net_identifiersContext):
		wires = str(ctx.getText()).split(",")
		wires = '\n'.join(wires)
		# self.f.write(wires+"\n")
		return str(ctx.getText())

	def visitInput_declaration(self, ctx: Verilog2001Parser.Input_declarationContext):
		inps = self.visit(ctx.list_of_port_identifiers())
		self.input_vars.append(inps.split(" "))
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
		return lv[:-2]

	def visitList_of_port_identifiers(self, ctx: Verilog2001Parser.List_of_port_identifiersContext):
		ids = ""
		for i in range(len(ctx.port_identifier())):
			ids += self.visit(ctx.port_identifier()[i]) + " "
		return ids
		
	def visitPort_identifier(self, ctx: Verilog2001Parser.Port_identifierContext):
		self.G.add_node(str(ctx.getText()))
		return str(ctx.getText())