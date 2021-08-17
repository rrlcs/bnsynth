from antlr4 import *
from Verilog2001Parser import Verilog2001Parser
from Verilog2001Visitor import Verilog2001Visitor

class expressionVisitor(Verilog2001Visitor):
	def visitExpression(self, ctx: Verilog2001Parser.ExpressionContext):
		# print("visitExpression: ", ctx.getText())
		terms = []
		for i in range(len(ctx.term())):
			terms.append(self.visit(ctx.term()[i]))
		oper = []
		for i in range(len(ctx.binary_operator())):
			oper.append(self.visit(ctx.binary_operator()[i]))
		# print("terms: ", terms)
		# print("oper: ", oper)
		exp = oper[0]+"("
		for i in range(len(terms)):
			exp += terms[i]
			if i < len(terms)-1:
				exp += ", "
		exp += ")"
		# print("exp: ", exp)

		return exp
		
	
	def visitTerm(self, ctx: Verilog2001Parser.TermContext):
		# print("visitTerm: ", ctx.getText())
		op = ""
		if ctx.unary_operator():
			op = self.visit(ctx.unary_operator())
		return op+"("+str(self.visit(ctx.primary()))+")"
	
	def visitPrimary(self, ctx: Verilog2001Parser.PrimaryContext):
		# print("visitPrimary: ", ctx.getText())
		if ctx.mintypmax_expression():
			return self.visit(ctx.mintypmax_expression())
		if ctx.hierarchical_identifier():
			return self.visit(ctx.hierarchical_identifier())
	
	def visitMintypmax_expression(self, ctx: Verilog2001Parser.Mintypmax_expressionContext):
		# print("visitMintypmax_expression: ", ctx.getText())
		# print(len(ctx.expression()))
		# print("exp mintype: ", self.visit(ctx.expression()[0]))
		return self.visit(ctx.expression()[0])
	
	def visitHierarchical_identifier(self, ctx: Verilog2001Parser.Hierarchical_identifierContext):
		# print("visitHierarchical_identifier: ", ctx.getText())
		return ctx.getText()
	
	def visitUnary_operator(self, ctx: Verilog2001Parser.Unary_operatorContext):
		return "Not"
	
	def visitBinary_operator(self, ctx: Verilog2001Parser.Binary_operatorContext):
		if str(ctx.getText()) == "&":
			op = "And"
		if str(ctx.getText()) == "|":
			op = "Or"
		return op