import sys

import antlr4

from experiments.visitors.Verilog2001Lexer import Verilog2001Lexer
from experiments.visitors.Verilog2001Parser import Verilog2001Parser
from experiments.visitors.verilogToZ3Visitor import verilogVisitor


def preparez3(verilog_spec, verilog_spec_location, num_of_ouputs):
    '''
    Input: verilog file
    Output: z3py equivalent of verilog file
    Functionality: Parses the verilog file and converts it to z3py format.
                                    It does the same for the NN output as well.
    '''

    # verilog_spec = "sampleskf.v"
    # verilog_spec_location = ""
    filename = verilog_spec_location+verilog_spec
    f = open(filename, "r")
    data = f.read()
    inputStream = antlr4.InputStream(data)
    lexer = Verilog2001Lexer(inputStream)
    tokenStream = antlr4.CommonTokenStream(lexer)
    parser = Verilog2001Parser(tokenStream)
    tree = parser.module_declaration()
    visitor = verilogVisitor(
        verilog_spec, verilog_spec_location, num_of_ouputs)
    print("hello1")
    z3filecontent = visitor.visit(tree)
    print("hello")
    with open('experiments/visitors/templateZ3Checker.py', 'r') as file:
        filedata = file.read()
        file.close()
    filedata = filedata.replace('#*#', z3filecontent)
    with open('experiments/visitors/z3ValidityChecker.py', 'w') as file:
        file.write(filedata)
        file.close()

    # Parse the NN output and Generate Z3Py Format
    f = open("gcln_output", "r")
    data = f.read()
    data = data.split("\n")
    sys.setrecursionlimit(2000)
    for i in range(len(data)):
        inputStream = antlr4.InputStream(data[i])
        lexer = Verilog2001Lexer(inputStream)
        tokenStream = antlr4.CommonTokenStream(lexer)
        parser = Verilog2001Parser(tokenStream)
        tree = parser.mintypmax_expression()
        visitor = verilogVisitor(
            verilog_spec, verilog_spec_location, num_of_ouputs)
        nnOut = visitor.visit(tree)
        with open('experiments/visitors/z3ValidityChecker.py', 'r') as file:
            filedata = file.read()
            file.close()
        text = "$$"+str(i)
        # print("text {}, nnout {} ".format(text, nnOut))
        nnOut = str(nnOut).replace('one', 'True')
        nnOut = str(nnOut).replace('zero', 'False')
        filedata = filedata.replace(text, nnOut)
        # filedata = replace_preref(filedata)
        with open('experiments/visitors/z3ValidityChecker.py', 'w') as file:
            file.write(filedata)
            file.close()


if __name__ == "__main__":
    path = '../../data/benchmarks/custom_examples/'
    preparez3('sample1.v',
              path, 2)
