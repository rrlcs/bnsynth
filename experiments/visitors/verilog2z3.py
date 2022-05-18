import sys

import antlr4

from experiments.visitors.Verilog2001Lexer import Verilog2001Lexer
from experiments.visitors.Verilog2001Parser import Verilog2001Parser
from experiments.visitors.verilogToZ3Visitor import verilogVisitor


def preparez3(verilog_spec, verilog_spec_location, num_of_ouputs, manthan=0):
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
    z3filecontent = visitor.visit(tree)
    with open('experiments/visitors/templateZ3Checker.py', 'r') as file:
        filedata = file.read()
        file.close()
    filedata = filedata.replace('#*#', z3filecontent)
    with open('experiments/visitors/z3ClauseCounter.py', 'w') as file:
        file.write(filedata)
        file.close()

    # Parse the NN output and Generate Z3Py Format
    if manthan:
        f = open("experiments/bnsynth_skfs/"+verilog_spec[:-9]+".skf", "r")
    else:
        f = open('experiments/bnsynth_skfs/'+verilog_spec[:-2]+".skf", "r")
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
        with open('experiments/visitors/z3ClauseCounter.py', 'r') as file:
            filedata = file.read()
            file.close()
        text = "($$"+str(i)+")"
        nnOut = str(nnOut).replace('one', 'True')
        nnOut = str(nnOut).replace('zero', 'False')

        if manthan:
            filedata = filedata.replace(text, '()')
        else:
            filedata = filedata.replace(text, nnOut)

        with open('experiments/visitors/z3ClauseCounter.py', 'w') as file:
            file.write(filedata)
            file.close()


if __name__ == "__main__":
    path = 'experiments/manthan_skfs/'
    preparez3('adder_4_9_skolem.v',
              path, 9, 1)
