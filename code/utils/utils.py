import argparse
import copy
import os
import tempfile
from math import floor
from typing import OrderedDict

import numpy as np
import pandas as pd
import torch
from numpy.core.numeric import indices


class cexmodels:

    def __init__(self, modelx, modely, modelyp):
        self.modelx = modelx
        self.modely = modely
        self.modelyp = modelyp

# Utilities class


class utils():
    def __init__(self):
        self.name = "product"
        self.res = []
        super(utils, self).__init__()

    def negation(self, x):
        return 1 - x

    # Continuous AND
    def tnorm_vectorized(self, t, u):
        if self.name == "luka":
            return max(0, t + u - 1)
        elif self.name == "godel":
            return torch.minimum(t, u)
        elif self.name == "product":
            return torch.multiply(t, u)
        else:
            print("Wrong Name!")

    def tconorm_vectorized(self, t, u):
        return 1 - self.tnorm_vectorized(1-t, 1-u)

    # Continuous AND with n inputs
    def tnorm_n_inputs(self, inp):
        if self.name == "luka":
            return torch.max(torch.tensor(0), torch.sum(inp) - 1)
        elif self.name == "godel":
            out, _ = torch.min(inp, dim=-2)
            return out
        elif self.name == "product":
            return torch.prod(inp, -2)
        else:
            print("Wrong Name!")

    # Continuous xor definition
    def continuous_xor(self, x, y):
        t = self.tnorm_vectorized(1-x, y)
        u = self.tnorm_vectorized(x, 1-y)
        return self.tconorm_vectorized(t, u)

    def continuous_xor_vectorized(self, XY_vars):
        op1 = XY_vars[0, :]
        for i in range(XY_vars.shape[0]-1):
            op2 = XY_vars[i+1, :]
            t = self.tnorm_vectorized(1-op1, op2)
            u = self.tnorm_vectorized(op1, 1-op2)
            res = self.tconorm_vectorized(t, u)
            op1 = res
        return res

    def generateAllBinaryStrings(self, n, arr, i):
        if i == n:
            a = copy.deepcopy(arr)
            self.res.append(a)
            return
        arr[i] = 0.0
        self.generateAllBinaryStrings(n, arr, i + 1)
        arr[i] = 1.0
        self.generateAllBinaryStrings(n, arr, i + 1)

    def proc(self, dat, range):
        return torch.where(
            dat > 0.5,
            torch.rand(dat.shape)*range + (1 - range),
            torch.rand(dat.shape)*range
        )

    def add_noise(self, samples, range=0.1):
        return self.proc(samples, range)

    # Seed based sampling from truth table
    def seed_sampling(
        self, 
        no_of_samples, 
        util, 
        py_spec, 
        threshold, 
        num_of_vars
        ):

        arr = [0 for i in range(num_of_vars)]
        self.generateAllBinaryStrings(num_of_vars, arr, 0)
        XY_vars = torch.from_numpy(np.array(self.res).T)
        res = py_spec.F(XY_vars, util)
        samples = XY_vars[:, res >= threshold].T
        df = pd.DataFrame(samples.numpy())
        samples = torch.tensor(df.values)
        samples.type(torch.DoubleTensor)
        gen_new_data = torch.cat(
            [(self.add_noise(samples.T)).T for _ in range(100)])
        # gen_new_data = (gen_new_data * 10**2).round() / (10**2)
        samples = gen_new_data.type(torch.DoubleTensor)
        print("Train Data Generated: ", samples.shape, samples.dtype)
        return samples

    # Fractional Sampling
    def fractional_sampling(
        self, 
        no_of_samples, 
        util, 
        py_spec, 
        threshold, 
        num_of_vars
        ):

        first_interval = np.array([0, 0.01])
        second_interval = np.array([0.99, 1])
        total_length = np.ptp(first_interval)+np.ptp(second_interval)
        n = (num_of_vars, no_of_samples)
        np.random.seed(0)
        numbers = np.random.random(n)*total_length
        numbers += first_interval.min()
        numbers[numbers > first_interval.max()] += second_interval.min() - \
            first_interval.max()
        XY_vars = torch.from_numpy(numbers)
        res = py_spec.F(XY_vars, util)
        samples = XY_vars[:, res >= threshold].T
        print("Train Data Generated: ", samples.shape)
        return samples

    # Fractional Sampling
    def fractional_sampling_pos_and_neg(
        self, 
        no_of_samples, 
        util, 
        threshold, 
        num_of_vars,
        py_spec
        ):

        first_interval = np.array([0, 0.3])
        second_interval = np.array([0.7, 1])

        total_length = np.ptp(first_interval)+np.ptp(second_interval)
        n = (num_of_vars, no_of_samples)
        numbers = np.random.random(n)*total_length
        numbers += first_interval.min()
        numbers[numbers > first_interval.max()] += second_interval.min() - \
            first_interval.max()

        XY_vars = torch.from_numpy(numbers)
        res = py_spec.F(XY_vars, util)
        samples = XY_vars[:num_of_vars, :]
        outs = (res > threshold).double()
        train_samples = torch.cat((samples.T, outs.reshape(-1, 1)), dim=1)
        sorted_data = torch.stack(
            sorted(train_samples, key=lambda train_samples: train_samples[-1], reverse=True))
        train_samples = sorted_data[:2*(outs == 1).sum(), :]
        print("Train Data Generated: ", train_samples.shape)

        return train_samples

        # Fractional Sampling
    # def correlated_fractional_sampling(
    #     self, 
    #     no_of_samples, 
    #     util, 
    #     threshold, 
    #     num_of_vars
    #     ):
        
    #     first_interval = np.array([0, 0.3])
    #     second_interval = np.array([0.7, 1])
    #     total_length = np.ptp(first_interval)+np.ptp(second_interval)
    #     n = (num_of_vars, no_of_samples)
    #     numbers = np.random.random(n)*total_length
    #     numbers += first_interval.min()
    #     numbers[numbers > first_interval.max()] += second_interval.min() - \
    #         first_interval.max()
    #     XY_vars = torch.from_numpy(numbers)
    #     if num_of_vars == 1:
    #         data = []
    #         for i in range(XY_vars.shape[1]):
    #             if XY_vars[0, i] > threshold:
    #                 t1 = torch.cat([XY_vars[0, i].unsqueeze(-1),
    #                                1-XY_vars[0, i].unsqueeze(-1)], dim=0)
    #                 data.append(t1)
    #                 t2 = torch.cat([XY_vars[0, i].unsqueeze(-1),
    #                                XY_vars[0, i].unsqueeze(-1)], dim=0)
    #                 data.append(t2)
    #         train_samples = torch.stack(data)
    #         res = func_spec.F(XY_vars, util)
    #         outs = (res > threshold).double()
    #         train_samples = torch.cat(
    #             (train_samples[:, :num_of_vars], outs.reshape(-1, 1)), dim=1)
    #         print("Train Data Generated: ", train_samples.shape)
    #         return train_samples
    #     res = self.continuous_xor_vectorized(XY_vars)
    #     data = []
    #     for i in range(res.shape[0]):
    #         if res[i] > threshold:
    #             t1 = torch.cat([XY_vars[:, i], 1-res[i].unsqueeze(-1)], dim=0)
    #             data.append(t1)
    #             t2 = torch.cat([XY_vars[:, i], res[i].unsqueeze(-1)], dim=0)
    #             data.append(t2)
    #     train_samples = torch.stack(data)
    #     res = self.continuous_xor_vectorized(train_samples.T)
    #     outs = (res > threshold).double()
    #     train_samples = torch.cat(
    #         (train_samples[:, :num_of_vars], outs.reshape(-1, 1)), dim=1)
    #     print("Train Data Generated: ", train_samples.shape)

    #     return train_samples


    def make_arg_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--threshold", metavar="--th", type=float,
                            default=0.8, help="Enter value between 0.5 <= th <= 1")
        parser.add_argument("--training_size", metavar="--n",
                            type=int, default=50000, help="Enter n >= 50000")
        parser.add_argument("--no_of_input_var", metavar="--noiv",
                            type=int, default=1, help="Enter value >= 1")
        parser.add_argument("--run_for_all_outputs", type=int, default=1,
                            help="0: Runs for only one output var, 1: Runs for all ouputs")
        parser.add_argument("--K", type=int, default=10,
                            help="No. of Clauses >= 1")
        parser.add_argument("--epochs", type=int, default=50,
                            help="No. of epochs to train")
        parser.add_argument("--learning_rate", metavar="--lr",
                            type=float, default=0.01, help="Default 0.01")
        parser.add_argument("--batch_size", type=int,
                            default=32, help="Enter batch size")
        parser.add_argument("--tnorm_name", type=str,
                            default="product", help="godel/product")
        parser.add_argument("--P", type=int, default=0,
                            help="0: Regression, 1: Classification with y as labels, 2: Classification with F out as labels")
        parser.add_argument("--train", type=int, default=0,
                            help="1/0; 0 loads the saved model")
        parser.add_argument("--correlated_sampling",
                            type=int, default=0, help="1/0")
        parser.add_argument("--verilog_spec", type=str,
                            default="sample1", help="Enter file name")
        parser.add_argument("--verilog_spec_location", type=str,
                            default="verilog", help="Enter file location")
        return parser

    def preprocess_wrapper(self, verilog_spec, verilog_spec_location):
        verilog, varlistfile = self.prepare_file_names(verilog_spec, verilog_spec_location)
        output_varlist = self.get_output_varlist(varlistfile)  # Y variable list
        output_varlist = ["i"+e.split("_")[1] for e in output_varlist]
        Xvar_tmp, Yvar_tmp, total_vars = self.get_temporary_variables(verilog, output_varlist)
        total_varsz3 = total_vars
        total_vars = ["i"+e.split("_")[1] for e in total_vars]
        verilog_formula = self.change_modulename(verilog)
        pos_unate, neg_unate, Xvar, Yvar, Xvar_map, Yvar_map = self.preprocess_manthan(
            varlistfile, verilog, Xvar_tmp, Yvar_tmp
            )
        return verilog, output_varlist, total_vars, total_varsz3, verilog_formula, pos_unate, neg_unate, Xvar, Yvar, Xvar_map, Yvar_map

    def make_dataset_larger(self, samples):
        
        # tensor dataset
        training_samples = torch.from_numpy(samples)
        # repeat samples n=2 times
        # training_samples = training_samples.repeat(2, 1)
        # add random noise to get fractional samples
        training_samples = torch.cat([
            self.add_noise((training_samples)) for _ in range(10)
            ])
        training_samples = training_samples.to(torch.double)

        return training_samples

    def prepare_io_dicts(self, total_vars, total_varsz3):

        io_dict = {}
        for index, value in enumerate(total_vars):
            io_dict[index] = value
        io_dict = OrderedDict(io_dict)

        io_dictz3 = {}
        for index, value in enumerate(total_varsz3):
            io_dictz3[index] = value
        io_dictz3 = OrderedDict(io_dictz3)
        
        return io_dict, io_dictz3

    def prepare_ce(self, io_dict, counter_examples, num_of_vars, num_of_outputs):
        '''
        Takes the counter examples generated by z3 solver
        and adds column for the bounded variable (output variable)
        and formats it required shapes

        Returns formatted counter examples
        '''

        ce1 = []
        noce = len(counter_examples[io_dict[0]])  # number of counter examples

        num_of_inputs = num_of_vars-num_of_outputs
        for i in range(len(io_dict)):
            if io_dict[i] in counter_examples:
                ce1.append(np.array(counter_examples[io_dict[i]]).T)

        arr = [0 for i in range(num_of_outputs)]
        self.res = []
        self.generateAllBinaryStrings(num_of_outputs, arr, 0)
        out_vars = np.array(self.res)
        out_vars = np.repeat(
            out_vars, noce, axis=0
        )
        ce = np.concatenate(ce1[:]).reshape((noce, num_of_inputs))
        ce = torch.from_numpy(np.concatenate(
            (
                np.repeat(
                    ce, 2**num_of_outputs, axis=0
                ),
                out_vars
            ),
            axis=1
        )
        )
        return ce.T
    
    def generate_counter_examples(self,
    n, io_dict, counter_examples, py_spec, util, num_of_vars, num_of_outputs
	):
        '''
        Prepares counter examples and filters it based on 
        specification encoded in py_spec
        ce: counter examples
        '''

        # n = 5000
        ce = self.prepare_ce(io_dict, counter_examples, num_of_vars, num_of_outputs)
        # print("ce shape: ", ce.shape)
        res = py_spec.F(ce, util)
        # print("ce:: ", ce[:, res >= 0.5].T)
        ce = torch.cat(
            [
                self.add_noise((ce[:, res >= 0.5].T)) for _ in range(n)
            ]
        ).double()
        return ce
    
    def store_nn_output(self, num_of_outputs, skfunc):
        '''
        Stores the skolem function into file
        skfunc: skolem function
        '''

        open('nn_output', 'w').close()
        f = open("nn_output", "a")
        for i in range(num_of_outputs):
            if i < num_of_outputs-1:
                f.write(skfunc[i][:-1]+"\n")
            else:
                f.write(skfunc[i][:-1])
        f.close()


    def store_losses(self, train_loss, valid_loss):
        '''
        stores losses into file
        '''

        f = open("train_loss", "w")
        train_loss = np.array(train_loss)
        train_loss.tofile(f, sep=",", format="%s")
        f.close()
        f = open("valid_loss", "w")
        valid_loss = np.array(valid_loss)
        valid_loss.tofile(f, sep=",", format="%s")
        f.close()


    def store_preprocess_time(self,
        verilog_spec, num_of_vars, num_out_vars, num_of_eqns, epochs, no_of_samples, preprocess_time
        ):
        line = verilog_spec+","+str(num_of_vars)+","+str(num_out_vars)+","+str(
            num_of_eqns)+","+str(epochs)+","+str(no_of_samples)+","+str(preprocess_time)+"\n"
        f = open("preprocess_data.csv", "a")
        f.write(line)
        f.close()


    def load_python_spec(self, filename):
        mod = __import__('python_specs', fromlist=[filename])
        py_spec = getattr(mod, filename)

        return py_spec


    def get_var_indices(self, num_of_vars, output_varlist, io_dict):
        output_var_idx = [list(io_dict.values()).index(output_varlist[i]) for i in range(len(output_varlist)) if output_varlist[i] in io_dict.values()]
        var_indices = [i for i in range(num_of_vars)]
        input_var_idx = torch.tensor(
            [x for x in var_indices if x not in output_var_idx])

        return var_indices, input_var_idx, output_var_idx


    def get_train_test_split(self, training_samples):
        data_size = training_samples.shape[0]
        val_size = floor(data_size*0.5)
        train_size = data_size - val_size
        validation_set = training_samples[train_size:, :]
        training_set = training_samples[:, :]

        return training_set, validation_set

    # MANTHAN MODULES:
    def prepare_file_names(self, verilog_spec, verilog_spec_location):

        filename = verilog_spec.split(".v")[0]+"_varstoelim.txt"
        varlistfile = "benchmarks/"+\
            verilog_spec_location+"/Yvarlist/"+filename
        verilog = "benchmarks/"+\
            verilog_spec_location+"/"+verilog_spec
        
        return verilog, varlistfile
    
    def get_output_varlist(self, varlistfile):

        return [line.rstrip('\n')
               for line in open(varlistfile)]
    
    def get_temporary_variables(self, verilog, output_varlist):

        Xvar_tmp = []
        Yvar_tmp = []
        with open(verilog, 'r') as f:
            for x, line in enumerate(f):
                if line.startswith("module"):
                    line_split = line.split("(")
                    total_var = line_split[1].split(",")
                    for var in range(len(total_var) - 1):
                        variable_check = total_var[var]
                        variable_check = variable_check.strip(" ").strip("\n")
                        if str(variable_check) in output_varlist:
                            # dg.add_node(var)
                            Yvar_tmp.append(var)
                        else:
                            Xvar_tmp.append(var)
        return Xvar_tmp, Yvar_tmp, total_var[:-1]

    def change_modulename(self, verilog):

        verilog_formula = ''
        with open(verilog, 'r') as f:
            for x, line in enumerate(f):
                if line.startswith("module"):
                    modulename = line.split("(")[0].split(" ")[1]
                    line = line.replace(modulename, "FORMULA")
                verilog_formula += line

        return verilog_formula

    def preprocess_manthan(self, varlistfile,verilog,Xvar_tmp,Yvar_tmp):
        inputfile_name = verilog.split(".v")[0]
        cmd = "./dependencies/preprocess2 -b %s -v %s > /dev/null 2>&1 " % (
            verilog, varlistfile)
        os.system(cmd)
        pos_unate = []
        neg_unate = []
        Xvar = []
        Yvar = []
        Xvar_map = []
        Yvar_map = []
        found_neg = 0
        exists = os.path.isfile(inputfile_name + "_vardetails")
        if exists:
            with open(inputfile_name + "_vardetails", 'r') as f:
                lines = f.readlines()
            f.close()
            for line in lines:
                if "Xvar " in line:
                    Xvar = line.split(":")[1].strip(" \n").split(" ")
                    Xvar = np.array(Xvar)
                    Xvar = Xvar.astype(int)
                    # first variable is 0 now, not 1
                    Xvar = np.subtract(Xvar, 1)
                    continue
                if "Yvar " in line:
                    Yvar = line.split(":")[1].strip(" \n").split(" ")
                    Yvar = np.array(Yvar)
                    Yvar = Yvar.astype(int)
                    # first variable is 0 now, not 1
                    Yvar = np.subtract(Yvar, 1)
                    continue
                if "Yvar_map " in line:
                    Yvar_map = line.split(":")[1].strip(" \n").split(" ")
                    Yvar_map = np.array(Yvar_map)
                    Yvar_map = Yvar_map.astype(int)
                    continue
                if "Xvar_map " in line:
                    Xvar_map = line.split(":")[1].strip(" \n").split(" ")
                    Xvar_map = np.array(Xvar_map)
                    Xvar_map = Xvar_map.astype(int)
                    continue
                if "Posunate" in line:
                    pos = line.split(":")[1].strip(" \n")
                    if pos != "":
                        pos_unate = pos.split(" ")
                        pos_unate = np.array(pos_unate)
                        pos_unate = pos_unate.astype(int)
                    continue
                if "Negunate" in line:
                    neg = line.split(":")[1].strip(" \n")
                    if neg != "":
                        neg_unate = neg.split(" ")
                        neg_unate = np.array(neg_unate)
                        neg_unate = neg_unate.astype(int)
                    continue
            print("preprocess done")
            print("creating cnf file..")
            os.unlink(inputfile_name + "_vardetails")
            Xvar_map = dict(zip(Xvar, Xvar_map))
            Yvar_map = dict(zip(Yvar, Yvar_map))
            Xvar = sorted(Xvar)
            Yvar = sorted(Yvar)
        else:
            print("preprocessing error .. continuing ")
            cmd = "./dependencies/file_generation_cnf %s %s.cnf %s_mapping.txt  > /dev/null 2>&1" % (
                verilog, inputfile_name, inputfile_name)
            os.system(cmd)
            print("cmd: ", cmd)
            with open(inputfile_name + "_mapping.txt", 'r') as f:
                lines = f.readlines()
            f.close()
            for line in lines:
                allvar_map = line.strip(" \n").split(" ")
            print("allvars: ", allvar_map)
            # os.unlink(inputfile_name + "_mapping.txt")
            allvar_map = np.array(allvar_map).astype(int)
            Xvar_map = dict(zip(Xvar_tmp, allvar_map[Xvar]))
            Yvar_map = dict(zip(Yvar_tmp, allvar_map[Yvar]))
            Xvar = np.sort(np.array(Xvar_tmp))
            Yvar = np.sort(np.array(Yvar_tmp))

        pos_unate_list = []
        neg_unate_list = []
        for unate in pos_unate:
            pos_unate_list.append(list(Yvar_map.keys())[list(Yvar_map.values()).index(unate)])
        for unate in neg_unate:
            neg_unate_list.append(list(Yvar_map.keys())[list(Yvar_map.values()).index(unate)])
        
        return pos_unate_list, neg_unate_list, Xvar, Yvar, Xvar_map, Yvar_map

    def unate_skolemfunction(self, Xvar, Yvar, pos_unate, neg_unate, inputfile_name):

        skolemformula = tempfile.gettempdir() + \
            '/' + inputfile_name + \
            "_skolem.v"  # F(X,Y')
        inputstr = 'module SKOLEMFORMULA ('
        declarestr = ''
        assignstr = ''
        itr = 1
        for var in range(len(Xvar) + len(Yvar)):
            inputstr += "i%s, " % (var)
            if var in Xvar:
                declarestr += "input i%s;\n" % (var)
            if var in Yvar:
                declarestr += "output i%s;\n" % (var)
                if var in neg_unate:
                    assignstr += "assign i%s = 0;\n" % (var)
                if var in pos_unate:
                    assignstr += "assign i%s = 1;\n" % (var)
        inputstr += ");\n"
        f = open(skolemformula, "w")
        f.write(inputstr)
        f.write(declarestr)
        f.write(assignstr)
        f.write("endmodule")
        f.close()
        cmd = "./dependencies/file_write_verilog %s %s > /dev/null 2>&1  " % (
            skolemformula, skolemformula)
        os.system(cmd)

    def check_unates(self, pos_unate, neg_unate, Xvar, Yvar, inputfile_name):
        # if all Y variables are unate
        if len(pos_unate) + len(neg_unate) == len(Yvar):
            print(len(pos_unate) + len(neg_unate))
            print("positive unate", len(pos_unate))
            print("all Y variables are unates")
            print("Solved !! done !")
            self.unate_skolemfunction(Xvar, Yvar, pos_unate, neg_unate, inputfile_name)
            skolemformula = tempfile.gettempdir() + \
                '/' + inputfile_name + "_skolem.v"
            exists = os.path.isfile(skolemformula)
            if exists:
                os.system("cp " + skolemformula +
                        " ./skfs/" + inputfile_name + "_skolem.v")
            exists = os.path.isfile("strash.txt")
            if exists:
                os.unlink("strash.txt")
            exists = os.path.isfile("variable_mapping.txt")
            if exists:
                os.unlink("variable_mapping.txt")
            return True

    def prepare_cnf_content(self, verilog, Xvar, Yvar, Xvar_map, Yvar_map, pos_unate, neg_unate):

        cnffile = verilog.split(".v")[0] + ".cnf"
        # print("prep cnf: ", Yvar, Yvar_map)

        # to add c ind and positive and negative unate in cnf
        unates = []
        indIter = 1
        indStr = 'c ind '
        allvar_map = []
        for i in range(len(Xvar) + len(Yvar)):
            if i in Xvar:
                i_map = Xvar_map[i]
            if i in Yvar:
                i_map = Yvar_map[i]
            allvar_map.append(i_map)
            if indIter % 10 == 0:
                indStr += ' 0\nc ind '
            indStr += "%d " % i_map
            indIter += 1
        indStr += " 0\n"
        allvar_map = np.array(allvar_map)
        fixedvar = ''
        for i in pos_unate:
            fixedvar += "%s 0\n" % (Yvar_map[i])
            unates.append(i)
        for i in neg_unate:
            fixedvar += "-%s 0\n" % (Yvar_map[i])
            unates.append(i)
        with open(cnffile, 'r') as f:
            lines = f.readlines()
        f.close()
        fixedvar += "%s 0\n" % (1)  # output variable always true.

        unates = np.sort(unates)
        cnf_content = ''
        for line in lines:
            line = line.strip(" \n")
            if line.startswith("c"):
                continue
            if line.startswith('p cnf'):
                numVar = int(line.split()[2])
                numCls = int(line.split()[3])
                line = line.replace("p cnf " + str(numVar) + " " + str(
                    numCls), "p cnf " + str(numVar) + " " + str(numCls + len(unates) + 1))
            cnf_content += line + "\n"
        cnf_content = cnf_content.strip("\n")
        cnf_content = indStr + cnf_content + "\n" + fixedvar.rstrip(' \n')
        # os.unlink(cnffile)

        return cnf_content, allvar_map
    
    def get_sample_cms(self, allvar_map, cnf_content, no_samples, verilog):
        weighted=1
        seed = 10
        verbose=0
        inputfile_name = verilog.split("/")[-1][:-2]

        tempcnffile = tempfile.gettempdir() + '/' + inputfile_name + ".cnf"
        f = open(tempcnffile, "w")
        f.write(cnf_content)
        f.close()
        tempoutputfile = tempfile.gettempdir() + '/' + inputfile_name + "_.txt"
        if weighted:
            print("weighted samples....")
            cmd = "./dependencies/cryptominisat5 -n1 --sls 0 --comps 0"
            cmd += " --restart luby  --nobansol --maple 0 --presimp 0"
            cmd += " --polar weight --freq 0.9999 --verb 0 --scc 0"
            cmd += " --random %s --maxsol %s" % (seed, no_samples)
            cmd += " %s" % (tempcnffile)
            cmd += " --dumpresult %s > /dev/null 2>&1" % (tempoutputfile)
        else:
            print("uniform samples....")
            cmd = "./dependencies/cryptominisat5 --restart luby"
            cmd += " --maple 0 --verb 0 --nobansol"
            cmd += " --scc 1 -n1 --presimp 0 --polar rnd --freq 0.9999"
            cmd += " --random %s --maxsol %s" % (seed, no_samples)
            cmd += " %s" % (tempcnffile)
            cmd += " --dumpresult %s > /dev/null 2>&1" % (tempoutputfile)
        if verbose:
            print("cmd: ", cmd)
        os.system(cmd)
        with open(tempoutputfile, 'r') as f:
            content = f.read()
        f.close()
        os.unlink(tempoutputfile)
        os.unlink(tempcnffile)
        content = content.replace("SAT\n", "").replace(
            "\n", " ").strip(" \n").strip(" ")
        models = content.split(" ")
        models = np.array(models)
        if(models[len(models) - 1] != '0'):
            models = np.delete(models, len(models) - 1, axis=0)
        index = np.where(models == "0")[0][0]
        var_model = np.reshape(models, (-1, index + 1)).astype(int)
        one = np.ones(len(allvar_map), dtype=int)
        allvar_map = np.subtract(allvar_map, one).astype(int)
        var_model = var_model[:, allvar_map]
        var_model = var_model > 1
        var_model = var_model.astype(int)
        return var_model


    def adaptive_samples(self, sample_cnf_content, Yvar_map, allvar_map):
        sample_cnf_content_one = ''
        sample_cnf_content_zero = ''
        bias = {}
        for var in Yvar_map.keys():
            sample_cnf_content_one += "w %d 0.9\n" % (Yvar_map[var])
            sample_cnf_content_zero += "w %d 0.1\n" % (Yvar_map[var])
        samples_one = self.get_sample_cms(
            allvar_map, sample_cnf_content + sample_cnf_content_one, 500)
        samples_zero = self.get_sample_cms(
            allvar_map, sample_cnf_content + sample_cnf_content_zero, 500)
        for var in Yvar_map.keys():
            len_one = np.count_nonzero(samples_one[:, var])
            p = round(float(len_one) / 500, 2)
            len_zero = np.count_nonzero(samples_zero[:, var])
            q = round(float(len_zero) / 500, 3)
            if 0.35 < p < 0.65 and 0.35 < q < 0.65:
                bias[var] = p
            else:
                bias[var] = 0.9
        return bias

    def gen_weighted_cnf(self, cnf_content, Xvar_map, Yvar_map, allvar_map):
        adaptivesample=0
        lines = cnf_content.split("\n")
        sample_cnf_content = ''
        for line in lines:
            line = line.strip("\n")
            if line == '':
                continue
            if line.startswith("c"):
                sample_cnf_content += line + "\n"
                continue
            if line.startswith('p cnf'):
                numVar = int(line.split()[2])
                numCls = int(line.split()[3])
                line = line.replace("p cnf " + str(numVar) + " " + str(numCls), "p cnf " + str(
                    numVar) + " " + str(numCls + len(Xvar_map.keys()) + len(Yvar_map.keys())))
                sample_cnf_content += line + "\n"
                continue
            sample_cnf_content += line + "\n"
        for var in Xvar_map.keys():
            sample_cnf_content += "w %d 0.5\n" % (Xvar_map[var])
        if adaptivesample:
            bias_y = self.adaptive_samples(sample_cnf_content, Yvar_map, allvar_map)
            for var in Yvar_map.keys():
                sample_cnf_content += "w %d %f\n" % (Yvar_map[var], bias_y[var])
        else:
            for var in Yvar_map.keys():
                sample_cnf_content += "w %d 0.9\n" % (Yvar_map[var])
        return sample_cnf_content

    def generate_samples(self, cnf_content, Xvar, Yvar, Xvar_map, Yvar_map, allvar_map, verilog, max_samples=1000):

        SAMPLER_CMS = 1
        samples = 1
        weighted = 1
        if SAMPLER_CMS:
            sample_cnf_content = cnf_content
            if samples:
                no_samples = max_samples
            else:
                if(len(Yvar) + len(Xvar) < 1200):
                    no_samples = 10000
                if ((len(Yvar) + len(Xvar) > 1200) and (len(Yvar) + len(Xvar) < 4000)):
                    no_samples = 5000
                if(len(Yvar) + len(Xvar) > 4000):
                    no_samples = 1000

            print("generating samples ", no_samples)

            if weighted:
                sample_cnf_content = self.gen_weighted_cnf(
                    cnf_content, Xvar_map, Yvar_map, allvar_map)
            samples = self.get_sample_cms(allvar_map, sample_cnf_content, no_samples, verilog)
        x_data, indices = np.unique(samples[:, Xvar], axis=0, return_index=True)
        samples = samples[indices, :]

        x_data, indices = np.unique(samples[:, Xvar], axis=0, return_index=True)
        samples = samples[indices, :]

        return samples
    
    def get_var_counts(self, Xvar, Yvar, verilog):
        num_of_vars = len(Xvar) + len(Yvar)
        num_out_vars = len(Yvar)
        f = open(verilog, 'r')
        data = f.read()
        num_of_eqns = data.count('assign')
        f.close()

        return num_of_vars, num_out_vars, num_of_eqns


    def prepare_candidateskf(self, skfunc, Yvar, pos_unate, neg_unate):
        candidateskf = {}
        j = 0
        for i in Yvar:
            # if i in neg_unate:
            #     candidateskf[i] = ' 0 '
            #     continue
            # if i in pos_unate:
            #     candidateskf[i] = ' 1 '
            #     continue
            if j < len(skfunc):
                candidateskf[i] = skfunc[j][:-1]
            j += 1
        
        return candidateskf


    def create_skolem_function(self, inputfile_name, candidateskf, Xvar, Yvar):
        # we have candidate skolem functions for every y in Y
        # Now, lets generate Skolem formula F(X,Y') : input X and output Y'
        tempOutputFile = tempfile.gettempdir() + \
            '/' + inputfile_name + \
            "_skolem.v"  # F(X,Y')

        inputstr = 'module SKOLEMFORMULA ('
        declarestr = ''
        assignstr = ''
        wirestr = 'wire zero;\nwire one;\n'
        wirestr += "assign zero = 0;\nassign one = 1;\n"
        outstr = ''
        itr = 1
        wtlist = []
        for var in range(len(Xvar) + len(Yvar)):
            inputstr += "i%s, " % (var)
            if var in Xvar:
                declarestr += "input i%s;\n" % (var)
            if var in Yvar:
                flag = 0
                declarestr += "input i%s;\n" % (var)
                wirestr += "wire wi%s;\n" % (var)
                assignstr += 'assign wi%s = (' % (var)
                if var in candidateskf:
                    temp = candidateskf[var].replace(
                        " 1 ", " one ").replace(" 0 ", " zero ")
                assignstr += temp + ");\n"
                outstr += "(~(wi%s ^ i%s)) & " % (var, var)
                if itr % 10 == 0:
                    flag = 1
                    outstr = outstr.strip("& ")
                    wirestr += "wire wt%s;\n" % (itr)
                    assignstr += "assign wt%s = %s;\n" % (itr, outstr)
                    wtlist.append(itr)
                    outstr = ''

                itr += 1
        if(flag == 0):
            outstr = outstr.strip("& ")
            wirestr += "wire wt%s;\n" % (itr)
            assignstr += "assign wt%s = %s;\n" % (itr, outstr)
            wtlist.append(itr)
        assignstr += "assign out = "
        for i in wtlist:
            assignstr += "wt%s & " % (i)
        assignstr = assignstr.strip("& ") + ";\n"
        inputstr += " out );\n"
        declarestr += "output out ;\n"
        f = open(tempOutputFile, "w")
        f.write(inputstr)
        f.write(declarestr)
        f.write(wirestr)
        f.write(assignstr)
        f.write("endmodule")
        f.close()


    def create_error_formula(self, Xvar, Yvar, verilog_formula):
        refine_var_log = {}
        
        inputformula = '('
        inputskolem = '('
        inputerrorx = 'module MAIN ('
        inputerrory = ''
        inputerroryp = ''
        declarex = ''
        declarey = ''
        declareyp = ''
        for var in range(len(Xvar) + len(Yvar)):
            if var in Xvar:
                inputformula += "i%s, " % (var)
                inputskolem += "i%s, " % (var)
                inputerrorx += "i%s, " % (var)
                declarex += "input i%s ;\n" % (var)
            if var in Yvar:
                refine_var_log[var] = 0
                inputformula += "i%s, " % (var)
                inputskolem += "ip%s, " % (var)
                inputerrory += "i%s, " % (var)
                inputerroryp += "ip%s, " % (var)
                declarey += "input i%s ;\n" % (var)
                declareyp += "input ip%s ;\n" % (var)
        inputformula += "out1 );\n"
        inputformula_sk = inputskolem + "out3 );\n"
        inputskolem += "out2 );\n"

        inputerrorx = inputerrorx + inputerrory + inputerroryp + "out );\n"
        declare = declarex + declarey + declareyp + 'output out;\n' + \
            "wire out1;\n" + "wire out2;\n" + "wire out3;\n"
        formula_call = "FORMULA F1 " + inputformula
        skolem_call = "SKOLEMFORMULA F2 " + inputskolem
        formulask_call = "FORMULA F2 " + inputformula_sk
        error_content = inputerrorx + declare + \
            formula_call + skolem_call + formulask_call
        error_content += "assign out = ( out1 & out2 & ~(out3) );\n" + \
            "endmodule\n"
        error_content += verilog_formula
        return error_content, refine_var_log
    
    def add_skolem_to_errorformula(self, error_content, selfsub, verilog):

        inputfile_name = verilog.split("/")[-1][:-2]
        skolemformula = tempfile.gettempdir() + '/' + inputfile_name + "_skolem.v"
        with open(skolemformula, 'r') as f:
            skolemcontent = f.read()
        f.close()
        errorformula = tempfile.gettempdir() + '/' + inputfile_name + "_errorformula.v"
        skolemcontent_write = ''
        if len(selfsub) != 0:
            for all_selfsub_var in selfsub:
                file_open = open(
                    tempfile.gettempdir()+"/selfsub/formula%s_true.v" % (all_selfsub_var), "r")
                content = file_open.read()
                file_open.close()
                skolemcontent_write += "\n" + content
        f = open(errorformula, "w")
        f.write(error_content)
        f.write(skolemcontent)
        f.write(skolemcontent_write)
        f.close()
    
    def createSkolem(self, candidateSkf, Xvar, Yvar, UniqueVars, UniqueDef, inputfile_name):
        tempOutputFile = tempfile.gettempdir() + '/' + inputfile_name + "_skolem.v"  # F(X,Y')
        inputstr = 'module SKOLEMFORMULA ('
        declarestr = ''
        assignstr = ''
        wirestr = 'wire zero;\nwire one;\n'
        wirestr += "assign zero = 0;\nassign one = 1;\n"
        outstr = ''
        itr = 1
        wtlist = []
        
        for var in Xvar:
            declarestr += "input i%s;\n" % (var)
            inputstr += "i%s, " % (var)
        for var in Yvar:
            flag = 0
            declarestr += "input o%s;\n" % (var)
            inputstr += "o%s, " % (var)
            wirestr += "wire w%s;\n" % (var)
            if var not in UniqueVars:
                assignstr += 'assign w%s = (' % (var)
                assignstr += candidateSkf[var].replace(" 1 ", " one ").replace(" 0 ", " zero ") +");\n"
            
            outstr += "(~(w%s ^ o%s)) & " % (var,var)
            if itr % 10 == 0:
                flag = 1
                outstr = outstr.strip("& ")
                wirestr += "wire wt%s;\n" % (itr)
                assignstr += "assign wt%s = %s;\n" % (itr, outstr)
                wtlist.append(itr)
                outstr = ''
            itr += 1
        if(flag == 0):
            outstr = outstr.strip("& ")
            wirestr += "wire wt%s;\n" % (itr)
            assignstr += "assign wt%s = %s;\n" % (itr, outstr)
            wtlist.append(itr)
        assignstr += "assign out = "
        for i in wtlist:
            assignstr += "wt%s & " % (i)
        assignstr = assignstr.strip("& ") + ";\n"
        inputstr += " out );\n"
        declarestr += "output out ;\n"
        f = open(tempOutputFile, "w")
        f.write(inputstr + declarestr + wirestr)
        # f.write(UniqueDef.strip("\n")+"\n")
        f.write(assignstr + "endmodule")
        f.close()
    
    def createErrorFormula(self, Xvar, Yvar, UniqueVars, verilog_formula):
        inputformula = '('
        inputskolem = '('
        inputerrorx = 'module MAIN ('
        inputerrory = ''
        inputerroryp = ''
        declarex = ''
        declarey = ''
        declareyp = ''
        for var in Xvar:
            inputformula += "%s, " % (var)
            inputskolem += "%s, " % (var)
            inputerrorx += "%s, " % (var)
            declarex += "input %s ;\n" % (var)
        for var in Yvar:
            inputformula += "%s, " % (var)
            inputerrory += "%s, " % (var)
            declarey += "input %s ;\n" % (var) 
            inputerroryp += "ip%s, " % (var)
            declareyp += "input ip%s ;\n" % (var)
            if var in UniqueVars:
                inputskolem += "%s, " %(var)
            else:
                inputskolem += "ip%s, " %(var)
        inputformula += "out1 );\n"
        inputformula_sk = inputskolem + "out3 );\n"
        inputskolem += "out2 );\n"
        inputerrorx = inputerrorx + inputerrory + inputerroryp + "out );\n"
        declare = declarex + declarey + declareyp + 'output out;\n' + \
            "wire out1;\n" + "wire out2;\n" + "wire out3;\n"
        formula_call = "FORMULA F1 " + inputformula
        skolem_call = "SKOLEMFORMULA F2 " + inputskolem
        formulask_call = "FORMULA F2 " + inputformula_sk
        error_content = inputerrorx + declare + \
            formula_call + skolem_call + formulask_call
        error_content += "assign out = ( out1 & out2 & ~(out3) );\n" + \
            "endmodule\n"
        error_content += verilog_formula
        return error_content

    def write_error_formula(self, verilog_spec, verilog, verilog_formula, skfunc, Xvar, Yvar, pos_unate, neg_unate):
        candidateskf = self.prepare_candidateskf(skfunc, Yvar, pos_unate, neg_unate)
        # self.create_skolem_function(
        #     verilog_spec.split('.v')[0], candidateskf, Xvar, Yvar)
        self.createSkolem(candidateskf, Xvar, Yvar, [], [], verilog_spec.split('.qdimacs')[0])
        # error_content, refine_var_log = self.create_error_formula(
        #     Xvar, Yvar, verilog_formula)
        error_content = self.createErrorFormula(Xvar, Yvar, [], verilog_formula)
        self.add_skolem_to_errorformula(error_content, [], verilog)

    def verify(self, Xvar, Yvar, verilog):
        print("In Verify !!")
        inputfile_name = verilog.split("/")[-1][:-2]
        errorformula = tempfile.gettempdir() + '/' + inputfile_name + "_errorformula.v"
        cexfile = tempfile.gettempdir() + '/' + inputfile_name + "_cex.txt"
        f = tempfile.gettempdir()+'/'+"hello.txt"

        e = os.path.isfile("strash.txt")
        if e:
            os.system("rm strash.txt")
        cmd = "./dependencies/file_generation_cex %s %s > /dev/null 2>&1 " % (
            errorformula, cexfile)
        os.system(cmd)
        e = os.path.isfile("strash.txt")
        if e:
            os.system("rm strash.txt")
            exists = os.path.isfile(cexfile)
            if exists:
                ret = 1
                with open(cexfile, 'r') as f:
                    lines = f.readlines()
                f.close()
                os.unlink(cexfile)
                for line in lines:
                    model = line.strip(" \n")
                cex = list(map(int, model))
                templist = np.split(cex, [len(Xvar), len(Xvar) + len(Yvar)])
                modelx = templist[0]
                modely = templist[1]
                modelyp = templist[2]
                assert(len(modelx) == len(Xvar))
                assert(len(modelyp) == len(Yvar))
                assert(len(modely) == len(Yvar))
                model_cex = cexmodels(modelx, modely, modelyp)
                return(1, model_cex, ret)
            else:
                return(1, [], 0)
        else:
            return(0, [0], 1)

# Init utilities
util = utils()
