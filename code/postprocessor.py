import os
import time
from matplotlib.pyplot import cla
import torch
import subprocess
import numpy as np
from code.utils.utils import util
from experiments.visitors.verilog2z3 import preparez3
import sys
from tabulate import tabulate

# sys.setrecursionlimit(200000)


def get_formula_for_uncovered_data(training_samples, disagreed_index, input_var_idx, output_var_idx, current_output=0):
    '''
    Constructs formula for uncovered samples by the Candidate Skolem function
    '''

    uncovered_samples = training_samples[disagreed_index]
    if uncovered_samples.shape[0] > 0:
        final_rem_formula = ""
        final_rem_inp_formula = ""
        for j in range(uncovered_samples.shape[0]):
            rem_formula = ""
            for i in range(len(input_var_idx)):
                if uncovered_samples[j, input_var_idx[i]] == 0:
                    rem_formula += "~i"+str(input_var_idx[i])+" & "
                else:
                    rem_formula += "i"+str(input_var_idx[i])+" & "
            rem_formula = "~("+rem_formula[:-3]+") "
            rem_inp_formula = rem_formula
            # for i in range(len(output_var_idx)):
            if uncovered_samples[j, output_var_idx[current_output]] == 0:
                rem_formula += "| zero"
            else:
                rem_formula += "| one"
            rem_formula = "("+rem_formula+")"
            final_rem_formula += rem_formula + " & "
            final_rem_inp_formula += rem_inp_formula + " & "
    else:
        final_rem_formula = ""
        final_rem_inp_formula = ""

    return final_rem_formula[:-3], final_rem_inp_formula[:-3]


def get_phi_prime_skf_list(skf_list, training_samples, disagreed_indices, input_var_idx, output_var_idx):
    '''
    Appends uncovered samples as formula to the Candidate Skolem function
    '''

    for i in range(len(skf_list)):
        rem_formula, rem_inp_formula = get_formula_for_uncovered_data(
            training_samples, disagreed_indices[i], input_var_idx, output_var_idx, current_output=i)
        phi = skf_list[i]
        if len(rem_formula) > 0 and len(rem_inp_formula) > 0:
            phi_new = rem_formula + " & " + \
                "(~("+rem_inp_formula+") | " + phi+")"
        else:
            phi_new = phi
        skf_list[i] = phi_new

    return skf_list


def print_skolem_function(args):
    path = 'data/benchmarks/'+args.verilog_spec_location+"/"
    f = open('experiments/simplified.skf', 'r')
    simple_skf = f.read()
    f.close()
    filename = args.verilog_spec.split(".v")[0]+"_varstoelim.txt"
    f = open(path+'Yvarlist/'+filename, 'r')
    outputvars = f.read().split('\n')[:-1]
    f.close()
    simplified_skolem_function_list = simple_skf.split("\n")[:-1]
    for i in range(len(simplified_skolem_function_list)):
        print("Skolem Function for Output Var", outputvars[i], ": ",
              simplified_skolem_function_list[i])
    f = open('experiments/' +
             args.verilog_spec[:-2]+'_simplified.skf', 'w')
    f.write(simple_skf)
    f.close()


def get_bnsynth_counts(args, num_of_outputs, total_varsz3, Xvar, ):
    path = 'data/benchmarks/'+args.verilog_spec_location+"/"
    preparez3(args.verilog_spec,
              path, num_of_outputs)

    cmd = 'python experiments/visitors/z3ClauseCounter.py'
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    out, err = p.communicate()
    inp_vars = [total_varsz3[i].replace('_', '') for i in Xvar]
    ftext = out.decode("utf-8").split('\n')[1]
    num_inputs_bnsynth = 0
    for v in inp_vars:
        if v in ftext:
            num_inputs_bnsynth += 1
    bnsynth_clause_counts = out.decode("utf-8").split('\n')[2:-1]
    return bnsynth_clause_counts, num_inputs_bnsynth


def run_manthan(args):
    cmd1 = 'python manthan.py --seed 1 --varlist data/benchmarks/final_custom_benchmarks/verilog/Yvarlist/' + \
        args.verilog_spec[:-2]+'_varstoelim.txt ' + \
        '--verilog data/benchmarks/final_custom_benchmarks/verilog/' + \
        args.verilog_spec+' > /dev/null 2>&1'
    os.system(cmd1)


def get_manthan_counts(args, num_of_outputs, io_dict, Xvar):
    path = 'experiments/manthan_skfs/'
    preparez3(args.verilog_spec[:-2]+'_skolem.v',
              path, num_of_outputs, manthan=1)
    cmd2 = 'python experiments/visitors/z3ClauseCounter.py'

    p = subprocess.Popen(cmd2, stdout=subprocess.PIPE, shell=True)
    out, err = p.communicate()
    ftext = out.decode("utf-8").split('\n')[1]
    total_vars = list(io_dict.values())
    inp_vars = [total_vars[i] for i in Xvar]
    num_inputs_manthan = 0
    for v in inp_vars:
        if v in ftext:
            num_inputs_manthan += 1
    manthan_clause_counts = out.decode("utf-8").split('\n')[2:-1]

    return manthan_clause_counts, num_inputs_manthan


def postprocess(args, model, accuracy, epochs, final_loss, loss_drop, verilogformula, total_varsz3, num_of_inputs, input_var_idx, num_of_outputs,
                output_var_idx, io_dict, io_dictz3, Xvar, Yvar, PosUnate, NegUnate, start_time, training_samples, disagreed_indices, num_of_ce):
    '''
    Extracts Candidate Skolem function from the trained GCLN network.
    Obtains Phi_prime when accuracy does not reach 100%
    Obtains counts of Clauses and Literal using z3py
    '''

    if args.cnf == 'cnf':
        if args.architecture == 1:
            skf_dict = {}
            temp_dict = {}
            for i in range(len(model)):
                skolem_function, temp_dict_ = util.get_skolem_function_cnf_2(
                    args, model[i], num_of_inputs, input_var_idx, num_of_outputs, output_var_idx, io_dictz3, i)
                temp_dict.update(temp_dict_)
                skf_dict[Yvar[i]] = temp_dict[skolem_function[0]]
            skf_list = list(skf_dict.values())
            skf_list = get_phi_prime_skf_list(
                skf_list, training_samples, disagreed_indices, input_var_idx, output_var_idx)

            skfs = '\n'.join(skf_list)
            f = open('experiments/bnsynth_skfs/' +
                     args.verilog_spec[:-2]+'.skf', 'w')
            f.write(skfs)
            f.close()
            bnsynth_time = time.time() - start_time
            f = open('experiments/bnsynth_skfs/' +
                     args.verilog_spec[:-2]+'.skf', 'w')
            f.write(skfs)
            f.close()
        elif args.architecture == 2:
            skolem_function, temp_dict = util.get_skolem_function_cnf_2(
                args, model, num_of_inputs, input_var_idx, num_of_outputs, output_var_idx, io_dictz3, 0)
            skf_dict = {}
            for i in range(num_of_outputs):
                skf_dict[Yvar[i]] = temp_dict[skolem_function[i]]
            skf_list = list(skf_dict.values())

            if len(disagreed_indices) > 0:
                skf_list = get_phi_prime_skf_list(
                    skf_list, training_samples, disagreed_indices, input_var_idx, output_var_idx)

            skfs = '\n'.join(skf_list)
            f = open('experiments/bnsynth_skfs/' +
                     args.verilog_spec[:-2]+'.skf', 'w')
            f.write(skfs)
            f.close()
            bnsynth_time = time.time() - start_time
            f = open('experiments/bnsynth_skfs/' +
                     args.verilog_spec[:-2]+'.skf', 'w')
            f.write(skfs)
            f.close()
            bnsynth_clause_counts, num_inputs_bnsynth = get_bnsynth_counts(
                args, num_of_outputs, total_varsz3, Xvar)
        else:
            skolem_function, temp_dict = util.get_skolem_function_cnf_2(
                args, model, num_of_inputs, input_var_idx, num_of_outputs, output_var_idx, io_dictz3, 0)
            skf_dict = {}
            for i in range(num_of_outputs):
                skf_dict[Yvar[i]] = temp_dict[skolem_function[i]]
            skf_list = list(skf_dict.values())

            if len(disagreed_indices) > 0:
                skf_list = get_phi_prime_skf_list(
                    skf_list, training_samples, disagreed_indices, input_var_idx, output_var_idx)

            skfs = '\n'.join(skf_list)
            f = open('experiments/bnsynth_skfs/' +
                     args.verilog_spec[:-2]+'.skf', 'w')
            f.write(skfs)
            f.close()
            bnsynth_time = time.time() - start_time
            f = open('experiments/bnsynth_skfs/' +
                     args.verilog_spec[:-2]+'.skf', 'w')
            f.write(skfs)
            f.close()
            bnsynth_clause_counts, num_inputs_bnsynth = get_bnsynth_counts(
                args, num_of_outputs, total_varsz3, Xvar)
    else:
        if args.architecture == 1:
            skf_dict = {}
            temp_dict = {}
            for i in range(len(model)):
                skolem_function, temp_dict_ = util.get_skolem_function_dnf(
                    args, model[i], num_of_inputs, input_var_idx, num_of_outputs, output_var_idx, io_dictz3, i)
                temp_dict.update(temp_dict_)
                skf_dict[Yvar[i]] = temp_dict[skolem_function[0]]
            skf_list = list(skf_dict.values())
            skf_list = get_phi_prime_skf_list(
                skf_list, training_samples, disagreed_indices, input_var_idx, output_var_idx)

            skfs = '\n'.join(skf_list)
            bnsynth_time = time.time() - start_time
            f = open('experiments/bnsynth_skfs/' +
                     args.verilog_spec[:-2]+'.skf', 'w')
            f.write(skfs)
            f.close()
            bnsynth_clause_counts, num_inputs_bnsynth = get_bnsynth_counts(
                args, num_of_outputs, total_varsz3, Xvar)

        elif args.architecture == 2:
            skolem_function, temp_dict = util.get_skolem_function_dnf(
                args, model, num_of_inputs, input_var_idx, num_of_outputs, output_var_idx, io_dictz3, 0)
            skf_dict = {}
            print("skolem function: ", skolem_function)
            for i in range(num_of_outputs):
                skf_dict[Yvar[i]] = temp_dict[skolem_function[i]]
            skf_list = list(skf_dict.values())

            if len(disagreed_indices) > 0:
                skf_list = get_phi_prime_skf_list(
                    skf_list, training_samples, disagreed_indices, input_var_idx, output_var_idx)

            skfs = '\n'.join(skf_list)
            f = open('experiments/bnsynth_skfs/' +
                     args.verilog_spec[:-2]+'.skf', 'w')
            f.write(skfs)
            f.close()
            bnsynth_time = time.time() - start_time
            f = open('experiments/bnsynth_skfs/' +
                     args.verilog_spec[:-2]+'.skf', 'w')
            f.write(skfs)
            f.close()
        else:
            skolem_function, temp_dict = util.get_skolem_function_dnf(
                args, model, num_of_inputs, input_var_idx, num_of_outputs, output_var_idx, io_dictz3, 0)
            skf_dict = {}
            for i in range(num_of_outputs):
                skf_dict[Yvar[i]] = temp_dict[skolem_function[i]]
            skf_list = list(skf_dict.values())

            if len(disagreed_indices) > 0:
                skf_list = get_phi_prime_skf_list(
                    skf_list, training_samples, disagreed_indices, input_var_idx, output_var_idx)

            skfs = '\n'.join(skf_list)
            f = open('experiments/bnsynth_skfs/' +
                     args.verilog_spec[:-2]+'.skf', 'w')
            f.write(skfs)
            f.close()
            bnsynth_time = time.time() - start_time
            f = open('experiments/bnsynth_skfs/' +
                     args.verilog_spec[:-2]+'.skf', 'w')
            f.write(skfs)
            f.close()
            bnsynth_clause_counts, num_inputs_bnsynth = get_bnsynth_counts(
                args, num_of_outputs, total_varsz3, Xvar)

    var_def = ""
    assigns = ""
    for i, (k, v) in enumerate(temp_dict.items()):
        var_def += "wire " + k + ";\n"
        assigns += "assign " + k + " = " + v + ";\n"
    temp_content = var_def + assigns
    if args.postprocessor == 1:
        inputfile_name = args.verilog_spec.split('.v')[0]
        # Write the error formula in verilog
        util.write_error_formula1(inputfile_name, args.verilog_spec,
                                  verilogformula, skf_list, temp_content, Xvar, Yvar, total_varsz3, PosUnate, NegUnate)

        # sat call to errorformula:
        check, sigma, ret = util.verify(Xvar, Yvar, args.verilog_spec)
        is_valid = 0
        counter_examples = []
        if check == 0:
            print("error...ABC network read fail")
            print("Skolem functions not generated")
            print("not solved !!")
        else:
            if ret == 0:
                print('\n\nError Formula UNSAT... Skolem functions generated')

                # ============================================================
                bnsynth_clause_counts, num_inputs_bnsynth = get_bnsynth_counts(
                    args, num_of_outputs, total_varsz3, Xvar)
                # ============================================================
                print('-----------------------------------------------------')
                print_skolem_function(args)
                print('-----------------------------------------------------')
                # ============================================================
                manthan_start_time = time.time()
                run_manthan(args)
                manthan_end_time = time.time()
                manthan_time = manthan_end_time - manthan_start_time
                manthan_clause_counts, num_inputs_manthan = get_manthan_counts(
                    args, num_of_outputs, io_dict, Xvar)
                # ============================================================

                is_valid = 1
                skfunc = [sk.replace('\n', '') for sk in skf_list]
                t = time.time() - start_time
                datastring = str(args.verilog_spec)+", "+str(args.architecture)+", "+str(args.cnf)+", "+str(args.layers)+", "+str(epochs)+", "+str(args.batch_size)+", "+str(args.learning_rate)+", "+str(args.K)+", "+str(len(input_var_idx)) + \
                    ", "+str(num_of_outputs)+", "+str(num_of_ce)+", "+"Valid" + \
                    ", "+str(bnsynth_time)+", "+str(manthan_time)+", "+str(final_loss)+", " + \
                    str(loss_drop)+", "+str(accuracy)+", " + \
                    bnsynth_clause_counts[0]+", "+bnsynth_clause_counts[1]+", " + \
                    bnsynth_clause_counts[2]+", "+bnsynth_clause_counts[3]+", "+str(num_inputs_bnsynth)+", "+bnsynth_clause_counts[4]+", "+bnsynth_clause_counts[5]+", " + \
                    manthan_clause_counts[0]+", "+manthan_clause_counts[1]+", " + \
                    manthan_clause_counts[2]+", " + \
                    manthan_clause_counts[3]+", " + \
                    str(num_inputs_manthan)+", " + \
                    manthan_clause_counts[4]+", "+manthan_clause_counts[5]+"\n"
                if args.cnf == 'cnf':
                    print("\nComparing CNF formulae for BNSynth with Manthan")
                    print(bnsynth_clause_counts)
                    print(tabulate([['Manthan', manthan_time, manthan_clause_counts[1], manthan_clause_counts[5], num_inputs_manthan],
                                    ['BNSynth', bnsynth_time, bnsynth_clause_counts[1], bnsynth_clause_counts[5], num_inputs_bnsynth], [
                        'Improvement factor', int(manthan_time)/int(bnsynth_time), int(manthan_clause_counts[1])/int(bnsynth_clause_counts[1]), int(manthan_clause_counts[5])/int(bnsynth_clause_counts[5]), int(num_inputs_manthan)/int(num_inputs_bnsynth)]], headers=['', 'T', 'C', 'L', 'I'], tablefmt='psql'))
                else:
                    print(
                        "At this point we don't compare BNSynth and Manthan for DNF formulae")
                f = open(args.output_file, "a")
                f.write(datastring)
                f.close()
                f = open("experiments/check", "w")
                f.write("OK")
                f.close()
                # os.system(
                #     'rm data/benchmarks/cav20_manthan_dataset/verilog/*.cnf')
                os.unlink('experiments/simplified.skf')
            else:
                counter_examples = torch.from_numpy(
                    np.concatenate(
                        (sigma.modelx, sigma.modely)
                    ).reshape((1, len(Xvar)+len(Yvar)))
                )
                f = open("experiments/check", "w")
                f.write("NOT OK")
                f.close()

    elif args.postprocessor == 2:

        print("skolem functions: ", skf_list)
        verilogformula = util.convert_verilog(
            "data/benchmarks/"+args.verilog_spec_location+"/"+args.verilog_spec, 0)
        inputfile_name = ("data/benchmarks/"+args.verilog_spec_location +
                          "/"+args.verilog_spec).split('/')[-1][:-8]
        verilog = inputfile_name+".v"

        # Write the error formula in verilog
        util.write_error_formula2(
            inputfile_name, verilog, verilogformula, skf_list, Xvar, Yvar, PosUnate, NegUnate)

        # sat call to errorformula:
        check, sigma, ret = util.verify(Xvar, Yvar, verilog)
        print(check, ret)
        if check == 0:
            print("error...ABC network read fail")
            print("Skolem functions not generated")
            print("not solved !!")
            is_valid = 0
            # exit()

        if ret == 0:
            print('error formula unsat.. skolem functions generated')
            print("success")
            is_valid = 1
            skfunc = [sk.replace('\n', '') for sk in skf_list]
            t = time.time() - start_time
            datastring = str(args.verilog_spec)+", "+str(epochs)+", "+str(args.batch_size)+", "+str(args.learning_rate)+", "+str(args.K)+", "+str(len(input_var_idx)) + \
                ", "+str(num_of_outputs)+", "+str(num_of_ce)+", "+'; '.join(skfunc)+", "+"Valid" + \
                ", "+str(t)+", "+str(final_loss)+", " + \
                str(loss_drop)+", "+str(accuracy)+"\n"
            print(datastring)
            f = open(args.output_file, "a")
            f.write(datastring)
            f.close()
            counter_examples = []

        else:
            counter_examples = torch.from_numpy(
                np.concatenate(
                    (sigma.modelx, sigma.modely)
                ).reshape((1, len(Xvar)+len(Yvar)))
            )
            print("counter examples: ", counter_examples)

    return skf_list, is_valid, counter_examples
