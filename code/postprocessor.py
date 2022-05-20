import os
import time
from matplotlib.pyplot import cla
import torch
import subprocess
import numpy as np
from code.utils.utils import util
from experiments.visitors.verilog2z3 import preparez3


def get_formula_for_uncovered_data(training_samples, disagreed_index, input_var_idx, output_var_idx, current_output=0):
    print(disagreed_index)
    uncovered_samples = training_samples[disagreed_index]
    print("run.py ", training_samples[disagreed_index], input_var_idx)
    print(uncovered_samples[:, input_var_idx])
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
        print("FINAL: ", final_rem_formula[:-3],
              "final rem inp:", final_rem_inp_formula[:-3])
    else:
        final_rem_formula = ""
        final_rem_inp_formula = ""

    return final_rem_formula[:-3], final_rem_inp_formula[:-3]


def get_phi_prime_skf_list(skf_list, training_samples, disagreed_indices, input_var_idx, output_var_idx):
    for i in range(len(skf_list)):
        rem_formula, rem_inp_formula = get_formula_for_uncovered_data(
            training_samples, disagreed_indices[i], input_var_idx, output_var_idx, current_output=i)
        phi = skf_list[i]
        if len(rem_formula) > 0 and len(rem_inp_formula) > 0:
            print("remaining table added to partial formula")
            phi_new = rem_formula + " & " + \
                "(~("+rem_inp_formula+") | " + phi+")"
        else:
            phi_new = phi
        print("final skolem function: ", phi_new, "phi: ", phi)
        skf_list[i] = phi_new

    return skf_list


def postprocess(args, model, accuracy, epochs, final_loss, loss_drop, verilogformula, total_varsz3, num_of_inputs, input_var_idx, num_of_outputs,
                output_var_idx, io_dict, io_dictz3, Xvar, Yvar, PosUnate, NegUnate, start_time, training_samples, disagreed_indices, num_of_ce):

    if args.cnf:
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
            path = 'data/benchmarks/'+args.verilog_spec_location+"/"
            preparez3(args.verilog_spec,
                      path, num_of_outputs)

            cmd = 'python experiments/visitors/z3ClauseCounter.py'
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
            out, err = p.communicate()
            # print("++++++++++cnf++++++++++++", out.decode("utf-8"))
            inp_vars = [total_varsz3[i].replace('_', '') for i in Xvar]
            # print("inp_vars: ", inp_vars)
            ftext = out.decode("utf-8").split('\n')[1]
            num_inputs_bnsynth = 0
            for v in inp_vars:
                if v in ftext:
                    num_inputs_bnsynth += 1
            bnsynth_clause_counts = out.decode("utf-8").split('\n')[2:-1]
            print("BNSynth clause counts: ", bnsynth_clause_counts)
            # print(out.decode('UTF-8'))
            f = open('experiments/simplified.skf', 'r')
            simple_skf = f.read()
            f.close()
            f = open('experiments/' +
                     args.verilog_spec[:-2]+'_simplified.skf', 'w')
            f.write(simple_skf)
            f.close()
        elif args.architecture == 2:
            skf_list, temp_dict = util.get_skolem_function_cnf_2(
                args, model, num_of_inputs, input_var_idx, num_of_outputs, output_var_idx, io_dict, 0)
        else:
            skf_list, temp_dict = util.get_skolem_function_cnf_2(
                args, model, num_of_inputs, input_var_idx, num_of_outputs, output_var_idx, io_dict, 0)
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
            path = 'data/benchmarks/'+args.verilog_spec_location+"/"
            preparez3(args.verilog_spec,
                      path, num_of_outputs)
            cmd = 'python experiments/visitors/z3ClauseCounter.py'

            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
            out, err = p.communicate()
            # print("++++++++++++++++++++++", out.decode("utf-8"))
            inp_vars = [total_varsz3[i].replace('_', '') for i in Xvar]
            # print("inp_vars: ", inp_vars)
            ftext = out.decode("utf-8").split('\n')[1]
            num_inputs_bnsynth = 0
            for v in inp_vars:
                if v in ftext:
                    num_inputs_bnsynth += 1
            bnsynth_clause_counts = out.decode("utf-8").split('\n')[2:-1]
            print("BNSynth clause counts: ", bnsynth_clause_counts)
            f = open('experiments/simplified.skf', 'r')
            simple_skf = f.read()
            f.close()
            f = open('experiments/' +
                     args.verilog_spec[:-2]+'_simplified.skf', 'w')
            f.write(simple_skf)
            f.close()

        elif args.architecture == 2:
            skf_list = util.get_skolem_function_dnf(
                args, model, num_of_inputs, input_var_idx, num_of_outputs, output_var_idx, io_dict)
        else:
            skf_list = util.get_skolem_function_dnf(
                args, model, num_of_inputs, input_var_idx, num_of_outputs, output_var_idx, io_dict)

    var_def = ""
    assigns = ""
    for i, (k, v) in enumerate(temp_dict.items()):
        var_def += "wire " + k + ";\n"
        assigns += "assign " + k + " = " + v + ";\n"
    temp_content = var_def + assigns
    # print(var_def+assigns)
    # print(assigns)
    # exit()
    if args.postprocessor == 1:
        inputfile_name = args.verilog_spec.split('.v')[0]
        # print("verilogformula: ", verilogformula)
        # Write the error formula in verilog
        util.write_error_formula1(inputfile_name, args.verilog_spec,
                                  verilogformula, skf_list, temp_content, Xvar, Yvar, total_varsz3, PosUnate, NegUnate)

        # sat call to errorformula:
        check, sigma, ret = util.verify(Xvar, Yvar, args.verilog_spec)
        print(check, ret)
        manthan_start_time = time.time()
        cmd1 = 'python manthan.py --seed 1 --varlist data/benchmarks/final_custom_benchmarks/verilog/Yvarlist/' + \
            args.verilog_spec[:-2]+'_varstoelim.txt ' + \
            '--verilog data/benchmarks/final_custom_benchmarks/verilog/'+args.verilog_spec

        p = subprocess.Popen(cmd1, stdout=subprocess.PIPE, shell=True)
        out, err = p.communicate()
        manthan_end_time = time.time()
        manthan_time = manthan_end_time - manthan_start_time
        print("Manthan outputs: ", out.decode('UTF-8'))

        path = 'experiments/manthan_skfs/'
        preparez3(args.verilog_spec[:-2]+'_skolem.v',
                  path, num_of_outputs, manthan=1)
        # exit()
        cmd2 = 'python experiments/visitors/z3ClauseCounter.py'

        p = subprocess.Popen(cmd2, stdout=subprocess.PIPE, shell=True)
        out, err = p.communicate()
        # print("++++++++++++++++++++++", out.decode("utf-8"))
        ftext = out.decode("utf-8").split('\n')[1]
        num_inputs_manthan = 0
        for v in inp_vars:
            if v in ftext:
                num_inputs_manthan += 1
        manthan_clause_counts = out.decode("utf-8").split('\n')[2:-1]
        print("Manthan clause counts: ", manthan_clause_counts)

        is_valid = 0
        counter_examples = []
        if check == 0:
            print("error...ABC network read fail")
            print("Skolem functions not generated")
            print("not solved !!")
            # exit()
        else:
            if ret == 0:
                print('error formula unsat.. skolem functions generated')
                print("success")
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
                print(datastring)
                f = open(args.output_file, "a")
                f.write(datastring)
                f.close()
                f = open("experiments/check", "w")
                f.write("OK")
                f.close()
                # os.system(
                #     'rm data/benchmarks/cav20_manthan_dataset/verilog/*.cnf')
            else:
                counter_examples = torch.from_numpy(
                    np.concatenate(
                        (sigma.modelx, sigma.modely)
                    ).reshape((1, len(Xvar)+len(Yvar)))
                )
                f = open("experiments/check", "w")
                f.write("NOT OK")
                f.close()
                print("counter examples: ", counter_examples)

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
