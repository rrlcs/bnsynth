import os
import time
import torch
import numpy as np
from code.utils.utils import util


def postprocess(args, model, accuracy, epochs, final_loss, loss_drop, verilogformula, num_of_inputs, input_var_idx, num_of_outputs,
                output_var_idx, io_dict, Xvar, Yvar, PosUnate, NegUnate, start_time, rem_formula, rem_inp_formula, num_of_ce):

    if args.cnf:
        if args.architecture == 1:
            skf_dict = {}
            temp_dict = {}
            for i in range(len(model)):
                skolem_function, temp_dict_ = util.get_skolem_function_cnf_2(
                    args, model[i], num_of_inputs, input_var_idx, num_of_outputs, output_var_idx, io_dict, i)
                skf_dict[Yvar[i]] = skolem_function[0]
                temp_dict.update(temp_dict_)
            skf_list = list(skf_dict.values())
            print("skf_list: ", skf_list)
            phi = skf_list[0]
            phi_new = rem_formula + " & " + \
                "(~("+rem_inp_formula+") | " + phi+")"
            print("final skolem function: ", phi_new)
            # skf_list[0] = phi_new
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
                print("i", i)
                skolem_function, temp_dict_ = util.get_skolem_function_dnf(
                    args, model[i], num_of_inputs, input_var_idx, num_of_outputs, output_var_idx, io_dict, i)
                skf_dict[Yvar[i]] = skolem_function[0]
                temp_dict.update(temp_dict_)
            skf_list = list(skf_dict.values())
            print("skf_list: ", skf_list)
            print("temp_dict: ", temp_dict)
            # phi = skf_list[0]
            # phi_new = rem_formula + " & " + \
            #     "(~("+rem_inp_formula+") | " + phi+")"
            # print("final skolem function: ", phi_new)
            # print("final skfs: ", skf_list)
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

    if args.postprocessor == 1:
        inputfile_name = args.verilog_spec.split('.v')[0]

        # Write the error formula in verilog
        util.write_error_formula1(inputfile_name, args.verilog_spec,
                                  verilogformula, skf_list, temp_content, Xvar, Yvar, PosUnate, NegUnate)

        # sat call to errorformula:
        check, sigma, ret = util.verify(Xvar, Yvar, args.verilog_spec)
        print(check, ret)
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
                    ", "+str(num_of_outputs)+", "+str(num_of_ce)+", "+'; '.join(skfunc)+", "+"Valid" + \
                    ", "+str(t)+", "+str(final_loss)+", " + \
                    str(loss_drop)+", "+str(accuracy)+"\n"
                print(datastring)
                f = open(args.output_file, "a")
                f.write(datastring)
                f.close()
                os.system(
                    'rm data/benchmarks/cav20_manthan_dataset/verilog/*.cnf')
            else:
                counter_examples = torch.from_numpy(
                    np.concatenate(
                        (sigma.modelx, sigma.modely)
                    ).reshape((1, len(Xvar)+len(Yvar)))
                )
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
