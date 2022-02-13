import importlib
import os
import tempfile
import time
from code.ce_train import ce_train_loop
from code.train import train
from code.utils import getSkolemFunc as skf
from code.utils import getSkolemFunc4z3 as skfz3
from code.utils import plot as pt
from code.utils.generateSamples_manthan import *
from code.utils.preprocess_manthan import *
from code.utils.utils import util

import networkx as nx
import numpy as np
import torch

# from benchmarks import z3ValidityChecker as z3
# from benchmarks.verilog2z3 import preparez3
from data.dataLoader import dataLoader

# if __name__ == "__main__":

#     # Get Argument Parser
#     parser = util.make_arg_parser()
#     args = parser.parse_args()

#     # Set device
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'

    start_time = time.time()
    # ----------------------------------------------------------------------------------------------------------
    # Manthan 2 Preprocessor
    # Xvar, Yvar, qdimacs_list = parse("data/benchmarks/"+args.verilog_spec_location+"/"+args.verilog_spec)
    # print("count X variables", len(Xvar))
    # print("count Y variables", len(Yvar))
    # print(qdimacs_list)
    # print(Xvar, Yvar)
    
    # all_var = Xvar + Yvar
    # total_vars = ["i"+str(v) for v in all_var]
    # output_varlist = ["i"+str(v) for v in Yvar]

    # inputfile_name = args.verilog_spec[:-8]
    # cnffile_name = tempfile.gettempdir()+"/"+inputfile_name+".cnf"

    # cnfcontent = convertcnf("data/benchmarks/"+args.verilog_spec_location+"/"+args.verilog_spec, cnffile_name)
    # cnfcontent = cnfcontent.strip("\n")+"\n"

    # # finding unates:
    # print("preprocessing: finding unates (constant functions)")
    # start_t = time.time()
    # if len(Yvar) > 0:
    #     PosUnate, NegUnate = preprocess(cnffile_name)
    # else:
    #     print("too many Y variables, let us proceed with Unique extraction\n")
    #     PosUnate = []
    #     NegUnate = []
    # end_t = time.time()
    # print("preprocessing time:", str(end_t-start_t))

    # print("count of positive unates", len(PosUnate))
    # print("count of negative unates", len(NegUnate))
    # print("positive unates", PosUnate)
    # print("negative unates", NegUnate)

    # Unates = PosUnate + NegUnate

    # for yvar in PosUnate:
    #     qdimacs_list.append([yvar])
    #     cnfcontent += "%s 0\n" % (yvar)

    # for yvar in NegUnate:
    #     qdimacs_list.append([-1 * int(yvar)])
    #     cnfcontent += "-%s 0\n" % (yvar)

    # end_time = time.time()
    # if len(Unates) == len(Yvar):
    #     print(PosUnate)
    #     print(NegUnate)
    #     print("all Y variables are unates and have constant functions")
    #     info = str(args.verilog_spec)+", "+str(len(Xvar))+", "+str(len(Yvar))+", "+"All Unates"+", "+str(end_time-start_time)+"\n"
    #     f = open("qdimacsinfo.csv", "a")
    #     f.write(info)
    #     f.close()
    #     # exit()
    #     # skolemfunction_preprocess(
    #     #     Xvar, Yvar, PosUnate, NegUnate, [], '', inputfile_name)
    
    #     # logtime(inputfile_name, "totaltime:"+str(end_time-start_time))
    #     # exit()
    # print("Preprocessing Time: ", end_time-start_time)
    
    # # Logging
    # info = str(args.verilog_spec)+", "+str(len(Xvar))+", "+str(len(Yvar))+", "+"Not All Unates"+", "+str(end_time-start_time)+"\n"
    # f = open("qdimacsinfo.csv", "a")
    # f.write(info)
    # f.close()
    # exit("success")
    
    # ==========================================================================================
    # Start Preprocessing
    # MANTHAN 1 PREPROCESSOR WRAPPER
    # start_time = time.time()
    # pre_t_s = time.time()
    # verilog, output_varlist, total_vars, total_varsz3,\
    #      verilog_formula, pos_unate, neg_unate, Xvar,\
    #           Yvar, Xvar_map, Yvar_map = util.preprocess_wrapper(
    #               args.verilog_spec, args.verilog_spec_location
    #               )
    
    # End Preprocessing
    
    # info = str(args.verilog_spec)+", "+str(len(Xvar))+", "+str(len(Yvar))+", "+"All Unates"+", "+str(pre_t_e - pre_t_s)+"\n"
    # f = open("qdimacsinfo.csv", "a")
    # f.write(info)
    # f.close()
    # print(output_varlist, total_vars)
    # ----------------------------------------------------------------------------------------------------------
    # exit()

    # def convert_verilog(input,cluster,dg):
    #     # ng = nx.Graph() # used only if args.multiclass

    #     with open(input, 'r') as f:
    #         lines = f.readlines()
    #     f.close()
    #     itr = 1
    #     declare = 'module FORMULA( '
    #     declare_input = ''
    #     declare_wire = ''
    #     assign_wire = ''
    #     tmp_array = []

    #     for line in lines:
    #         line = line.strip(" ")
    #         if (line == "") or (line == "\n"):
    #             continue
    #         if line.startswith("c "):
    #             continue

    #         if line.startswith("p "):
    #             continue


    #         if line.startswith("a"):
    #             a_variables = line.strip("a").strip("\n").strip(" ").split(" ")[:-1]
    #             for avar in a_variables:
    #                 declare += "%s," %(avar)
    #                 declare_input += "input %s;\n" %(avar)
    #             continue

    #         if line.startswith("e"):
    #             e_variables = line.strip("e").strip("\n").strip(" ").split(" ")[:-1]
    #             for evar in e_variables:
    #                 tmp_array.append(int(evar))
    #                 declare += "%s," %(evar)
    #                 declare_input += "input %s;\n" %(evar)
    #                 if int(evar) not in list(dg.nodes):
    #                     dg.add_node(int(evar))
    #             continue

    #         declare_wire += "wire t_%s;\n" %(itr)
    #         assign_wire += "assign t_%s = " %(itr)
    #         itr += 1

    #         clause_variable = line.strip(" \n").split(" ")[:-1]
    #         for var in clause_variable:
    #             if int(var) < 0:
    #                 assign_wire += "~%s | " %(abs(int(var)))
    #             else:
    #                 assign_wire += "%s | " %(abs(int(var)))

    #         assign_wire = assign_wire.strip("| ")+";\n"
            
    #         ### if args.multiclass, then add an edge between variables of the clause ###

    #         # if cluster:
    #         #     for literal1 in clause_variable:
    #         #         literal1 = abs(int(literal1))
    #         #         if literal1 in tmp_array:
    #         #             if literal1 not in list(ng.nodes):
    #         #                 ng.add_node(literal1)
    #         #             for literal2 in clause_variable:
    #         #                 literal2 = abs(int(literal2))
    #         #                 if (literal1 != abs(literal2)) and (literal2 in tmp_array):
    #         #                     if literal2 not in list(ng.nodes):
    #         #                         ng.add_node(literal2)
    #         #                     if not ng.has_edge(literal1, literal2):
    #         #                         ng.add_edge(literal1,literal2)



    #     count_tempvariable = itr

    #     declare += "out);\n"
    #     declare_input += "output out;\n"

    #     temp_assign = ''
    #     outstr = ''

    #     itr = 1
    #     while itr < count_tempvariable:
    #         temp_assign += "t_%s & " %(itr)
    #         if itr % 100 == 0:
    #             declare_wire += "wire tcount_%s;\n" %(itr)
    #             assign_wire += "assign tcount_%s = %s;\n" %(itr,temp_assign.strip("& "))
    #             outstr += "tcount_%s & " %(itr)
    #             temp_assign = ''
    #         itr += 1

    #     if temp_assign != "":
    #         declare_wire += "wire tcount_%s;\n" %(itr)
    #         assign_wire += "assign tcount_%s = %s;\n" %(itr,temp_assign.strip("& "))
    #         outstr += "tcount_%s;\n" %(itr)
    #     outstr = "assign out = %s" %(outstr)


    #     verilogformula = declare + declare_input + declare_wire + assign_wire + outstr +"endmodule\n"

    #     return verilogformula
    # dg = nx.Graph()
    # verilogformula = convert_verilog("data/benchmarks/"+args.verilog_spec_location+"/"+args.verilog_spec, 0)
    # inputfile_name = ("data/benchmarks/"+args.verilog_spec_location+"/"+args.verilog_spec).split('/')[-1][:-8]
    # verilog = inputfile_name+".v"
    # f = open(verilog, "r")
    # verilogformula = f.readlines()
    # verilogformula = ''.join(verilogformula[1:])
    # f.close()
    # print("Num of unates: ", len(pos_unate)+len(neg_unate))

    # ----------------------------------------------------------------------------------------------------------
    # TO DO:
    # 1. USE THE UNATES TO CONSTRUCT UNATE_SKOLEMFORMULA
    
    # result = util.check_unates(pos_unate, neg_unate, Xvar, Yvar, args.verilog_spec[:-2])
    # if result:
    #     unate_data = str(args.verilog_spec)+", "+str(len(Xvar))+", "+str(len(Yvar))+", "+"All Unates"+"\n"
    #     f = open("unates.csv", "a")
    #     f.write(unate_data)
    #     f.close()
    #     exit("All Unates!")
    # else:
    #     unate_data = str(args.verilog_spec)+", "+str(len(Xvar))+", "+str(len(Yvar))+", "+"All not Unates"+"\n"
    #     f = open("unates.csv", "a")
    #     f.write(unate_data)
    #     f.close()
    #     exit("All not Unates!")
    # exit() 
    # ----------------------------------------------------------------------------------------------------------


    # ----------------------------------------------------------------------------------------------------------
    # to sample, we need a cnf file and variable mapping coressponding to
    # varilog variables
    # FROM TRIAL
    # data_t_s = time.time()
    # cnf_content, allvar_map = util.prepare_cnf_content(
    #     verilog, Xvar, Yvar, Xvar_map, Yvar_map, pos_unate, neg_unate
    #     )


    # maxsamples = 0
    # sampling_cnf = cnfcontent
    # if not maxsamples:
    #     if len(Xvar) > 4000:
    #         num_samples = 1000
    #     if (len(Xvar) > 1200) and (len(Xvar) <= 4000):
    #         num_samples = 5000
    #     if len(Xvar) <= 1200:
    #         num_samples = 10000
    # else:
    #     num_samples = maxsamples
    
    # weighted = 1
    # adaptivesample = 0

    # if weighted:
    #     sampling_weights_y_1 = ''
    #     sampling_weights_y_0 = ''
    #     for xvar in Xvar:
    #         sampling_cnf += "w %s 0.5\n" % (xvar)
    #     for yvar in Yvar:
    #         # if yvar in UniqueVars:
    #         #     sampling_cnf += "w %s 0.5\n" % (yvar)
    #         #     continue
    #         if (yvar in PosUnate) or (yvar in NegUnate):
    #             continue

    #         sampling_weights_y_1 += "w %s 0.5\n" % (yvar)
    #         sampling_weights_y_0 += "w %s 0.1\n" % (yvar)

    #     if adaptivesample:
    #         weighted_sampling_cnf = computeBias(
    #             Xvar, Yvar, sampling_cnf, sampling_weights_y_1, sampling_weights_y_0, inputfile_name, Unates, args)
    #     else:
    #         weighted_sampling_cnf = sampling_cnf + sampling_weights_y_1
    #     # print(weighted_sampling_cnf)
    #     print("generating weighted samples")
    #     samples = generatesample(
    #         args, num_samples, weighted_sampling_cnf, inputfile_name, 1)
    # else:
    #     print("generating uniform samples")
    #     samples = generatesample(
    #         args, num_samples, sampling_cnf, inputfile_name, 0)

    # samples = np.array([[1, 0], [0, 1]])
    
# <<<<<<< HEAD
    # print(samples, Xvar)
    # Xvar_tmp = [i-1 for i in Xvar]
    # x_data, indices = np.unique(samples[:, Xvar_tmp], axis=0, return_index=True)
    # samples = samples[indices, :]
    # print("samples: ", samples.shape)
    # print(samples)
    # end_t = time.time()
    # exit()

    # generate sample
    # samples = util.generate_samples(
    #     cnf_content, Xvar, Yvar, Xvar_map, Yvar_map, allvar_map,verilog,
    #     max_samples=args.training_size
    #     )
    
    # print("samples: ", samples)

    # Repeat or add noise to get larger dataset
    # # 
    # samples = np.array([[0,1],[1,1]])
    # print(samples)
    # training_samples = util.make_dataset_larger(samples)
    # training_samples = torch.from_numpy(samples[:100, :]).to(torch.double)
    # print(training_samples.shape)
# =======

    # Get train test split
    # training_set, validation_set = util.get_train_test_split(training_samples)
    # print("Total, Train, and Valid shapes", training_samples.shape,
    #       training_set.shape, validation_set.shape)

    data_t_e = time.time()
    # print("Data Sampling Time: ", data_t_e - data_t_s)

    # num_of_vars, num_out_vars = len(Xvar)+len(Yvar), len(Yvar)

    # num_of_vars, num_out_vars, num_of_eqns = util.get_var_counts(Xvar, Yvar, verilog)
    # print("No. of vars: {}, No. of output vars: {}, No. of eqns: {}".format(num_of_vars, num_out_vars, num_of_eqns))
    # ----------------------------------------------------------------------------------------------------------

    
    # ----------------------------------------------------------------------------------------------------------
    # # Prepare input output dictionaries
    # io_dict, io_dictz3 = util.prepare_io_dicts(total_vars, total_varsz3=[])
    # print("io dict: ", io_dict)

    # # Obtain variable indices
    # var_indices, input_var_idx, output_var_idx = util.get_var_indices(num_of_vars, output_varlist, io_dict)
    # input_size = 2*len(input_var_idx)
    # print("Input size: ", input_size)
    # print("Output size: ", len(output_var_idx))

    # store_preprocess_time(args.verilog_spec, num_of_vars, num_out_vars, num_of_eqns, 
    # args.epochs, args.no_of_samples, preprocess_time)
    
    # if args.run_for_all_outputs == 1:
    #     num_of_outputs = len(output_var_idx)
    # else:
    #     num_of_outputs = 1
    # ----------------------------------------------------------------------------------------------------------
    # print("out size: ", num_of_outputs)

    # ----------------------------------------------------------------------------------------------------------
    # load data
    # train_loader = dataLoader(training_set, args.training_size, args.P, input_var_idx,
    #                           output_var_idx, num_of_outputs, args.threshold, args.batch_size)
    # validation_loader = dataLoader(validation_set, args.training_size, args.P, input_var_idx,
    #                                output_var_idx, num_of_outputs, args.threshold, args.batch_size)
    # validation_loader = []
    # ----------------------------------------------------------------------------------------------------------


    # Initialize skolem funcition with empty string
    # skolem_functions = ""

    # ----------------------------------------------------------------------------------------------------------
    # TRAINING MODEL
    train_t_s = time.time()
# <<<<<<< HEAD
# =======
    # num_of_outputs = 1
    skf_dict_z3 = {}
    skf_dict_verilog = {}
# For single output
    # final_accuracy = 0
    # final_epochs = 0
    # for i in range(len(Yvar)):
    # i=0
# For single output
# =======
    i = 0
# >>>>>>> shared
    current_output = i
# >>>>>>> trial
    gcln, train_loss, valid_loss, accuracy, epochs = train(
        args.P, args.train, train_loader, validation_loader, args.learning_rate, args.epochs, 
        input_size, num_of_outputs, current_output, args.K, device, num_of_vars, input_var_idx, output_var_idx, 
        io_dict, io_dictz3, args.threshold, args.verilog_spec, args.verilog_spec_location, 
        Xvar, Yvar, verilog_formula=[], verilog=[], pos_unate=[], neg_unate=[]
        )
    train_t_e = time.time()
    print("Training Time: ", train_t_e - train_t_s)
    print("accuracy run: ", accuracy)
    # ----------------------------------------------------------------------------------------------------------
# For single output
    # final_accuracy += accuracy
    # final_epochs += epochs
# For single output
# =======

    final_loss = train_loss[-1]
    loss_drop = train_loss[0] - train_loss[-1]

# # >>>>>>> shared
#     util.store_losses(train_loss, valid_loss)
#     pt.plot()

#     # ----------------------------------------------------------------------------------------------------------
#     # Checking Skolem Function using Z3
    extract_t_s = time.time()
#     # Skolem function in z3py format
# # <<<<<<< HEAD
# # For single output
#     # skfunc = skfz3.get_skolem_function(
#     #     gcln, num_of_vars, input_var_idx, num_of_outputs, output_var_idx, io_dictz3, args.threshold, args.K
#     #     )
#     # skf_dict_z3[Yvar[i]] = skfunc[0]

#     # Skolem function in verilog format
#     # skfunc = skf.get_skolem_function(
#     #     gcln, num_of_vars, input_var_idx, num_of_outputs, output_var_idx, io_dict, args.threshold, args.K)
#     # skf_dict_verilog[Yvar[i]] = skfunc[0]
# # For single output

    # Skolem function in verilog format
    skfuncv = skf.get_skolem_function(
        gcln, num_of_vars, input_var_idx, num_of_outputs, output_var_idx, io_dict, args.threshold, args.K)
    # skf_dict_verilog[Yvar[i]] = skfunc[0]
# >>>>>>> shared
    
    extract_t_e = time.time()
    print("Formula Extraction Time: ", extract_t_e - extract_t_s)
    # print("-----------------------------------------------------------------------------")
# <<<<<<< HEAD
# For single output
    # print("skolem function run: ", skfunc)
    # print("-----------------------------------------------------------------------------")
    # print(skf_dict_z3)
    # final_accuracy = final_accuracy / num_of_outputs
    # final_loss = train_loss[-1]
    # loss_drop = train_loss[0] - train_loss[-1]
    # if any(v=='()\n' or v == '\n' for v in skf_dict_z3.values()):
    #     t = time.time() - start_time
    #     datastring = str(args.verilog_spec)+", "+str(final_epochs)+", "+str(args.batch_size)+", "+str(args.learning_rate)+", "+str(args.K)+", "+str(len(input_var_idx))+", "+str(num_of_outputs)+", "+str(0)+", "+str(skf_dict_z3)+", "+"empty string"+", "+str(t)+", "+str(final_loss)+", "+str(loss_drop)+", "+str(final_accuracy)+"\n"
# For single output
# =======
#     print("skolem function run: ", skfuncz3)
#     # print("-----------------------------------------------------------------------------")
    print("start loss: ", train_loss[0])
    print("end loss: ", train_loss[-1])

    

# VERILOG VERSION
    # if any(v=='()\n' or v == '\n' for v in skfunc):
    #     t = time.time() - start_time
    #     datastring = str(args.verilog_spec)+", "+str(args.epochs)+", "+str(args.batch_size)+", "+str(args.learning_rate)+", "+str(args.K)+", "+str(len(input_var_idx))+", "+str(num_of_outputs)+", "+str(0)+", "+str(skfunc)+", "+"Empty String"+", "+str(t)+", "+str(final_loss)+", "+str(loss_drop)+", "+str(accuracy)+"\n"
    #     print(datastring)
    #     f = open("multi_output_results.csv", "a")
    #     f.write(datastring)
    #     f.close()
    #     exit("No Skolem Function Learned!! Try Again.")


    # Run the Z3 Validity Checker
# <<<<<<< HEAD
#     # util.store_nn_output(num_of_outputs, skfunc)
#     # preparez3(args.verilog_spec, args.verilog_spec_location, num_of_outputs)
# =======
#     # num_of_outputs = len(skf_dict_z3)
#     util.store_nn_output(len(skfuncz3), skfuncz3)
#     # preparez3(args.verilog_spec, args.verilog_spec_location, len(skfuncz3))
# >>>>>>> trial
    # importlib.reload(z3)
    # result, _ = z3.check_validity()
    # if result:
    #     print("Z3: Valid")
    # else:
    #     print("Z3: Not Valid")
    # ----------------------------------------------------------------------------------------------------------
    # print("io dict: ", io_dict)

    # ----------------------------------------------------------------------------------------------------------
# <<<<<<< HEAD
    # Skolem function in verilog format
    skfunc = skf.get_skolem_function(
        gcln, num_of_vars, input_var_idx, num_of_outputs, output_var_idx, io_dict, args.threshold, args.K)
    print("skfunc veri: ", skfunc)
    # Write the error formula in verilog
    util.write_error_formula(args.verilog_spec, verilog, verilogformula, skfunc, Xvar, Yvar, PosUnate, NegUnate)
# =======

    # Run Manthan's Validity Checker
    # sat call to errorformula:
    check, sigma, ret = util.verify(Xvar, Yvar, verilog)
    # ----------------------------------------------------------------------------------------------------------

    
    if check == 0:
        print("error...ABC network read fail")
        print("Skolem functions not generated")
        print("not solved !!")
        exit()
    
    if ret == 0:
        print('error formula unsat.. skolem functions generated')
        print("success")
# <<<<<<< HEAD
        skfunc = [sk.replace('\n', '') for sk in skfunc]
        print("==============", '; '.join(skfunc))
        t = time.time() - start_time
        datastring = str(args.verilog_spec)+", "+str(epochs)+", "+str(args.batch_size)+", "+str(args.learning_rate)+", "+str(args.K)+", "+str(len(input_var_idx))+", "+str(num_of_outputs)+", "+str(0)+", "+'; '.join(skfunc)+", "+"Valid"+", "+str(t)+", "+str(final_loss)+", "+str(loss_drop)+", "+str(accuracy)+"\n"
        print(datastring)
        f = open("multi_output_results.csv", "a")
        f.write(datastring)
        f.close()
    # os.system('rm /tmp/*.v')
# =======
#         # inputfile_name = args.verilog_spec[:-2]
#         # skolemformula = tempfile.gettempdir(
#         #     ) + '/' + inputfile_name + "_skolem.v"
#         # print("self sub: ", ref.selfsub)
#         # sub_skolem(skolemformula, Xvar, Yvar, Yvar_order, verilog_formula, ref.selfsub)
#     else:
#         counter_examples = torch.from_numpy(
#             np.concatenate(
#                 (sigma.modelx, sigma.modely)
#                 ).reshape((1, num_of_vars))
#             )
#         # print("ce shape: ", counter_examples.shape)
#         # # Counter example loop
#         # print("\nCounter Example Guided Trainig Loop\n", counter_examples)
#         # ret, ce_time2, ce_data_time = ce_train_loop(
#         #     training_samples, io_dict, io_dictz3, ret, counter_examples, num_of_vars, num_of_outputs, input_size, 
#         #     start_time, pos_unate, neg_unate, args.training_size, input_var_idx, output_var_idx, args.P, 
#         #     args.threshold, args.batch_size, args.verilog_spec, args.verilog_spec_location, Xvar, Yvar, 
#         #     verilog_formula, verilog, args.learning_rate, args.epochs, args.K, device
#         #     )

# >>>>>>> trial
    end_time = time.time()
    total_time = int(end_time - start_time)
    print("Time = ", end_time - start_time)
    print("Result:", ret==0)

