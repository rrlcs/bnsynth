import importlib
import time
from code.ce_train import ce_train_loop
from code.train import train
from code.utils import getSkolemFunc as skf
from code.utils import getSkolemFunc4z3 as skfz3
from code.utils import plot as pt
from code.utils.utils import util
from xml.sax.xmlreader import InputSource

import numpy as np
import torch

from benchmarks import z3ValidityChecker as z3
from benchmarks.verilog2z3 import preparez3
from data.dataLoader import dataLoader

if __name__ == "__main__":

    # Get Argument Parser
    parser = util.make_arg_parser()
    args = parser.parse_args()

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ----------------------------------------------------------------------------------------------------------
    # Start Preprocessing
    start_time = time.time()
    pre_t_s = time.time()
    verilog, output_varlist, total_vars, total_varsz3,\
         verilog_formula, pos_unate, neg_unate, Xvar,\
              Yvar, Xvar_map, Yvar_map = util.preprocess_wrapper(
                  args.verilog_spec, args.verilog_spec_location
                  )
    pre_t_e = time.time()
    # End Preprocessing
    print("Preprocessing Time: ", pre_t_e - pre_t_s)
    # ----------------------------------------------------------------------------------------------------------


    # ----------------------------------------------------------------------------------------------------------
    # TO DO:
    # 1. USE THE UNATES TO CONSTRUCT UNATE_SKOLEMFORMULA
    # result = util.check_unates(pos_unate, neg_unate, Xvar, Yvar, args.verilog_spec[:-2])
    # if result:
    #     exit("All Unates!")
    # ----------------------------------------------------------------------------------------------------------


    # ----------------------------------------------------------------------------------------------------------
    # to sample, we need a cnf file and variable mapping coressponding to
    # varilog variables
    data_t_s = time.time()
    cnf_content, allvar_map = util.prepare_cnf_content(
        verilog, Xvar, Yvar, Xvar_map, Yvar_map, pos_unate, neg_unate
        )
    
    # generate sample
    samples = util.generate_samples(
        cnf_content, Xvar, Yvar, Xvar_map, Yvar_map, allvar_map,verilog,
        max_samples=args.training_size
        )
    
    # print("samples: ", samples.shape)

    # Repeat or add noise to get larger dataset
    
    # samples = np.array([[0., 1., 0.], [0., 0., 1.], [1., 1., 1.],
    #     [1., 0., 0.]])
    # training_samples = util.make_dataset_larger(samples)
    training_samples = torch.from_numpy(samples).to(torch.double)
    print(training_samples)

    # Get train test split
    training_set, validation_set = util.get_train_test_split(training_samples)
    print("Total, Train, and Valid shapes", training_samples.shape,
          training_set.shape, validation_set.shape)

    data_t_e = time.time()
    print("Data Sampling Time: ", data_t_e - data_t_s)

    num_of_vars, num_out_vars, num_of_eqns = util.get_var_counts(Xvar, Yvar, verilog)
    print("No. of vars: {}, No. of output vars: {}, No. of eqns: {}".format(num_of_vars, num_out_vars, num_of_eqns))
    # ----------------------------------------------------------------------------------------------------------

    
    # ----------------------------------------------------------------------------------------------------------
    # Prepare input output dictionaries
    io_dict, io_dictz3 = util.prepare_io_dicts(total_vars, total_varsz3)

    # Obtain variable indices
    var_indices, input_var_idx, output_var_idx = util.get_var_indices(num_of_vars, output_varlist, io_dict)
    input_size = 2*len(input_var_idx)
    print("Input size: ", input_size)
    print("Output size: ", len(output_var_idx))

    # store_preprocess_time(args.verilog_spec, num_of_vars, num_out_vars, num_of_eqns, 
    # args.epochs, args.no_of_samples, preprocess_time)
    
    if args.run_for_all_outputs == 1:
        num_of_outputs = len(output_var_idx)
    else:
        num_of_outputs = 1
    # ----------------------------------------------------------------------------------------------------------
    print("out size: ", num_of_outputs)

    # ----------------------------------------------------------------------------------------------------------
    # load data
    train_loader = dataLoader(training_set, args.training_size, args.P, input_var_idx,
                              output_var_idx, num_of_outputs, args.threshold, args.batch_size)
    # validation_loader = dataLoader(validation_set, args.training_size, args.P, input_var_idx,
    #                                output_var_idx, num_of_outputs, args.threshold, args.batch_size)
    validation_loader = []
    # ----------------------------------------------------------------------------------------------------------


    # Initialize skolem funcition with empty string
    skolem_functions = ""

    # ----------------------------------------------------------------------------------------------------------
    # TRAINING MODEL
    train_t_s = time.time()
    # num_of_outputs = 1
    skf_dict_z3 = {}
    skf_dict_verilog = {}
    # for i in range(len(Yvar)):
    i = 0
    current_output = i
    gcln, train_loss, valid_loss = train(
        args.P, args.train, train_loader, validation_loader, args.learning_rate, args.epochs, 
        input_size, num_of_outputs, current_output, args.K, device, num_of_vars, input_var_idx, output_var_idx, 
        io_dict, io_dictz3, args.threshold, args.verilog_spec, args.verilog_spec_location, 
        Xvar, Yvar, verilog_formula, verilog, pos_unate, neg_unate
        )
    train_t_e = time.time()
    # print("Training Time: ", train_t_e - train_t_s)
    # ----------------------------------------------------------------------------------------------------------

    util.store_losses(train_loss, valid_loss)
    pt.plot()

    # ----------------------------------------------------------------------------------------------------------
    # Checking Skolem Function using Z3
    extract_t_s = time.time()
    # Skolem function in z3py format
    skfuncz3 = skfz3.get_skolem_function(
        gcln, num_of_vars, input_var_idx, num_of_outputs, output_var_idx, io_dictz3, args.threshold, args.K
        )
    # skf_dict_z3[Yvar[i]] = skfunc[0]

    # Skolem function in verilog format
    skfuncv = skf.get_skolem_function(
        gcln, num_of_vars, input_var_idx, num_of_outputs, output_var_idx, io_dict, args.threshold, args.K)
    # skf_dict_verilog[Yvar[i]] = skfunc[0]
    
    extract_t_e = time.time()
    # print("Formula Extraction Time: ", extract_t_e - extract_t_s)
    # print("-----------------------------------------------------------------------------")
    print("skolem function run: ", skfuncz3)
    # print("-----------------------------------------------------------------------------")
    print("start loss: ", train_loss[0])
    print("end loss: ", train_loss[-1])

    print(skf_dict_z3)
    if any(v=='()\n' or v == '\n' for v in skfuncz3):
        t = time.time() - start_time
        datastring = str(args.epochs)+", "+str(args.K)+", "+str(0)+", "+"empty string"+", "+"Invalid"+", "+str(t)+"\n"
        print(datastring)
        f = open("abalation_original.csv", "a")
        f.write(datastring)
        f.close()
        exit("No Skolem Function Learned!! Try Again.")
    print("hello")
    # exit()
    # Store train and test losses in a file
    

    # Run the Z3 Validity Checker
    # num_of_outputs = len(skf_dict_z3)
    util.store_nn_output(len(skfuncz3), skfuncz3)
    preparez3(args.verilog_spec, args.verilog_spec_location, len(skfuncz3))
    importlib.reload(z3)
    result, _ = z3.check_validity()
    if result:
        print("Z3: Valid")
    else:
        print("Z3: Not Valid")
    # ----------------------------------------------------------------------------------------------------------


    # ----------------------------------------------------------------------------------------------------------


    # Write the error formula in verilog
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@skf manthan: ", skfuncv)
    util.write_error_formula(args.verilog_spec, verilog, verilog_formula, skfuncv, Xvar, Yvar, pos_unate, neg_unate)

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
        t = time.time() - start_time
        datastring = str(args.epochs)+", "+str(args.K)+", "+str(0)+", "+str(skf_dict_z3)+", "+"Valid"+", "+str(t)+"\n"
        print(datastring)
        f = open("abalation_original.csv", "a")
        f.write(datastring)
        f.close()
    else:
        counter_examples = torch.from_numpy(
            np.concatenate(
                (sigma.modelx, sigma.modely)
                ).reshape((1, num_of_vars))
            )
        # print("ce shape: ", counter_examples.shape)
        # # Counter example loop
        # print("\nCounter Example Guided Trainig Loop\n", counter_examples)
        # ret, ce_time2, ce_data_time = ce_train_loop(
        #     training_samples, io_dict, io_dictz3, ret, counter_examples, num_of_vars, num_of_outputs, input_size, 
        #     start_time, pos_unate, neg_unate, args.training_size, input_var_idx, output_var_idx, args.P, 
        #     args.threshold, args.batch_size, args.verilog_spec, args.verilog_spec_location, Xvar, Yvar, 
        #     verilog_formula, verilog, args.learning_rate, args.epochs, args.K, device
        #     )

    end_time = time.time()
    total_time = int(end_time - start_time)
    print("Time = ", end_time - start_time)
    print("Result:", ret==0)

    # line = args.verilog_spec+","+str(num_of_vars)+","+str(args.epochs)+","+str(args.no_of_samples)+","+result+","+str(total_time)+"\n"
    # f = open("results.csv", "a")
    # f.write(line)
    # f.close()
