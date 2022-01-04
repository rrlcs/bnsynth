import argparse
import importlib
import time
from code.ce_train import ce_train_loop
from code.train import train
from code.utils import getSkolemFunc as skf
from code.utils import getSkolemFunc4z3 as skfz3
from code.utils import plot as pt
from code.utils.utils import util
from typing import OrderedDict

import numpy as np
import torch

from benchmarks import z3ValidityChecker as z3
from benchmarks.verilog2z3 import preparez3
from data.dataLoader import dataLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", metavar="--th", type=float,
                        default=0.8, help="Enter value between 0.5 <= th <= 1")
    parser.add_argument("--no_of_samples", metavar="--n",
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
    args = parser.parse_args()

    util.name = args.tnorm_name
    training_size = args.no_of_samples

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    start_time = time.time()

    # Start Preprocessing
    pre_t_s = time.time()
    verilog, varlistfile = util.prepare_file_names(args.verilog_spec, args.verilog_spec_location)
    output_varlist = util.get_output_varlist(varlistfile)  # Y variable list
    output_varlist = ["i"+e.split("_")[1] for e in output_varlist]
    Xvar_tmp, Yvar_tmp, total_vars = util.get_temporary_variables(verilog, output_varlist)
    total_varsz3 = total_vars
    total_vars = ["i"+e.split("_")[1] for e in total_vars]
    verilog_formula = util.change_modulename(verilog)
    pos_unate, neg_unate, Xvar, Yvar, Xvar_map, Yvar_map = util.preprocess_manthan(
        varlistfile,verilog,Xvar_tmp,Yvar_tmp
        )
    pre_t_e = time.time()
    print("Preprocessing Time: ", pre_t_e - pre_t_s)

    # print("-------------", pos_unate, neg_unate, Xvar, Yvar, Xvar_map, Yvar_map, total_vars, output_varlist)

    # TO DO:
    # 1. USE THE UNATES TO CONSTRUCT UNATE_SKOLEMFORMULA
    # result = util.check_unates(pos_unate, neg_unate, Xvar, Yvar, args.verilog_spec[:-2])
    # if result:
    #     exit("All Unates!")
    
    # to sample, we need a cnf file and variable mapping coressponding to
    # varilog variables
    data_t_s = time.time()
    cnf_content, allvar_map = util.prepare_cnf_content(
        verilog, 
        Xvar, 
        Yvar, 
        Xvar_map, 
        Yvar_map, 
        pos_unate, 
        neg_unate
        )
    
    # generate sample
    samples = util.generate_samples(
        cnf_content, 
        Xvar, 
        Yvar, 
        Xvar_map, 
        Yvar_map, 
        allvar_map,
        verilog,
        max_samples=50000
        )
    
    data = np.asarray(samples)
    print(data.shape)
    np.savetxt("sample.csv", data, delimiter=',')

    # Generate training data
    training_samples = torch.from_numpy(samples)
    training_samples = training_samples.repeat(2, 1)
    training_samples = torch.cat([
        util.add_noise((training_samples)) for _ in range(20)
        ])
    training_samples = training_samples.to(torch.double)
    print(training_samples.shape)
    # Get train test split
    training_set, validation_set = util.get_train_test_split(training_samples)
    print("Total, Train, and Valid shapes", training_samples.shape,
          training_set.shape, validation_set.shape)

    data_t_e = time.time()
    print("Data Sampling Time: ", data_t_e - data_t_s)

    num_of_vars, num_out_vars, num_of_eqns = util.get_counts(Xvar, Yvar, verilog)
    
    print("No. of vars: {}, No. of output vars: {}, No. of eqns: {}".format(num_of_vars, num_out_vars, num_of_eqns))
    
    # Prepare input output dictionaries
    io_dict = {}
    for index, value in enumerate(total_vars):
        io_dict[index] = value
    io_dict = OrderedDict(io_dict)

    io_dictz3 = {}
    for index, value in enumerate(total_varsz3):
        io_dictz3[index] = value
    io_dictz3 = OrderedDict(io_dictz3)

    # Obtain variable indices
    var_indices, input_var_idx, output_var_idx = util.get_indices(num_of_vars, output_varlist, io_dict)
    input_size = 2*len(input_var_idx)

    preprocess_end_time = time.time()
    preprocess_time = preprocess_end_time - start_time
    # store_preprocess_time(args.verilog_spec, num_of_vars, num_out_vars, num_of_eqns, args.epochs, args.no_of_samples, preprocess_time)

    if args.run_for_all_outputs == 1:
        num_of_outputs = len(output_var_idx)
    else:
        num_of_outputs = 1

    # load data
    train_loader = dataLoader(training_set, training_size, args.P, input_var_idx,
                              output_var_idx, num_of_outputs, args.threshold, args.batch_size)
    validation_loader = dataLoader(validation_set, training_size, args.P, input_var_idx,
                                   output_var_idx, num_of_outputs, args.threshold, args.batch_size)
    
    # End Preprocessing


    # Initialize skolem funcition with empty string
    skolem_functions = ""

    # TRAINING MODEL
    train_t_s = time.time()
    gcln, train_loss, valid_loss = train(
        args.P, 
        args.train, 
        train_loader, 
        validation_loader, 
        args.learning_rate, 
        args.epochs, 
        input_size, 
        num_of_outputs, 
        args.K, 
        device, 
        num_of_vars, 
        input_var_idx, 
        output_var_idx, 
        io_dict, 
        io_dictz3, 
        args.threshold, 
        args.verilog_spec, 
        args.verilog_spec_location, 
        Xvar, 
        Yvar, 
        verilog_formula, 
        verilog, 
        pos_unate, 
        neg_unate
        )
    train_t_e = time.time()

    print("Training Time: ", train_t_e - train_t_s)

    extract_t_s = time.time()
    skfunc = skfz3.get_skolem_function(
        gcln, num_of_vars, input_var_idx, num_of_outputs, output_var_idx, io_dictz3, args.threshold, args.K)
    extract_t_e = time.time()
    print("Formula Extraction Time: ", extract_t_e - extract_t_s)
    util.store_losses(train_loss, valid_loss)
    pt.plot()

    print("-----------------------------------------------------------------------------")
    print("skolem function run: ", skfunc)
    print("-----------------------------------------------------------------------------")

    # Run the Z3 Validity Checker
    util.store_nn_output(num_of_outputs, skfunc)
    preparez3(args.verilog_spec, args.verilog_spec_location, num_of_outputs)
    importlib.reload(z3)
    result, _ = z3.check_validity()
    if result:
        print("Z3: Valid")
    else:
        print("Z3: Not Valid")

    # skfunc = [s.replace("_", "") for s in skfunc]
    skfunc = skf.get_skolem_function(
        gcln, num_of_vars, input_var_idx, num_of_outputs, output_var_idx, io_dict, args.threshold, args.K)
    verify_t_s = time.time()
    candidateskf = util.prepare_candidateskf(skfunc, Yvar, pos_unate, neg_unate)
    util.create_skolem_function(
        args.verilog_spec.split('.v')[0], candidateskf, Xvar, Yvar)
    error_content, refine_var_log = util.create_error_formula(
        Xvar, Yvar, verilog_formula)
    util.add_skolem_to_errorformula(error_content, [], verilog)

    # Run the Validity Checker
    # sat call to errorformula:
    check, sigma, ret = util.verify(Xvar, Yvar, verilog)
    verify_t_e = time.time()
    print("Verification Time: ", verify_t_e - verify_t_s)
    
    if check == 0:
        print("error...ABC network read fail")
        print("Skolem functions not generated")
        print("not solved !!")
        exit()
    
    if ret == 0:
        print('error formula unsat.. skolem functions generated')
        print("success")
    else:
        # print(check, sigma.modelx, sigma.modely, sigma.modelyp, ret)
        counter_examples = torch.from_numpy(
            np.concatenate(
                (sigma.modelx, sigma.modely)
                ).reshape((1, num_of_vars))
            )
        
        # print("counter examples from ABC network: ", counter_examples)
        ce_time1 = 0
        ce_time2 = 0
        ce_data_time = 0
        print("\nCounter Example Guided Trainig Loop\n", counter_examples)
        ret, ce_time2, ce_data_time = ce_train_loop(
            training_samples, 
            io_dict, 
            io_dictz3,
            ret,
            counter_examples, 
            num_of_vars, 
            num_of_outputs, 
            input_size, 
            start_time,
            pos_unate,
            neg_unate,
            training_size,
	        input_var_idx, output_var_idx, args.P, args.threshold, args.batch_size,
            args.verilog_spec, args.verilog_spec_location, Xvar, Yvar, 
            verilog_formula, verilog, args.learning_rate, args.epochs, 
            args.K, device
            )

    end_time = time.time()
    total_time = int(end_time - start_time)
    # print("Counter Example generation time: ", ce_time1+ce_time2)
    # print("Data Generation Time: ", data_time + ce_data_time)
    print("Time = ", end_time - start_time)
    print("Result:", ret==0)

    # line = args.verilog_spec+","+str(num_of_vars)+","+str(args.epochs)+","+str(args.no_of_samples)+","+result+","+str(total_time)+"\n"
    # f = open("results.csv", "a")
    # f.write(line)
    # f.close()
