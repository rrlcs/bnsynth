import argparse
import importlib
import os
import subprocess
import time
from code.algorithms import trainClassification2ndForm as tc2
from code.algorithms.trainClassification import train_classifier
from code.algorithms.trainRegression import train_regressor
from code.model import gcln as gcln
from code.utils import getSkolemFunc as skf
from code.utils import plot as pt
from code.utils.utils import utils
from math import floor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from matplotlib.pyplot import flag

import python_specs
from data.dataLoader import dataLoader
from data.generateTrainData import generateTrainData
from data_preparation_and_result_checking import z3ValidityChecker as z3
from data_preparation_and_result_checking.preprocess import preprocess
from data_preparation_and_result_checking.verilog2python import build_spec
# from code.utils import getSkolemFunc1 as skf
from data_preparation_and_result_checking.verilog2z3 import preparez3

# Init utilities
util = utils()


def prepare_ce(io_dict, counter_examples, num_of_vars, num_of_outputs):
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
    util.res = []
    util.generateAllBinaryStrings(num_of_outputs, arr, 0)
    out_vars = np.array(util.res)
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


def generate_counter_examples(
    n, io_dict, counter_examples, py_spec, util, num_of_vars, num_of_outputs
	):
    '''
    Prepares counter examples and filters it based on 
    specification encoded in py_spec
    ce: counter examples
    '''

    # n = 5000
    ce = prepare_ce(io_dict, counter_examples, num_of_vars, num_of_outputs)
    # print("ce shape: ", ce.shape)
    res = py_spec.F(ce, util)
    ce = torch.cat(
        [
            util.add_noise((ce[:, res >= 0.5].T)) for _ in range(n)
        ]
    ).double()
    return ce


def store_nn_output(num_of_outputs, skfunc):
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


def store_losses(train_loss, valid_loss):
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


def store_preprocess_time(
	verilog_spec, num_of_vars, num_out_vars, num_of_eqns, epochs, no_of_samples, preprocess_time
	):
    line = verilog_spec+","+str(num_of_vars)+","+str(num_out_vars)+","+str(
        num_of_eqns)+","+str(epochs)+","+str(no_of_samples)+","+str(preprocess_time)+"\n"
    f = open("preprocess_data.csv", "a")
    f.write(line)
    f.close()


def load_python_spec(filename):
    mod = __import__('python_specs', fromlist=[filename])
    py_spec = getattr(mod, filename)

    return py_spec


def get_indices(num_of_vars, output_var_idx):
    var_indices = [i for i in range(num_of_vars)]
    input_var_idx = torch.tensor(
        [x for x in var_indices if x not in output_var_idx])

    return var_indices, input_var_idx


def get_train_test_split(training_samples):
    data_size = training_samples.shape[0]
    val_size = floor(data_size*0.2)
    train_size = data_size - val_size
    validation_set = training_samples[train_size:, :]
    training_set = training_samples[:train_size, :]

    return training_set, validation_set


def ce_train_loop(
    training_samples, io_dict, result, model,
    num_of_vars, num_of_outputs, input_size,
	start_time
	):
    loop = 0
    ce_time = 0
    ce_data_time = 0
    n = 1000
    while result == False and loop < 50:
        loop += 1
        print("Counter Example Loop: ", loop)
        s = time.time()
        ce = generate_counter_examples(
            n, io_dict, model, py_spec,
            util, num_of_vars, num_of_outputs
        )
        e = time.time()
        data_t = e - s
        ce_data_time += data_t
		# Add counter examples to existing training data
        training_samples = torch.cat((training_samples, ce))

		# Re-Train only on counter-examples
        # training_samples = ce

        data_size = training_samples.shape[0]
        val_size = floor(data_size*0.2)
        train_size = data_size - val_size
        validation_set = training_samples[train_size:, :]
        training_set = training_samples[:train_size, :]

        print("Training Data: ", training_set.shape, validation_set.shape)

        train_loader = dataLoader(
            training_set, training_size, args.P,
            input_var_idx, output_var_idx,
            num_of_outputs, args.threshold, args.batch_size
        )
        validation_loader = dataLoader(
            validation_set, training_size, args.P,
            input_var_idx, output_var_idx,
            num_of_outputs, args.threshold, args.batch_size
        )

        flag = 0
        # args.epochs += 5
        if args.P == 0:
            gcln, train_loss, valid_loss = train_regressor(
                train_loader,
                validation_loader,
                args.learning_rate,
                args.epochs,
                input_size,
                num_of_outputs,
                args.K,
                device,
                args.P,
				flag, num_of_vars, input_var_idx,
                output_var_idx, io_dict, args.threshold,
                args.verilog_spec, args.verilog_spec_location
            )
        elif args.P == 1:
            gcln, train_loss, valid_loss = train_classifier(
                train_loader,
                validation_loader,
                loss_fn,
                args.learning_rate,
                args.epochs, input_size,
                num_of_outputs, args.K,
                device,
                args.P,
				flag
            )
        
        # Extract and Check
        skfunc = skf.get_skolem_function(
            gcln, num_of_vars,
            input_var_idx, num_of_outputs, output_var_idx, io_dict,
            args.threshold, args.K
        )
        
        store_nn_output(num_of_outputs, skfunc)
        s = time.time()
        preparez3(
            args.verilog_spec, args.verilog_spec_location, num_of_outputs
        )
        e = time.time()
        print("Formula Extraction Time: ", e-s)


        store_losses(train_loss, valid_loss)
        pt.plot()
        # Run the Validity Checker
        importlib.reload(z3)  # Reload the package
        s = time.time()
        result, model = z3.check_validity()
        e = time.time()
        ce_time += e-s
        print("Time Elapsed = ", e - start_time)
        n += 1000

    return result, ce_time, ce_data_time


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
    num_of_vars, num_out_vars, output_var_idx, io_dict, num_of_eqns, filename = preprocess(
        args.verilog_spec, args.verilog_spec_location)
    build_spec(args.verilog_spec, args.verilog_spec_location)
    py_spec = load_python_spec(filename)
    var_indices, input_var_idx = get_indices(num_of_vars, output_var_idx)
    input_size = 2*len(input_var_idx)

    preprocess_end_time = time.time()
    preprocess_time = preprocess_end_time - start_time
    # store_preprocess_time(args.verilog_spec, num_of_vars, num_out_vars, num_of_eqns, args.epochs, args.no_of_samples, preprocess_time)
    # End Preprocessing

    # Initialize skolem funcition with empty string
    skolem_functions = ""

    if args.run_for_all_outputs == 1:
        num_of_outputs = len(output_var_idx)
    else:
        num_of_outputs = 1

    # Generate training data
    s_d = time.time()
    training_samples = generateTrainData(
        args.P, util, py_spec, args.no_of_samples, args.threshold, num_of_vars, input_var_idx, args.correlated_sampling)
    e_d = time.time()
    data_time = e_d - s_d

    # Get train test split
    training_set, validation_set = get_train_test_split(training_samples)
    print("Train, Test, and Valid shapes", training_samples.shape,
          training_set.shape, validation_set.shape)

    # load data
    train_loader = dataLoader(training_set, training_size, args.P, input_var_idx,
                              output_var_idx, num_of_outputs, args.threshold, args.batch_size)
    validation_loader = dataLoader(validation_set, training_size, args.P, input_var_idx,
                                   output_var_idx, num_of_outputs, args.threshold, args.batch_size)

    '''
	Select Problem:
	0: Regression
	1: Classification 1
	2: Classification 2
	3: Classification 3
	'''

    if args.P == 0:
        if args.train:
            gcln, train_loss, valid_loss = train_regressor(
                train_loader, validation_loader, args.learning_rate, args.epochs, input_size, num_of_outputs, args.K, device, args.P, 0, num_of_vars, input_var_idx,
                output_var_idx, io_dict, args.threshold,
                args.verilog_spec, args.verilog_spec_location)
        else:
            print("no train")
            gcln = gcln.GCLN(input_size, len(output_var_idx),
                             args.K, device, args.P, p=0).to(device)
            gcln.load_state_dict(torch.load("regressor_multi_output"))
            gcln.eval()
            print(list(gcln.G1))
    elif args.P == 1:
        if args.train:
            loss_fn = nn.BCEWithLogitsLoss()
            gcln, train_loss, valid_loss = train_classifier(
                train_loader, validation_loader, loss_fn, args.learning_rate, args.epochs, input_size, num_of_outputs, args.K, device, args.P)
            torch.save(gcln.state_dict(), "classifier1")
        else:
            gcln = gcln.GCLN(input_size, args.K, device,
                             args.P, p=0).to(device)
            gcln.load_state_dict(torch.load("classifier1"))
            gcln.eval()
    # elif args.P == 2:
    # 	if args.train:
    # 		loss_fn = nn.BCEWithLogitsLoss()
    # 		gcln, lossess = tc2.train_classifier(train_loader, loss_fn, args.learning_rate, args.epochs, input_size, args.K, device, args.P, torch, gcln.GCLN, util, py_spec)
    # 		torch.save(gcln.state_dict(), "classifier2")
    # 	else:
    # 		gcln = gcln.GCLN(input_size, args.K, device, args.P, p=0).to(device)
    # 		gcln.load_state_dict(torch.load("classifier2"))
    # 		gcln.eval()
    # 	skfunc = skf.get_skolem_function(gcln, num_of_vars, input_var_idx, output_var_idx, io_dict, args.threshold, args.K)
    # 	skolem_functions += skfunc

    skfunc = skf.get_skolem_function(
        gcln, num_of_vars, input_var_idx, num_of_outputs, output_var_idx, io_dict, args.threshold, args.K)
    store_nn_output(num_of_outputs, skfunc)
    store_losses(train_loss, valid_loss)
    pt.plot()
    preparez3(args.verilog_spec, args.verilog_spec_location, num_of_outputs)

    print("-----------------------------------------------------------------------------")
    print("skolem function: ", skfunc[0][:-1])
    print("-----------------------------------------------------------------------------")

    # Run the Validity Checker
    importlib.reload(z3)
    s = time.time()
    result, counter_examples = z3.check_validity()
    e = time.time()
    ce_time1 = e - s
    ce_time2 = 0
    ce_data_time = 0
    if not result:
        print("\nCounter Example Guided Trainig Loop\n", counter_examples)
        result, ce_time2, ce_data_time = ce_train_loop(training_samples, io_dict, result,
                               counter_examples, num_of_vars, num_of_outputs, input_size, start_time)

    end_time = time.time()
    total_time = int(end_time - start_time)
    print("Counter Example generation time: ", ce_time1+ce_time2)
    print("Data Generation Time: ", data_time + ce_data_time)
    print("Time = ", end_time - start_time)
    print("Result:", result)

    # line = args.verilog_spec+","+str(num_of_vars)+","+str(args.epochs)+","+str(args.no_of_samples)+","+result+","+str(total_time)+"\n"
    # f = open("results.csv", "a")
    # f.write(line)
    # f.close()
