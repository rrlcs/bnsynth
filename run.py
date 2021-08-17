import os
import argparse

from matplotlib.pyplot import pause
from data.dataLoader import dataLoader
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from data.generateTrainData import generateTrainData
from code.algorithms import trainRegression as tr
from code.algorithms import trainClassification as tc1
from code.algorithms import trainClassification2ndForm as tc2
from code.utils.utils import utils
from code.utils import plot as pt
from code.utils import getSkolemFunc as skf
from data_preparation_and_result_checking.verilog2python import build_spec
from data_preparation_and_result_checking.verilog2z3 import preparez3
from data_preparation_and_result_checking.verilogPreprocess import verilog_preprocess

# Init utilities
util = utils()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--threshold", metavar="--th", type=float, default=0.8, help="Enter value between 0.5 <= th <= 1")
	parser.add_argument("--no_of_samples", metavar="--n", type=int, default=50000, help="Enter n >= 50000")
	parser.add_argument("--no_of_input_var", metavar="--noiv", type=int, default=1, help="Enter value >= 1")
	parser.add_argument("--run_for_all_outputs", type=int, default=1, help="0: Runs for only one output var, 1: Runs for all ouputs")
	parser.add_argument("--K", type=int, default=10, help="No. of Clauses >= 1")
	parser.add_argument("--epochs", type=int, default=50, help="No. of epochs to train")
	parser.add_argument("--learning_rate", metavar="--lr", type=float, default=0.01, help="Default 0.01")
	parser.add_argument("--batch_size", type=int, default=32, help="Enter batch size")
	parser.add_argument("--tnorm_name", type=str, default="product", help="godel/product")
	parser.add_argument("--P", type=int, default=0, help="0: Regression, 1: Classification with y as labels, 2: Classification with F out as labels")
	parser.add_argument("--train", type=int, default=0, help="1/0; 0 loads the saved model")
	parser.add_argument("--correlated_sampling", type=int, default=0, help="1/0")
	parser.add_argument("--verilog_spec", type=str, default="sample1", help="Enter file name")
	parser.add_argument("--verilog_spec_location", type=str, default="sample_examples", help="Enter file location")
	args = parser.parse_args()

	util.name = args.tnorm_name
	# training_size = min(args.no_of_samples, 50000)
	training_size = args.no_of_samples

	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	# verilog_preprocess(args.verilog_spec, args.verilog_spec_location)
	# exit()
	F, num_of_vars, num_out_vars, output_var_idx, io_dict = build_spec(args.verilog_spec, args.verilog_spec_location)
	var_indices = [i for i in range(num_of_vars)]
	input_var_idx = torch.tensor([x for x in var_indices if x not in output_var_idx])

	skolem_functions = ""
	if args.run_for_all_outputs:
		num_of_outputs = len(output_var_idx)
	else:
		num_of_outputs = 1
	for i in range(num_of_outputs):
		from code.model import gcln as gcln
		output_var_pos = output_var_idx[i]
		var_idx_except_one_out = torch.tensor([x for x in var_indices if x != output_var_pos])
		input_size = 2*len(input_var_idx)
		# exit()

		# generate training data
		training_samples = generateTrainData(args.P, util, args.no_of_samples, args.threshold, num_of_vars, input_var_idx, args.correlated_sampling)

		# load data
		train_loader = dataLoader(training_samples, training_size, args.P, input_var_idx, output_var_pos, args.threshold, args.batch_size, TensorDataset, DataLoader)

		# exit()
		'''
		Select Problem:
		0: Regression
		1: Classification 1
		2: Classification 2
		3: Classification 3
		'''

		if args.P == 0:
			if args.train:
				print("train", args.train)
				loss_fn = nn.MSELoss()
				gcln, lossess = tr.train_regressor(train_loader, loss_fn, args.learning_rate, args.epochs, input_size, args.K, device, args.P, torch, gcln.GCLN)
				torch.save(gcln.state_dict(), "regressor")
			else:
				print("no train")
				gcln = gcln.GCLN(input_size, args.K, device, args.P, p=0).to(device)
				gcln.load_state_dict(torch.load("regressor"))
				gcln.eval()
			skfunc = skf.get_skolem_function(gcln, num_of_vars, input_var_idx, output_var_pos, io_dict, args.threshold, args.K)
			skolem_functions += skfunc
		elif args.P == 1:
			if args.train:
				loss_fn = nn.BCEWithLogitsLoss()
				gcln, lossess = tc1.train_classifier(train_loader, loss_fn, args.learning_rate, args.epochs, input_size, args.K, device, args.P, torch, gcln.GCLN)
				torch.save(gcln.state_dict(), "classifier1")
			else:
				gcln = gcln.GCLN(input_size, args.K, device, args.P, p=0).to(device)
				gcln.load_state_dict(torch.load("classifier1"))
				gcln.eval()
			skfunc = skf.get_skolem_function(gcln, num_of_vars, input_var_idx, output_var_pos, io_dict, args.threshold, args.K)
			skolem_functions += skfunc
		elif args.P == 2:
			if args.train:
				loss_fn = nn.BCEWithLogitsLoss()
				gcln, lossess = tc2.train_classifier(train_loader, loss_fn, args.learning_rate, args.epochs, input_size, args.K, device, args.P, torch, gcln.GCLN, util, args.verilog_spec)
				torch.save(gcln.state_dict(), "classifier2")
			else:
				gcln = gcln.GCLN(input_size, args.K, device, args.P, p=0).to(device)
				gcln.load_state_dict(torch.load("classifier2"))
				gcln.eval()
			skfunc = skf.get_skolem_function(gcln, num_of_vars, input_var_idx, output_var_pos, io_dict, args.threshold, args.K)
			skolem_functions += skfunc

		if args.train:
			f = open("lossess", "w")
			lossess = np.array(lossess)
			lossess.tofile(f, sep=",", format="%s")

		pt.plot()

	f = open("nn_output", "w")
	f.write(skolem_functions[:-1])
	f.close()
	# Check Validity
	preparez3(args.verilog_spec, args.verilog_spec_location)
	# Run the Validity Checker
	os.system("python3 data_preparation_and_result_checking/z3ValidityChecker.py")