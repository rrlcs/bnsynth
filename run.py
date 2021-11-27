from math import floor
import os
import argparse

from matplotlib.pyplot import flag
import python_specs
import time
import subprocess
from data.dataLoader import dataLoader
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from data.generateTrainData import generateTrainData
from code.algorithms import trainRegression as tr
from code.algorithms import trainClassification as tc1
from code.algorithms import trainClassification2ndForm as tc2
from code.utils.utils import utils
from code.utils import plot as pt
from code.utils import getSkolemFunc as skf
# from code.utils import getSkolemFunc1 as skf
from data_preparation_and_result_checking.verilog2z3 import preparez3
from data_preparation_and_result_checking.verilog2python import build_spec
from data_preparation_and_result_checking.preprocess import preprocess
from data_preparation_and_result_checking.z3ValidityChecker import check_validity

# Init utilities
util = utils()

def generate_counter_examples(n, io_dict, model, py_spec, util):
	ce1 = []
	ce2 = []
	for i in range(len(io_dict)):
		if io_dict[i] in model:
			ce1.append(model[io_dict[i]])
			ce2.append(model[io_dict[i]])
		else:
			ce1.append([0 for _ in range(1)])
			ce2.append([1 for _ in range(1)])
	ce = [ce1, ce2]
	print("------------ce------------: ", ce)
	ce = torch.from_numpy(np.array(ce)).squeeze(-1)
	res = py_spec.F(ce.T, util)
	ce = (ce.T[:, res >= 0.5]).T
	print("ce shape: ", ce.shape, ce)
	ce = torch.cat([(util.add_noise(ce)) for _ in range(5000)])
	ce = ce.double()
	print("ce shape: ", ce.shape)
	return ce

def store_nn_output(num_of_outputs, skfunc):
	open('nn_output', 'w').close()
	f = open("nn_output", "a")
	print(num_of_outputs)
	for i in range(num_of_outputs):
		if i < num_of_outputs-1:
			f.write(skfunc[i][:-1]+"\n")
		else:
			f.write(skfunc[i][:-1])
	f.close()

def store_losses(train_loss, valid_loss):
	if args.train:
		f = open("train_loss", "w")
		train_loss = np.array(train_loss)
		train_loss.tofile(f, sep=",", format="%s")
		f.close()
		f = open("valid_loss", "w")
		valid_loss = np.array(valid_loss)
		valid_loss.tofile(f, sep=",", format="%s")
		f.close()

def ce_train_loop(training_samples, io_dict, result, model, gcln, saved_model):
	loop = 0
	data = training_samples
	print("model", model)
	while result == 'Not Valid' and loop < 50:
		loop += 1
		print(loop)
		ce = generate_counter_examples(args.no_of_samples, io_dict, model, py_spec, util)
		training_samples = torch.cat((data, ce))
		# data = ce

		data_size = training_samples.shape[0]
		val_size = floor(data_size*0.2)
		train_size = data_size - val_size
		validation_set = training_samples[train_size:, :]
		training_set = training_samples[:train_size, :]
		print("Training Data: ", data.shape, training_set.shape, validation_set.shape)
		train_loader = dataLoader(training_set, training_size, args.P, input_var_idx, output_var_idx, num_of_outputs, args.threshold, args.batch_size, TensorDataset, DataLoader)
		validation_loader = dataLoader(validation_set, training_size, args.P, input_var_idx, output_var_idx, num_of_outputs, args.threshold, args.batch_size, TensorDataset, DataLoader)
		# checkpoint = torch.load(saved_model)
		# gcln.load_state_dict(checkpoint)
		flag = 0
		args.epochs += 5
		gcln, train_loss, valid_loss = tr.train_regressor(train_loader, validation_loader, loss_fn, args.learning_rate, args.epochs, input_size, num_of_outputs, output_var_idx, args.K, device, args.P, flag, checkpoint=None)
		# torch.save(gcln.state_dict(), saved_model)
		skfunc = skf.get_skolem_function(gcln, num_of_vars, input_var_idx, num_of_outputs, output_var_idx, io_dict, args.threshold, args.K)
		store_nn_output(num_of_outputs, skfunc)
		store_losses(train_loss, valid_loss)
		pt.plot()
		preparez3(args.verilog_spec, args.verilog_spec_location, num_of_outputs)
		# Run the Validity Checker
		result, model = check_validity()

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
	parser.add_argument("--verilog_spec_location", type=str, default="verilog", help="Enter file location")
	args = parser.parse_args()

	util.name = args.tnorm_name
	# training_size = min(args.no_of_samples, 50000)
	training_size = args.no_of_samples

	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	device = 'cpu'
	start_time = time.time()

	num_of_vars, num_out_vars, output_var_idx, io_dict, num_of_eqns, filename = preprocess(args.verilog_spec, args.verilog_spec_location)
	build_spec(args.verilog_spec, args.verilog_spec_location)
	# exit()	
	# print("io dict: ", io_dict)
	mod = __import__('python_specs', fromlist=[filename])
	py_spec = getattr(mod, filename)
	var_indices = [i for i in range(num_of_vars)]
	input_var_idx = torch.tensor([x for x in var_indices if x not in output_var_idx])
	preprocess_end_time = time.time()
	preprocess_time = preprocess_end_time - start_time
	line = args.verilog_spec+","+str(num_of_vars)+","+str(num_out_vars)+","+str(num_of_eqns)+","+str(args.epochs)+","+str(args.no_of_samples)+","+str(preprocess_time)+"\n"
	f = open("preprocess_data.csv", "a")
	f.write(line)
	f.close()

	skolem_functions = ""
	if args.run_for_all_outputs == 1:
		num_of_outputs = len(output_var_idx)
	else:
		num_of_outputs = 1

	# generate training data
	training_samples = generateTrainData(args.P, util, py_spec, args.no_of_samples, args.threshold, num_of_vars, input_var_idx, args.correlated_sampling)
	from code.model import gcln as gcln
	input_size = 2*len(input_var_idx)
	
	data_size = training_samples.shape[0]
	val_size = floor(data_size*0.2)
	train_size = data_size - val_size
	validation_set = training_samples[train_size:, :]
	training_set = training_samples[:train_size, :]
	print("------shape----------", training_samples.shape, training_set.shape, validation_set.shape)
	# load data
	train_loader = dataLoader(training_set, training_size, args.P, input_var_idx, output_var_idx, num_of_outputs, args.threshold, args.batch_size, TensorDataset, DataLoader)
	validation_loader = dataLoader(validation_set, training_size, args.P, input_var_idx, output_var_idx, num_of_outputs, args.threshold, args.batch_size, TensorDataset, DataLoader)
	# print(train_loader.data.shape, validation_loader.data.shape)

	'''
	Select Problem:
	0: Regression
	1: Classification 1
	2: Classification 2
	3: Classification 3
	'''

	if args.P == 0:
		if args.train:
			# print("train", args.train)
			loss_fn = nn.MSELoss()
			flag = 1
			gcln, train_loss, valid_loss = tr.train_regressor(train_loader, validation_loader, loss_fn, args.learning_rate, args.epochs, input_size, num_of_outputs, output_var_idx, args.K, device, args.P, flag, checkpoint=None)
			torch.save(gcln.state_dict(), "regressor")
			# print(list(gcln.G1))
			# print(list(gcln.G2))
		else:
			print("no train")
			gcln = gcln.GCLN(input_size, len(output_var_idx), args.K, device, args.P, p=0).to(device)
			gcln.load_state_dict(torch.load("regressor_multi_output"))
			gcln.eval()
			print(list(gcln.G1))

		skfunc = skf.get_skolem_function(gcln, num_of_vars, input_var_idx, num_of_outputs, output_var_idx, io_dict, args.threshold, args.K)
		store_nn_output(num_of_outputs, skfunc)
		store_losses(train_loss, valid_loss)
		pt.plot()
		preparez3(args.verilog_spec, args.verilog_spec_location, num_of_outputs)
		# Run the Validity Checker
		result, model = check_validity()
		print("\nCounter Example Guided Trainig Loop\n")
		# saved_model = "regressor"
		# ce_train_loop(training_samples, io_dict, result, model, gcln, saved_model)
	elif args.P == 1:
		if args.train:
			loss_fn = nn.BCEWithLogitsLoss()
			gcln, lossess = tc1.train_classifier(train_loader, loss_fn, args.learning_rate, args.epochs, input_size, num_of_outputs, args.K, device, args.P, torch, gcln.GCLN)
			torch.save(gcln.state_dict(), "classifier1")
		else:
			gcln = gcln.GCLN(input_size, args.K, device, args.P, p=0).to(device)
			gcln.load_state_dict(torch.load("classifier1"))
			gcln.eval()
		skfunc = skf.get_skolem_function(gcln, num_of_vars, input_var_idx, num_of_outputs, output_var_idx, io_dict, args.threshold, args.K)
		store_nn_output(num_of_outputs, skfunc)
		preparez3(args.verilog_spec, args.verilog_spec_location, num_of_outputs)
		# Run the Validity Checker
		result, model = check_validity()
		print("\nCounter Example Guided Trainig Loop\n")
		# saved_model = "classifier1"
		# ce_train_loop(training_samples, io_dict, result, model, gcln, saved_model)
	elif args.P == 2:
		if args.train:
			loss_fn = nn.BCEWithLogitsLoss()
			gcln, lossess = tc2.train_classifier(train_loader, loss_fn, args.learning_rate, args.epochs, input_size, args.K, device, args.P, torch, gcln.GCLN, util, py_spec)
			torch.save(gcln.state_dict(), "classifier2")
		else:
			gcln = gcln.GCLN(input_size, args.K, device, args.P, p=0).to(device)
			gcln.load_state_dict(torch.load("classifier2"))
			gcln.eval()
		skfunc = skf.get_skolem_function(gcln, num_of_vars, input_var_idx, output_var_idx, io_dict, args.threshold, args.K)
		skolem_functions += skfunc

	end_time = time.time()
	total_time = int(end_time - start_time)
	print("Time = ", end_time - start_time)
	print("Result:", result)
	
	# line = args.verilog_spec+","+str(num_of_vars)+","+str(args.epochs)+","+str(args.no_of_samples)+","+result+","+str(total_time)+"\n"
	# f = open("results.csv", "a")
	# f.write(line)
	# f.close()
