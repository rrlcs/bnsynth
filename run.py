import argparse
from dataLoader import dataLoader
import torch
import torch.nn as nn
import numpy as np
from imports import TensorDataset, DataLoader
from generateTrainData import generateTrainData
import trainRegression as tr
import trainClassification as tc1
import trainClassification2ndForm as tc2
from utils import utils
import gcln as gcln
import plot as pt
import getSkolemFunc as skf

# Init utilities
util = utils()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--threshold", metavar="--th", type=float, default=0.8, help="Enter value between 0.5 <= th <= 1")
	parser.add_argument("--no_of_samples", metavar="--n", type=int, default=50000, help="Enter n >= 50000")
	parser.add_argument("--no_of_input_var", metavar="--noiv", type=int, default=1, help="Enter value >= 1")
	parser.add_argument("--K", type=int, default=10, help="No. of Clauses >= 1")
	parser.add_argument("--epochs", type=int, default=50, help="No. of epochs to train")
	parser.add_argument("--learning_rate", metavar="--lr", type=float, default=0.01, help="Default 0.01")
	parser.add_argument("--batch_size", type=int, default=32, help="Enter batch size")
	parser.add_argument("--tnorm_name", type=str, default="product", help="godel/product")
	parser.add_argument("--P", type=int, default=0, help="0: Regression, 1: Classification with y as labels, 2: Classification with F out as labels")
	parser.add_argument("--train", type=int, default=0, help="True/False; False loads the saved model")
	args = parser.parse_args()

	training_size = min(args.no_of_samples, 50000)
	output_var_pos = args.no_of_input_var
	input_size = 2*args.no_of_input_var
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	# generate training data
	training_samples = generateTrainData(args.P, util, args.no_of_samples, args.tnorm_name, args.threshold, args.no_of_input_var)

	# load data
	train_loader = dataLoader(training_samples, training_size, args.P, args.no_of_input_var, output_var_pos, args.threshold, args.batch_size, TensorDataset, DataLoader)

	'''
	Select Problem:
	0: Regression
	1: Classification with y as labels
	2: Classification with output of F as labels
	'''

	if args.P == 0:
		if args.train:
			print("train", args.train)
			loss_fn = nn.MSELoss()
			cln, lossess = tr.train_regressor(train_loader, loss_fn, args.learning_rate, args.epochs, input_size, args.K, device, args.tnorm_name, torch, gcln.CLN)
			torch.save(cln.state_dict(), "regressor")
		else:
			print("no train")
			cln = gcln.CLN(input_size, args.K, device, args.tnorm_name, classify=True, p=0).to(device)
			cln.load_state_dict(torch.load("regressor"))
			cln.eval()
		skf.get_skolem_function(cln, args.no_of_input_var, args.threshold, args.K)
	elif args.P == 1:
		if args.train:
			loss_fn = nn.CrossEntropyLoss()
			cln, lossess = tc1.train_classifier(train_loader, loss_fn, args.learning_rate, args.epochs, input_size, args.K, device, args.tnorm_name, torch, gcln.CLN)
			torch.save(cln.state_dict(), "classifier1")
		else:
			cln = gcln.CLN(input_size, args.K, device, args.tnorm_name, classify=True, p=0).to(device)
			cln.load_state_dict(torch.load("classifier1"))
			cln.eval()
		skf.get_skolem_function(cln, args.no_of_input_var, args.threshold, args.K)
	elif args.P == 2:
		if args.train:
			loss_fn = nn.BCEWithLogitsLoss()
			cln, lossess = tc2.train_classifier(train_loader, loss_fn, args.learning_rate, args.epochs, input_size, args.K, device, args.tnorm_name, torch, gcln.CLN, util)
			torch.save(cln.state_dict(), "classifier2")
		else:
			cln = gcln.CLN(input_size, args.K, device, args.tnorm_name, classify=True, p=0).to(device)
			cln.load_state_dict(torch.load("classifier2"))
			cln.eval()
		skf.get_skolem_function(cln, args.no_of_input_var, args.threshold, args.K)

	if args.train:
		f = open("lossess", "w")
		lossess = np.array(lossess)
		lossess.tofile(f, sep=",", format="%s")

	pt.plot()