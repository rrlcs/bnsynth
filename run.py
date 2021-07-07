import os
import argparse
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
from code.model import gcln as gcln
from code.utils import plot as pt
from code.utils import getSkolemFunc as skf

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
	parser.add_argument("--train", type=int, default=0, help="1/0; 0 loads the saved model")
	parser.add_argument("--correlated_sampling", type=int, default=0, help="1/0")
	parser.add_argument("--spec", type=int, default=1, help="Enter values from 1 to 5")
	args = parser.parse_args()

	# training_size = min(args.no_of_samples, 50000)
	training_size = args.no_of_samples
	output_var_pos = args.no_of_input_var
	input_size = 2*args.no_of_input_var
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	# generate training data
	training_samples = generateTrainData(args.P, util, args.no_of_samples, args.tnorm_name, args.spec, args.threshold, args.no_of_input_var, args.correlated_sampling)

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
			cln, lossess = tr.train_regressor(train_loader, loss_fn, args.learning_rate, args.epochs, input_size, args.K, device, args.tnorm_name, args.P, torch, gcln.CLN)
			torch.save(cln.state_dict(), "regressor")
		else:
			print("no train")
			cln = gcln.CLN(input_size, args.K, device, args.tnorm_name, args.P, p=0).to(device)
			cln.load_state_dict(torch.load("regressor"))
			cln.eval()
		skf.get_skolem_function(cln, args.no_of_input_var, args.threshold, args.K)
	elif args.P == 1:
		if args.train:
			loss_fn = nn.BCEWithLogitsLoss()
			cln, lossess = tc1.train_classifier(train_loader, loss_fn, args.learning_rate, args.epochs, input_size, args.K, device, args.tnorm_name, args.P, torch, gcln.CLN)
			torch.save(cln.state_dict(), "classifier1")
		else:
			cln = gcln.CLN(input_size, args.K, device, args.tnorm_name, args.P, p=0).to(device)
			cln.load_state_dict(torch.load("classifier1"))
			cln.eval()
		skf.get_skolem_function(cln, args.no_of_input_var, args.threshold, args.K)
	elif args.P == 2:
		if args.train:
			loss_fn = nn.BCEWithLogitsLoss()
			cln, lossess = tc2.train_classifier(train_loader, loss_fn, args.learning_rate, args.epochs, input_size, args.K, device, args.tnorm_name, args.P, torch, gcln.CLN, util)
			torch.save(cln.state_dict(), "classifier2")
		else:
			cln = gcln.CLN(input_size, args.K, device, args.tnorm_name, args.P, p=0).to(device)
			cln.load_state_dict(torch.load("classifier2"))
			cln.eval()
		skf.get_skolem_function(cln, args.no_of_input_var, args.threshold, args.K)

	if args.train:
		f = open("lossess", "w")
		lossess = np.array(lossess)
		lossess.tofile(f, sep=",", format="%s")

	pt.plot()

	# Check Validity
	os.system("python3 compareWithManthan/MyApp.py --spec="+str(args.spec))