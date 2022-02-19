import argparse
import os

import pandas as pd

'''
This is a script for running all test files at once

'''

parser = argparse.ArgumentParser()
parser.add_argument("--verilog_spec_location", type=str, default="sample_examples", help="Enter file location")
args = parser.parse_args()

# This is the path where all the files are stored.
folder_path = './data_preparation_and_result_checking/'+args.verilog_spec_location+'/'
# Open one of the files,
file_names = []
for data_file in sorted(os.listdir(folder_path)):
	file_names.append(data_file)

data = pd.read_csv('preprocess_data.csv')
files = data['Benchmark']

for i in range(len(file_names)):
	if os.path.isfile(folder_path+file_names[i]):
		f = open(folder_path+file_names[i], "r")
		content = f.read()
		f.close()
		content = content.replace(file_names[i].split(".v")[0], "formula")
		f = open(folder_path+file_names[i], "w")
		f.write(content)
		f.close()

file_left = file_names
f1 = open('./scripts/output/largefiles.txt', 'a')
for i in range(len(file_left)):
	if os.path.isfile(folder_path+file_left[i]):
		print(file_left[i], i)
		os.system("./scripts/regression.sh "+file_left[i])
		f1.write(file_left[i])
