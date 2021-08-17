import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--verilog_spec_location", type=str, default="sample_examples", help="Enter file location")
args = parser.parse_args()

# This is the path where all the files are stored.
folder_path = './data_preparation_and_result_checking/'+args.verilog_spec_location+'/'
# Open one of the files,
file_names = []
for data_file in sorted(os.listdir(folder_path)):
	file_names.append(data_file)

for i in range(len(file_names)):
	if os.path.isfile(folder_path+file_names[i]):
		os.system("./scripts/regression.sh "+file_names[i])