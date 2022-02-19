import os
import subprocess
from os import listdir
from os.path import isfile, join

verilog_path = "data/benchmarks/cav20_manthan_dataset/verilog"
# varstoelim_path = "./verilog/Yvarlist"

verilog_files = [f for f in listdir(verilog_path) if isfile(join(verilog_path, f))]
verilog_files = [files for files in verilog_files if files.count(".v") > 0]
import pandas as pd

data = pd.read_csv("experiments/unates.csv")
files_done = list(data['Spec'])
# print(list(data['Spec']))
# print(files_done)
# print(len(verilog_files), len(files_done))
verilog_files = [vfile for vfile in verilog_files if vfile not in files_done]
# print(len(verilog_files))
# print(verilog_files)
for name in verilog_files:
	print("Current file: ", name)
	# os.system('./run.sh 0 '+name+' 1 1 &> log.txt')
	subprocess.run(["timeout", "500", "./run.sh", "0", name, "1", "1", "&>", "log.txt"])
