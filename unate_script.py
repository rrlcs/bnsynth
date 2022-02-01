import os
import subprocess
from os import listdir
from os.path import isfile, join

verilog_path = "./benchmarks/verilog"
# varstoelim_path = "./verilog/Yvarlist"

verilog_files = [f for f in listdir(verilog_path) if isfile(join(verilog_path, f))]
verilog_files = [files for files in verilog_files if files.count(".v") > 0]
# print(verilog_files)
for name in verilog_files:
	# os.system('./run.sh 0 '+name+' 1 1 &> log.txt')
	subprocess.run(["./run.sh", "0", name, "1", "1", "&>", "log.txt"])
