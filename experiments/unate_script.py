import pandas as pd
import os
import subprocess
from os import listdir
from os.path import isfile, join

# verilog_path = "data/benchmarks/cav20_manthan_dataset/verilog"
# # varstoelim_path = "./verilog/Yvarlist"

# verilog_files = [f for f in listdir(
#     verilog_path) if isfile(join(verilog_path, f))]
# verilog_files = [files for files in verilog_files if files.count(".v") > 0]

# print(len(verilog_files))
# data = pd.read_csv("experiments/unates.csv")
# files_done = list(data['Spec'])
# # print(list(data['Spec']))
# # print(files_done)
# # print(len(verilog_files), len(files_done))
# verilog_files = [vfile for vfile in verilog_files if vfile not in files_done]
# print(len(verilog_files))
# print(verilog_files)
verilog_files = [
    "small-dyn-partition-fixpoint-1.v",
    "small-dyn-partition-fixpoint-2.v",
    "small-dyn-partition-fixpoint-3.v",
    "small-dyn-partition-fixpoint-4.v",
    "small-dyn-partition-fixpoint-5.v",
    "small-dyn-partition-fixpoint-6.v",
    "small-dyn-partition-fixpoint-7.v"
    "small-dyn-partition-fixpoint-9.v",
    "small-dyn-partition-fixpoint-10.v"
    "small-synabs-fixpoint-3.v",
    "sdlx-fixpoint-1.v",
    "stmt41_262_275.v",
    "small-synabs-fixpoint-10.v",
    "usb-phy-fixpoint-1.v",
    "cache-coherence-3-fixpoint-1.v",
    "cache-coherence-2-fixpoint-2.v",
    "cache-coherence-3-fixpoint-2.v",
    "cache-coherence-2-fixpoint-4.v",
    "cache-coherence-3-fixpoint-3.v",
    "cache-coherence-2-fixpoint-5.v",
    "cache-coherence-2-fixpoint-6.v",
    "neclaftp4002_all_bit_differing_from_cycle.v",
    "neclaftp4001_all_bit_differing_from_cycle.v"
]
print("Num of files: ", len(verilog_files))
# for name in verilog_files:
name = "cache-coherence-2-fixpoint-6.v"
print("Current file: ", name)
# os.system('./run.sh 0 '+name+' 1 1 &> log.txt')
for _ in range(10):
    subprocess.run(["timeout", "5000", "./run.sh", "0",
                    name, "1", "10"])
