import os
import subprocess
from sys import stderr
import pandas as pd
from os import listdir
from os.path import isfile, join

verilog_files = [
    "ex1.v", "ex2.v",
    "and_2_1.v",
    "or_2_1_1.v",
    "xor_2_1.v",
    "or_2_1_2.v",
    "or_2_1_3.v",
    "xnor_2_1.v",
    "xor_3_2.v",
    "xor_and_mix_3_1.v",
    "or_and_xor_4_1.v",
    "xor_4_1.v",
    "xor_and_or_5_3.v",
    "xor_or_and_or_5_3.v",
    "adder_13_9.v",
    "mixed_7_1.v",
    "xor_5_1.v",
    "xor_6_1.v",
    "xor_16_10.v",
    "xor_32_26",
    "xnor_8_2.v",
    "xor_implies_8_1.v",
    "xnor_implies_8_1.v"
]

verilog_path = "data/benchmarks/final_custom_benchmarks/verilog"
# varstoelim_path = "./verilog/Yvarlist"

verilog_files = [f for f in listdir(
    verilog_path) if isfile(join(verilog_path, f))]
verilog_files = [files for files in verilog_files if files.count(".v") > 0]
# verilog_files = ['xor_2_4.v', 'misc4_2_3.v', 'xor_5_3.v', 'xor_6_10.v', 'multiplexer_1_1.v', 'xor_1_3.v', 'xor_4_2.v', 'xnor_7_1.v', 'misc2_3_1.v', 'misc1_2_1.v', 'xor_2_2.v', 'xor_8_8.v', 'xor_3_1.v', 'xor_5_1.v', 'xor_3_2.v', 'xor_2_3.v', 'xor_8_1.v', 'xor_4_1.v', 'mirror_20_20.v', 'misc5_6_1.v', 'xor_5_2.v',
#                  'misc3_2_3.v', 'xor_1_2.v', 'xnor_6_2.v', 'xor_3_3.v', 'xor_6_26.v', 'xor-implies_8_8.v', 'xor_7_1.v', 'xor_6_2.v', 'mirror_10_10.v', 'multiplexer_3_3.v', 'xor_6_1.v', 'mirror_5_5.v', 'xor-implies_7_1.v', 'xor_1_4.v', 'adder_4_9.v', 'xor_4_3.v', 'multiplexer_2_2.v', 'xor-implies_1_1.v', 'xor_1_5.v', 'xnor-implies_7_1.v']
# print((verilog_files))

# exit()
data = pd.read_csv("experiments/results_with_reg_cnf.csv")
files_done = list(data['Spec'])
verilog_files = [vfile for vfile in verilog_files if vfile not in files_done]

print("Num of files: ", len(verilog_files), verilog_files)
exit()


choices = [(1, 60), (5, 120), (20, 120), (50, 180), (500, 300), (1000, 600)]
rev_choices = (choices[::-1])
# print(rev_choices)
# exit()

for name in verilog_files:
    for i in range(len(choices)):
        print("choice: ", i)
        max_time = choices[i][1]
        K = choices[i][0]
        cmd = "timeout "+str(max_time)+" ./run.sh 0 " + name+" 1 "+str(K)
        try:
            output = subprocess.check_output(
                cmd, stderr=subprocess.STDOUT, shell=True,
                universal_newlines=True)
        except subprocess.CalledProcessError as exc:
            print("Status : FAIL", exc.returncode, exc.output)
            continue
        else:
            print("Output: \n{}\n".format(output))
        # p = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
        # out, err = p.communicate()
        # p.wait()
        f = open("experiments/check", "r")
        check = f.read()
        print("OK")
        if check == "OK":
            break

# subprocess.run(["timeout", "400", "./run.sh", "0",
#                 name, "1", "50"])
