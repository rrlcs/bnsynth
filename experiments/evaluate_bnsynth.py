import subprocess

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
print("Num of files: ", len(verilog_files))
for name in verilog_files:
    subprocess.run(["timeout", "5000", "./run.sh", "0",
                    name, "1", "50"])
