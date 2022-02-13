import os
import subprocess

epochs = [1, 5, 10, 50, 100]
clauses = [1, 2, 3, 4, 5, 10, 50]
max_iter = 10
for e in epochs:
	for c in clauses:
		for _ in range(max_iter):
			print("here")
			cmd = "python3 run.py \
				--th=0.5 \
				--P=0 \
				--train=1 \
				--tnorm_name=product \
				--learning_rate=0.001 \
				--run_for_all_outputs=1 \
				--verilog_spec_location=sample_examples \
				--verilog_spec=ex5.v \
				--training_size=10000 \
				--epochs="+str(e)+" \
				--batch_size=32 \
				--K="+str(c)
			os.system(cmd)
		f = open("abalation_original.csv", "a")
		f.write("-,-,-,-,-"+"\n")
		f.close()
			# os.system("./run.sh 0 ex4.v "+str(e)+" "+str(c)+" &> log.txt")
