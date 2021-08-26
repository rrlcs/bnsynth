import matplotlib.pyplot as plt
num_of_vars = [3,4,5,13]
time = [12,15,12,15]
plt.xlabel("num of variables in verilog")
plt.ylabel("total time in seconds")
plt.scatter(num_of_vars, time)
plt.savefig("total_time.png")

# num_of_vars = [3, 4, 5, 13, 66]
# time = [0.083, 0.087, 0.093, 0.17, 62.41]
# plt.xlabel("num of variables in verilog")
# plt.ylabel("preprocess time in seconds")
# plt.scatter(num_of_vars, time)
# plt.savefig("preprocess_time.png")