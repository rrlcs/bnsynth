import matplotlib.pyplot as plt
# num_of_vars = [3,4,5,13]
# time = [12,15,12,15]
# plt.xlabel("num of variables in verilog")
# plt.ylabel("total time in seconds")
# plt.scatter(num_of_vars, time)
# plt.savefig("total_time.png")

num_of_vars = [3, 4, 5, 13, 66, 73, 156, 178, 179, 228, 268, 304, 359, 391, 452, 464, 600, 601, 639, 721, 722, 723, 934, 1447]
# time = [0.083, 0.087, 0.093, 0.17, 62.41, 28.27, 96.23]
time = [0.083, 0.087, 0.093, 0.17, 11.93, 81.76, 28.27, 29.47, 30.2, 51.12, 601.95, 137.16, 83.75, 96.23, 338.61, 157.80, 592.47, 195.68, 145.68, 270.14, 275.04, 287.36, 571.98, 1042.67]
plt.xlabel("num of variables in verilog")
plt.ylabel("preprocess time in seconds")
plt.scatter(num_of_vars, time)
plt.plot(num_of_vars, time)
plt.savefig("scripts/output/preprocess_time.png")