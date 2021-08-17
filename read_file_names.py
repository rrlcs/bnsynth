import os
# This is the path where all the files are stored.
folder_path = './compareWithManthan/verilog/'
# Open one of the files,
file_names = []
for data_file in sorted(os.listdir(folder_path)):
	file_names.append(data_file)
print(file_names)