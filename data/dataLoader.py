def dataLoader(training_samples, training_size, P, input_var_idx, output_var_idx, num_of_outputs, threshold, batch_size, TensorDataset, DataLoader):
	# Define training data loader
	inps = training_samples[:training_size, input_var_idx]
	if num_of_outputs == 1:
		tgts = training_samples[:training_size, output_var_idx[0]]
	else:
		tgts = training_samples[:training_size, output_var_idx]
	if P == 1:
		tgts = (tgts > threshold).double()
	dataset = TensorDataset(inps, tgts)
	train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
	return train_loader