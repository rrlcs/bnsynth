def dataLoader(training_samples, training_size, P, input_var_idx, output_var_pos, threshold, batch_size, TensorDataset, DataLoader):
	# Define training data loader
	inps = training_samples[:training_size, input_var_idx]
	print("inpshape: ", inps.shape)
	tgts = training_samples[:training_size, output_var_pos]
	if P == 1:
		tgts = (tgts > threshold).double()
	dataset = TensorDataset(inps, tgts)
	train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
	return train_loader