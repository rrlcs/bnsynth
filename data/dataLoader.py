from torch.utils.data import DataLoader, TensorDataset


def dataLoader(training_samples, training_size, P, input_var_idx, output_var_idx, num_of_outputs, threshold, batch_size):
	# Define training data loader
	inps = training_samples[:, input_var_idx]
	if num_of_outputs == 1:
		tgts = training_samples[:, output_var_idx]
	else:
		tgts = training_samples[:, output_var_idx]
	if P == 1:
		tgts = (tgts > threshold).double()
	print(inps.shape, tgts.shape)
	dataset = TensorDataset(inps, tgts)
	train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
	return train_loader
