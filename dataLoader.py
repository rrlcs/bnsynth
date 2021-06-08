from imports import torch, TensorDataset, DataLoader
from generateTrainData import training_samples
from hyperParam import training_size, no_of_input_var, output_var_pos, threshold, batch_size

# Define training data loader
inps = torch.from_numpy(training_samples[:training_size, :no_of_input_var])
tgts = torch.from_numpy(training_samples[:training_size, output_var_pos])
tgts = (tgts > threshold).double()
dataset = TensorDataset(inps, tgts)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)