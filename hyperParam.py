from imports import torch

# Hyperparameters
threshold = 0.8
no_of_samples = 10000000
no_of_input_var = 2
output_var_pos = no_of_input_var
K = 10
input_size = 2*no_of_input_var
epochs =50
learning_rate=0.01
batch_size = 32
training_size = 50000
types_of_oper = ['luka', 'godel', 'product']
if no_of_input_var > 3:
    name = types_of_oper[2]
else:
    name = types_of_oper[2]
device = 'cuda' if torch.cuda.is_available() else 'cpu'