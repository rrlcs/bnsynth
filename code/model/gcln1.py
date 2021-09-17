import torch
import torch.nn as nn
from run import util

# Gated CLN
class GCLN(torch.nn.Module):
    literal_pairs = []
    def __init__(self, input_size, num_of_output_var, K, device, P, p=0):
        super(GCLN, self).__init__()
        self.device = device
        self.P = P
        self.K = K
        self.input_size = input_size
        self.output_size = num_of_output_var
        self.dropout = nn.Dropout(p)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.linear = nn.Linear(1, 2)

        # Weights and Biases
        # self.G1.shape: 2 * no_input_var x K
        self.G11 = torch.nn.Parameter(
            torch.Tensor(self.input_size, K).uniform_(0, 1).to(dtype=torch.double).to(self.device)
            )
        self.G12 = torch.nn.Parameter(
            torch.Tensor(self.input_size, K).uniform_(0, 1).to(dtype=torch.double).to(self.device)
            )
        # self.G2.shape: K x 1
        self.G21 = torch.nn.Parameter(
            torch.Tensor(K, 1).uniform_(0, 1).to(dtype=torch.double).to(self.device)
            )
        self.G22 = torch.nn.Parameter(
            torch.Tensor(K, 1).uniform_(0, 1).to(dtype=torch.double).to(self.device)
            )
        # self.b1.shape: 2 * no_input_var x K
        self.b1 = torch.nn.Parameter(torch.randn((self.input_size, K)).to(self.device))
        # self.b2.shape: K x 1
        self.b2 = torch.nn.Parameter(torch.randn((K, 1)).to(self.device))

    def apply_gates(self, x, y):
        return torch.mul(x, y)
    def apply_bias(self, x, bias):
        return x + bias

    # FORWARD
    def forward(self, x):
        # x.shape and neg_x.shape: batch_size x no_input_vars
        x = x.to(self.device)
        neg_x = 1 - x

        # inputs.shape: batch_size x 2*no_input_vars x 1
        inputs = torch.cat((x, neg_x), dim=1).unsqueeze(-1)
        # print(inputs.shape)

        # gated_inputs.shape: batch_size x 2*no_input_vars x K
        gated_inputs1 = self.apply_gates(self.G11, inputs)
        gated_inputs2 = self.apply_gates(self.G12, inputs)
        # gated_inputs = self.apply_bias(gated_inputs, self.b1)
        # print(gated_inputs.shape)

        # or_res.shape: batch_size x K
        or_res1 = 1 - util.tnorm_n_inputs(1 - gated_inputs1)
        or_res1 = or_res1.unsqueeze(-1)
        or_res2 = 1 - util.tnorm_n_inputs(1 - gated_inputs2)
        or_res2 = or_res2.unsqueeze(-1)
        # print(or_res.shape)

        # gated_or_res.shape: batch_size x K
        gated_or_res1 = self.apply_gates(self.G21, or_res1)
        gated_or_res1 = torch.add(gated_or_res1, 1 - self.G21, alpha=1)
        gated_or_res2 = self.apply_gates(self.G22, or_res2)
        gated_or_res2 = torch.add(gated_or_res2, 1 - self.G22, alpha=1)
        # gated_or_res = self.apply_bias(gated_or_res, self.b2)
        # print(gated_or_res.shape)

        # out.shape: batch_size x 1
        out1 = util.tnorm_n_inputs(gated_or_res1).to(self.device)
        out2 = util.tnorm_n_inputs(gated_or_res2).to(self.device)
        # out1 = util.tnorm_n_inputs(gated_or_res[:,:self.K,:]).to(self.device)
        # out2 = util.tnorm_n_inputs(gated_or_res[:,self.K:,:]).to(self.device)
        # print(out1.shape, out2.shape)
        out = torch.cat((out1, out2), dim=-1)
        # print(out.shape)

        return out