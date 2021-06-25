import torch
import torch.nn as nn
from run import util

# Gated CLN
class CLN(torch.nn.Module):
    literal_pairs = []
    def __init__(self, input_size, K, device, name, P, p=0):
        super(CLN, self).__init__()
        self.device = device
        self.P = P
        self.input_size = input_size
        self.dropout = nn.Dropout(p)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.linear = nn.Linear(1, 2)

        # Weights and Biases
        # self.G1.shape: 2 * no_input_var x K
        self.G1 = torch.nn.Parameter(
            torch.Tensor(self.input_size, K).uniform_(0, 1).to(dtype=torch.double).to(self.device)
            )
        # self.G2.shape: K x 1
        self.G2 = torch.nn.Parameter(
            torch.Tensor(K, 1).uniform_(0, 1).to(dtype=torch.double).to(self.device)
            )
        # self.b1.shape: 2 * no_input_var x K
        self.b1 = torch.nn.Parameter(torch.randn((self.input_size, K)).to(self.device))
        # self.b2.shape: K x 1
        self.b2 = torch.nn.Parameter(torch.randn((K, 1)).to(self.device))
        
        self.name = name

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

        with torch.no_grad():
            self.G1.data.clamp_(0.0, 1.0)
            self.b1.data.clamp_(0.0, 1.0)
            self.G2.data.clamp_(0.0, 1.0)
            self.b2.data.clamp_(0.0, 1.0)

        # gated_inputs.shape: batch_size x 2*no_input_vars x K
        gated_inputs = self.apply_gates(self.G1, inputs)
        # gated_inputs = self.apply_bias(gated_inputs, self.b1)
        # print(gated_inputs.shape)

        # or_res.shape: batch_size x K
        or_res = 1 - util.tnorm_n_inputs(1 - gated_inputs, self.name)
        or_res = or_res.unsqueeze(-1)
        # print(or_res.shape)
        # neg_or_res = 1 - or_res

        '''# or_res_stacked.shape: batch_size x 2K
        # or_res_stacked = torch.stack((or_res, neg_or_res)).permute(1, 2, 0).reshape((x.shape[0], -1))'''

        # gated_or_res.shape: batch_size x K
        gated_or_res = self.apply_gates(self.G2, or_res)
        gated_or_res = self.apply_bias(gated_or_res, self.b2)
        # print(gated_or_res.shape)

        # out.shape: batch_size x 1
        out = util.tnorm_n_inputs(gated_or_res, self.name).to(self.device)

        return out