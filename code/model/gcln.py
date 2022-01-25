import torch
import torch.nn as nn
from run import util
from torch._C import dtype


# Gated CLN
class GCLN(torch.nn.Module):
    literal_pairs = []

    def __init__(self, input_size, num_of_output_var, K, device, P):
        super(GCLN, self).__init__()
        self.device = device
        self.P = P
        self.K = K
        self.input_size = input_size
        self.output_size = num_of_output_var

        # Weights and Biases
        # self.G1.shape: 2 * no_input_var x K
        self.G1 = torch.nn.Parameter(
            torch.Tensor(
                self.input_size, K
            ).uniform_(0.4, 0.6).to(dtype=torch.double).to(self.device)
        )
        # with torch.no_grad():
        # self.G1.data = torch.tensor([[1.0], [1.0]])
        # self.G2.shape: K x 1
        self.G2 = torch.nn.Parameter(
            torch.Tensor(
                K, num_of_output_var
            ).uniform_(0.4, 0.6).to(dtype=torch.double).to(self.device)
        )
        # self.G2.data = torch.tensor([[1.0]])
        # print(self.G1, self.G2)
        # self.b1.shape: 2 * no_input_var x K
        self.b1 = torch.nn.Parameter(torch.randn(
            (self.input_size, K)).to(self.device))
        # self.b2.shape: K x 1
        self.b2 = torch.nn.Parameter(torch.randn((K, 1)).to(self.device))

        self.l1 = nn.Linear(self.input_size, self.input_size)

    def apply_gates(self, x, y):
        return torch.mul(x, y)

    def apply_bias(self, x, bias):
        return x + bias

    # FORWARD
    def forward(self, x):
        
        with torch.no_grad():
            self.G1.data.clamp_(0.0, 1.0)
            self.G2.data.clamp_(0.0, 1.0)

        # x.shape and neg_x.shape: batch_size x no_input_vars
        x = x.to(self.device)
        neg_x = 1 - x

        # inputs.shape: batch_size x 2*no_input_vars x 1
        inputs = torch.cat((x, neg_x), dim=1).unsqueeze(-1)

        # gated_inputs.shape: batch_size x 2*no_input_vars x K
        gated_inputs = self.apply_gates(self.G1, inputs)
        # print(gated_inputs.shape, inputs.shape)
        # gated_inputs = self.apply_bias(gated_inputs, self.b1)

        # or_res.shape: batch_size x K
        or_res = 1 - util.tnorm_n_inputs(1 - gated_inputs)
        or_res = or_res.unsqueeze(-1)

        # gated_or_res.shape: batch_size x K
        gated_or_res = self.apply_gates(self.G2, or_res)
        gated_or_res = torch.add(gated_or_res, 1 - self.G2, alpha=1)
        # gated_or_res = self.apply_bias(gated_or_res, self.b2)

        # out.shape: batch_size x 1
        out = util.tnorm_n_inputs(gated_or_res).to(self.device)

        return out
