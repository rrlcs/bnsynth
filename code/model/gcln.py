from code.utils.utils import util

import torch
import torch.nn as nn

# Gated CLN

# # GCLN in CNF form
# Architecture 1


class GCLN_CNF_Arch1(torch.nn.Module):
    literal_pairs = []

    def __init__(self, input_size, num_of_output_var, K, device):
        super(GCLN_CNF_Arch1, self).__init__()
        self.device = device
        self.K = K
        self.input_size = input_size
        self.output_size = num_of_output_var
        print("gcln no out: ", num_of_output_var)

        # Weights and Biases
        # self.G1.shape: 2 * no_input_var x num_of_output_var * K
        self.layer_or_weights = torch.nn.Parameter(
            torch.Tensor(
                self.input_size, K
            ).uniform_(0., 1.).to(dtype=torch.double).to(self.device)
        )
        # self.layer_or_weights.data = torch.tensor([[1.0], [1.0]])
        # self.G2.shape: num_of_output_var * K x 1
        self.layer_and_weights = torch.nn.Parameter(
            torch.Tensor(
                K, 1
            ).uniform_(0., 1.).to(dtype=torch.double).to(self.device)
        )
        # self.layer_and_weights.data = torch.tensor([[1.0]])
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
            self.layer_or_weights.data.clamp_(0.0, 1.0)
            self.layer_and_weights.data.clamp_(0.0, 1.0)
        # x.shape and neg_x.shape: batch_size x no_input_vars
        x = x.to(self.device)
        neg_x = 1 - x

        # inputs.shape: batch_size x 2*no_input_vars x 1
        inputs = torch.cat((x, neg_x), dim=1).unsqueeze(-1)

        # gated_inputs.shape: batch_size x 2*no_input_vars x K
        gated_inputs = self.apply_gates(self.layer_or_weights, inputs)
        # gated_inputs = torch.stack(gated_inputs)
        # gated_inputs = self.apply_bias(gated_inputs, self.b1)

        # or_res.shape: batch_size x K
        or_res = (1 - util.tnorm_n_inputs(1 - gated_inputs)).unsqueeze(-1)
        # or_res = torch.stack(or_res)

        # gated_or_res.shape: batch_size x K
        gated_or = self.apply_gates(self.layer_and_weights, or_res)
        gated_or_res = torch.add(gated_or, 1 - self.layer_and_weights, alpha=1)
        # gated_or_res.append(gated_or)
        # gated_or_res = torch.stack(gated_or_res)
        # gated_or_res = self.apply_bias(gated_or_res, self.b2)

        # out.shape: batch_size x 1
        outs = util.tnorm_n_inputs(gated_or_res).to(self.device)
        print(outs.shape)
        return outs


class GCLN_CNF_Arch2(torch.nn.Module):
    literal_pairs = []

    def __init__(self, input_size, num_of_output_var, K, device):
        super(GCLN_CNF_Arch2, self).__init__()
        self.device = device
        self.K = K
        self.input_size = input_size
        self.output_size = num_of_output_var
        print("gcln no out: ", num_of_output_var)

        # Weights and Biases
        # self.G1.shape: 2 * no_input_var x num_of_output_var * K
        self.layer_or_weights = torch.nn.Parameter(
            torch.Tensor(
                self.input_size, K
            ).uniform_(0.4, 0.6).to(dtype=torch.double).to(self.device)
        )
        # self.layer_or_weights.data = torch.tensor([[1.0], [1.0]])
        # self.G2.shape: num_of_output_var * K x 1
        self.layer_and_weights = torch.nn.Parameter(
            torch.Tensor(
                K, num_of_output_var
            ).uniform_(0.5, 0.6).to(dtype=torch.double).to(self.device)
        )
        # self.layer_and_weights.data = torch.tensor([[1.0]])

    def apply_gates(self, x, y):
        return torch.mul(x, y)

    def apply_bias(self, x, bias):
        return x + bias

    # FORWARD
    def forward(self, x):
        with torch.no_grad():
            self.layer_or_weights.data.clamp_(0.0, 1.0)
            self.layer_and_weights.data.clamp_(0.0, 1.0)
        # x.shape and neg_x.shape: batch_size x no_input_vars
        x = x.to(self.device)
        neg_x = 1 - x

        # inputs.shape: batch_size x 2*no_input_vars x 1
        inputs = torch.cat((x, neg_x), dim=1).unsqueeze(-1)

        # gated_inputs.shape: batch_size x 2*no_input_vars x K
        gated_inputs = self.apply_gates(self.layer_or_weights, inputs)
        # gated_inputs = self.apply_bias(gated_inputs, self.b1)

        # or_res.shape: batch_size x K
        or_res = 1 - util.tnorm_n_inputs(1 - gated_inputs)
        or_res = or_res.unsqueeze(-1)

        # gated_or_res.shape: batch_size x K
        gated_or_res = self.apply_gates(self.layer_and_weights, or_res)
        gated_or_res = torch.add(
            gated_or_res, 1.0 - self.layer_and_weights, alpha=1)
        # gated_or_res = self.apply_bias(gated_or_res, self.b2)

        # out.shape: batch_size x 1
        outs = util.tnorm_n_inputs(gated_or_res).to(self.device)
        return outs


class GCLN_CNF_Arch3(torch.nn.Module):
    literal_pairs = []

    def __init__(self, input_size, num_of_output_var, K, device):
        super(GCLN_CNF_Arch3, self).__init__()
        self.device = device
        self.K = K
        self.input_size = input_size
        self.output_size = num_of_output_var
        print("gcln no out: ", num_of_output_var)

        # Weights and Biases
        # self.G1.shape: 2 * no_input_var x num_of_output_var * K
        self.layer_or_weights = torch.nn.Parameter(
            torch.Tensor(
                self.input_size, num_of_output_var * K
            ).uniform_(0.4, 0.6).to(dtype=torch.double).to(self.device)
        )
        # self.layer_or_weights.data = torch.tensor([[1.0], [1.0]])
        # self.G2.shape: num_of_output_var * K x 1
        self.layer_and_weights = torch.nn.Parameter(
            torch.Tensor(
                num_of_output_var * K, 1
            ).uniform_(0.5, 0.6).to(dtype=torch.double).to(self.device)
        )
        # self.layer_and_weights.data = torch.tensor([[1.0]])

    def apply_gates(self, x, y):
        return torch.mul(x, y)

    def apply_bias(self, x, bias):
        return x + bias

    # FORWARD
    def forward(self, x):
        with torch.no_grad():
            self.layer_or_weights.data.clamp_(0.0, 1.0)
            self.layer_and_weights.data.clamp_(0.0, 1.0)
        # x.shape and neg_x.shape: batch_size x no_input_vars
        x = x.to(self.device)
        neg_x = 1 - x

        # inputs.shape: batch_size x 2*no_input_vars x 1
        inputs = torch.cat((x, neg_x), dim=1).unsqueeze(-1)

        # gated_inputs.shape: batch_size x 2*no_input_vars x K
        gated_inputs = []
        for i in range(0, self.output_size):
            gated_inputs.append(self.apply_gates(
                self.layer_or_weights[:, i*self.K:(i+1)*self.K], inputs))
        gated_inputs = torch.stack(gated_inputs)
        # gated_inputs = self.apply_bias(gated_inputs, self.b1)

        # or_res.shape: batch_size x K
        or_res = []
        for i in range(self.output_size):
            or_res.append(
                (1 - util.tnorm_n_inputs(1 - gated_inputs[i, :, :, :])).unsqueeze(-1))
        or_res = torch.stack(or_res)

        # gated_or_res.shape: batch_size x K
        gated_or_res = []
        for i in range(self.output_size):
            gated_or = self.apply_gates(
                self.layer_and_weights[i*self.K:(i+1)*self.K, :], or_res[i, :, :, :])
            gated_or_res.append(torch.add(
                gated_or, 1 - self.layer_and_weights[i*self.K:(i+1)*self.K, :], alpha=1))
            # gated_or_res.append(gated_or)
        gated_or_res = torch.stack(gated_or_res)
        # gated_or_res = self.apply_bias(gated_or_res, self.b2)

        # out.shape: batch_size x 1
        outs = []
        for i in range(self.output_size):
            outs.append(util.tnorm_n_inputs(
                gated_or_res[i, :, :, :]).to(self.device))
        return torch.stack(outs)

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# GCLN in DNF form
# Architecture 1


class GCLN_DNF_Arch1(torch.nn.Module):
    literal_pairs = []

    def __init__(self, input_size, num_of_output_var, K, device):
        super(GCLN_DNF_Arch1, self).__init__()
        self.device = device
        self.K = K
        self.input_size = input_size
        self.output_size = num_of_output_var
        print("gcln no out: ", num_of_output_var)

        # Weights and Biases
        # self.G1.shape: 2 * no_input_var x num_of_output_var * K
        self.layer_or_weights = torch.nn.Parameter(
            torch.Tensor(
                self.input_size, K
            ).uniform_(0.4, 0.6).to(dtype=torch.double).to(self.device)
        )
        # self.layer_or_weights.data = torch.tensor([[1.0], [1.0]])
        # self.G2.shape: num_of_output_var * K x 1
        self.layer_and_weights = torch.nn.Parameter(
            torch.Tensor(
                K, 1
            ).uniform_(0.5, 0.6).to(dtype=torch.double).to(self.device)
        )
        # self.layer_and_weights.data = torch.tensor([[1.0]])
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
            self.layer_or_weights.data.clamp_(0.0, 1.0)
            self.layer_and_weights.data.clamp_(0.0, 1.0)
        # x.shape and neg_x.shape: batch_size x no_input_vars
        x = x.to(self.device)
        neg_x = 1 - x

        # inputs.shape: batch_size x 2*no_input_vars x 1
        inputs = torch.cat((x, neg_x), dim=1).unsqueeze(-1)

        # gated_inputs.shape: batch_size x 2*no_input_vars x K
        gated_inputs = self.apply_gates(self.layer_or_weights, inputs)
        # gated_inputs = torch.stack(gated_inputs)
        # gated_inputs = self.apply_bias(gated_inputs, self.b1)

        # or_res.shape: batch_size x K
        or_res = (1 - util.tnorm_n_inputs(1 - gated_inputs)).unsqueeze(-1)
        # or_res = torch.stack(or_res)

        # gated_or_res.shape: batch_size x K
        gated_or = self.apply_gates(self.layer_and_weights, or_res)
        gated_or_res = torch.add(gated_or, 1 - self.layer_and_weights, alpha=1)
        # gated_or_res.append(gated_or)
        # gated_or_res = torch.stack(gated_or_res)
        # gated_or_res = self.apply_bias(gated_or_res, self.b2)

        # out.shape: batch_size x 1
        outs = util.tnorm_n_inputs(gated_or_res).to(self.device)
        print(outs.shape)
        return outs


class GCLN_DNF_Arch2(torch.nn.Module):
    literal_pairs = []

    def __init__(self, input_size, num_of_output_var, K, device):
        super(GCLN_DNF_Arch2, self).__init__()
        self.device = device
        self.K = K
        self.input_size = input_size
        self.output_size = num_of_output_var
        print("gcln no out: ", num_of_output_var)

        # Weights and Biases
        # self.G1.shape: 2 * no_input_var x num_of_output_var * K
        self.layer_or_weights = torch.nn.Parameter(
            torch.Tensor(
                self.input_size, K
            ).uniform_(0.4, 0.6).to(dtype=torch.double).to(self.device)
        )
        # self.layer_or_weights.data = torch.tensor([[1.0], [1.0]])
        # self.G2.shape: num_of_output_var * K x 1
        self.layer_and_weights = torch.nn.Parameter(
            torch.Tensor(
                K, num_of_output_var
            ).uniform_(0.5, 0.6).to(dtype=torch.double).to(self.device)
        )
        # self.layer_and_weights.data = torch.tensor([[1.0]])

    def apply_gates(self, x, y):
        return torch.mul(x, y)

    def apply_bias(self, x, bias):
        return x + bias

    # FORWARD
    def forward(self, x):
        with torch.no_grad():
            self.layer_or_weights.data.clamp_(0.0, 1.0)
            self.layer_and_weights.data.clamp_(0.0, 1.0)
        # x.shape and neg_x.shape: batch_size x no_input_vars
        x = x.to(self.device)
        neg_x = 1 - x

        # inputs.shape: batch_size x 2*no_input_vars x 1
        inputs = torch.cat((x, neg_x), dim=1).unsqueeze(-1)

        # gated_inputs.shape: batch_size x 2*no_input_vars x K
        gated_inputs = self.apply_gates(self.layer_or_weights, inputs)
        # gated_inputs = self.apply_bias(gated_inputs, self.b1)

        # or_res.shape: batch_size x K
        or_res = 1 - util.tnorm_n_inputs(1 - gated_inputs)
        or_res = or_res.unsqueeze(-1)

        # gated_or_res.shape: batch_size x K
        gated_or_res = self.apply_gates(self.layer_and_weights, or_res)
        gated_or_res = torch.add(
            gated_or_res, 1.0 - self.layer_and_weights, alpha=1)
        # gated_or_res = self.apply_bias(gated_or_res, self.b2)

        # out.shape: batch_size x 1
        outs = util.tnorm_n_inputs(gated_or_res).to(self.device)
        return outs


class GCLN_DNF_Arch3(torch.nn.Module):
    literal_pairs = []

    def __init__(self, input_size, num_of_output_var, K, device):
        super(GCLN_DNF_Arch3, self).__init__()
        self.device = device
        self.K = K
        self.input_size = input_size
        self.output_size = num_of_output_var
        print("gcln no out: ", num_of_output_var)

        # Weights and Biases
        # self.G1.shape: 2 * no_input_var x num_of_output_var * K
        self.layer_or_weights = torch.nn.Parameter(
            torch.Tensor(
                self.input_size, num_of_output_var * K
            ).uniform_(0.4, 0.6).to(dtype=torch.double).to(self.device)
        )
        # self.layer_or_weights.data = torch.tensor([[1.0], [1.0]])
        # self.G2.shape: num_of_output_var * K x 1
        self.layer_and_weights = torch.nn.Parameter(
            torch.Tensor(
                num_of_output_var * K, 1
            ).uniform_(0.5, 0.6).to(dtype=torch.double).to(self.device)
        )
        # self.layer_and_weights.data = torch.tensor([[1.0]])

    def apply_gates(self, x, y):
        return torch.mul(x, y)

    def apply_bias(self, x, bias):
        return x + bias

    # FORWARD
    def forward(self, x):
        with torch.no_grad():
            self.layer_or_weights.data.clamp_(0.0, 1.0)
            self.layer_and_weights.data.clamp_(0.0, 1.0)
        # x.shape and neg_x.shape: batch_size x no_input_vars
        x = x.to(self.device)
        neg_x = 1 - x

        # inputs.shape: batch_size x 2*no_input_vars x 1
        inputs = torch.cat((x, neg_x), dim=1).unsqueeze(-1)

        # gated_inputs.shape: batch_size x 2*no_input_vars x K
        gated_inputs = []
        for i in range(0, self.output_size):
            gated_inputs.append(self.apply_gates(
                self.layer_or_weights[:, i*self.K:(i+1)*self.K], inputs))
        gated_inputs = torch.stack(gated_inputs)
        # gated_inputs = self.apply_bias(gated_inputs, self.b1)

        # or_res.shape: batch_size x K
        or_res = []
        for i in range(self.output_size):
            or_res.append(
                (1 - util.tnorm_n_inputs(1 - gated_inputs[i, :, :, :])).unsqueeze(-1))
        or_res = torch.stack(or_res)

        # gated_or_res.shape: batch_size x K
        gated_or_res = []
        for i in range(self.output_size):
            gated_or = self.apply_gates(
                self.layer_and_weights[i*self.K:(i+1)*self.K, :], or_res[i, :, :, :])
            gated_or_res.append(torch.add(
                gated_or, 1 - self.layer_and_weights[i*self.K:(i+1)*self.K, :], alpha=1))
            # gated_or_res.append(gated_or)
        gated_or_res = torch.stack(gated_or_res)
        # gated_or_res = self.apply_bias(gated_or_res, self.b2)

        # out.shape: batch_size x 1
        outs = []
        for i in range(self.output_size):
            outs.append(util.tnorm_n_inputs(
                gated_or_res[i, :, :, :]).to(self.device))
        return torch.stack(outs)
