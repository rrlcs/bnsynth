from turtle import forward
from code.utils.utils import util

import torch
import torch.nn as nn

# CNF Network


class CNF_Netowrk(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size, device) -> None:
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.layer_or_weights = torch.nn.Parameter(
            torch.Tensor(
                self.input_size, self.hidden_size
            ).uniform_(0., 1.0).to(dtype=torch.double).to(self.device)
        )

        self.layer_and_weights = torch.nn.Parameter(
            torch.Tensor(
                self.hidden_size, self.output_size
            ).uniform_(0., 1.).to(dtype=torch.double).to(self.device)
        )

    def apply_gates(self, x, y):
        return torch.mul(x, y)

    def apply_bias(self, x, bias):
        return x + bias

    def forward(self, inputs):
        with torch.no_grad():
            self.layer_or_weights.data.clamp_(0.0, 1.0)
            self.layer_and_weights.data.clamp_(0.0, 1.0)

        # gated_inputs.shape: batch_size x 2*no_input_vars x K
        gated_inputs = self.apply_gates(self.layer_or_weights, inputs)
        o = 1 - gated_inputs

        # or_res.shape: batch_size x K
        or_res = 1 - util.tnorm_n_inputs(o)
        or_res = or_res.unsqueeze(-1)

        # gated_or_res.shape: batch_size x K
        gated_or_res = self.apply_gates(self.layer_and_weights, or_res)
        gated_or_res = torch.add(
            gated_or_res, 1.0 - self.layer_and_weights, alpha=1)

        # out.shape: batch_size x 1
        outs = util.tnorm_n_inputs(gated_or_res).unsqueeze(-1)

        return outs


class DNF_Netowrk(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size, device) -> None:
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.layer_and_weights = torch.nn.Parameter(
            torch.Tensor(
                self.input_size, self.hidden_size
            ).uniform_(0., 1.).to(dtype=torch.double).to(self.device)
        )

        self.layer_or_weights = torch.nn.Parameter(
            torch.Tensor(
                self.hidden_size, self.output_size
            ).uniform_(0., 1.).to(dtype=torch.double).to(self.device)
        )

    def apply_gates(self, x, y):
        return torch.mul(x, y)

    def apply_bias(self, x, bias):
        return x + bias

    def forward(self, inputs):
        with torch.no_grad():
            self.layer_or_weights.data.clamp_(0.0, 1.0)
            self.layer_and_weights.data.clamp_(0.0, 1.0)

        # gated_inputs.shape: batch_size x 2*no_input_vars x K

        gated_inputs = self.apply_gates(self.layer_and_weights, inputs)

        and_res = torch.add(
            gated_inputs, 1 - self.layer_and_weights, alpha=1)

        # or_res.shape: batch_size x K
        and_res = util.tnorm_n_inputs(and_res).unsqueeze(-1)

        # gated_or_res.shape: batch_size x K
        gated_and = self.apply_gates(self.layer_or_weights, and_res)
        gated_and_res = (1 - util.tnorm_n_inputs(1 - gated_and)).unsqueeze(-1)

        # out.shape: batch_size x 1
        outs = gated_and_res.squeeze(-1)

        return outs


class CNF_Netowrk3(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size, K, device) -> None:
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.K = K

        self.layer_or_weights = torch.nn.Parameter(
            torch.Tensor(
                self.input_size, self.hidden_size
            ).uniform_(0., 1.).to(dtype=torch.double).to(self.device)
        )

        self.layer_and_weights = torch.nn.Parameter(
            torch.Tensor(
                self.hidden_size, 1
            ).uniform_(0., 1.).to(dtype=torch.double).to(self.device)
        )

    def apply_gates(self, x, y):
        return torch.mul(x, y)

    def apply_bias(self, x, bias):
        return x + bias

    def forward(self, inputs):
        with torch.no_grad():
            self.layer_or_weights.data.clamp_(0.0, 1.0)
            self.layer_and_weights.data.clamp_(0.0, 1.0)

        # gated_inputs.shape: batch_size x 2*no_input_vars x K

        gated_inputs = []
        for i in range(self.output_size):
            gated_inputs.append(self.apply_gates(
                self.layer_or_weights[:, i*self.K:(i+1)*self.K], inputs))
        gated_inputs = torch.stack(gated_inputs)

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
        gated_or_res = torch.stack(gated_or_res)

        # out.shape: batch_size x 1
        outs = []
        for i in range(self.output_size):
            outs.append(util.tnorm_n_inputs(
                gated_or_res[i, :, :, :]).to(self.device))
        outs = torch.stack(outs).permute(1, 0, 2)

        return outs


class DNF_Netowrk3(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size, K, device) -> None:
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.K = K

        self.layer_or_weights = torch.nn.Parameter(
            torch.Tensor(
                self.hidden_size, 1
            ).uniform_(0., 1.).to(dtype=torch.double).to(self.device)
        )

        self.layer_and_weights = torch.nn.Parameter(
            torch.Tensor(
                self.input_size, self.hidden_size
            ).uniform_(0., 1.).to(dtype=torch.double).to(self.device)
        )

    def apply_gates(self, x, y):
        return torch.mul(x, y)

    def apply_bias(self, x, bias):
        return x + bias

    def forward(self, inputs):
        with torch.no_grad():
            self.layer_or_weights.data.clamp_(0.0, 1.0)
            self.layer_and_weights.data.clamp_(0.0, 1.0)

        # gated_inputs.shape: batch_size x 2*no_input_vars x K
        gated_inputs = []
        for i in range(self.output_size):
            gated_ = self.apply_gates(
                self.layer_and_weights[:, i*self.K:(i+1)*self.K], inputs)

            gated_inputs.append(torch.add(
                gated_, 1 - self.layer_and_weights[:, i*self.K:(i+1)*self.K], alpha=1))
        gated_inputs = torch.stack(gated_inputs)

        gated_and = []
        for i in range(self.output_size):
            gated_and.append(util.tnorm_n_inputs(
                gated_inputs[i, :, :, :]).to(self.device))
        gated_and = torch.stack(gated_and).unsqueeze(-1)  # .permute(1, 0, 2)

        gated_and_ = []
        for i in range(self.output_size):
            gated_and_.append(self.apply_gates(
                self.layer_or_weights[i*self.K:(i+1)*self.K, :], gated_and[i, :, :, :]))
        gated_and_ = torch.stack(gated_and_)

        # or_res.shape: batch_size x K
        or_res = []
        for i in range(self.output_size):
            or_res.append(
                (1 - util.tnorm_n_inputs(1 - gated_and_[i, :, :, :])))
        or_res = torch.stack(or_res).squeeze(-1).permute(1, 0)
        # gated_or_res.shape: batch_size x K

        outs = or_res
        return outs


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

        self.cnf_layer_1 = CNF_Netowrk(
            self.input_size, 1, self.K, device)

        self.cnf_layer_2 = CNF_Netowrk(
            self.output_size, 1, self.K, device)

        self.cnf_layers = nn.Sequential(self.cnf_layer_1)

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

        outs = self.cnf_layers(inputs)

        return outs.squeeze(-1)


class GCLN_CNF_Arch2(torch.nn.Module):
    literal_pairs = []

    def __init__(self, input_size, num_of_output_var, K, device):
        super(GCLN_CNF_Arch2, self).__init__()
        self.device = device
        self.K = K
        self.input_size = input_size
        self.output_size = num_of_output_var

        self.cnf_layer_1 = CNF_Netowrk(
            self.input_size, self.output_size, self.K, device)
        self.cnf_layer_2 = CNF_Netowrk(
            self.output_size, self.output_size, self.K, device)
        self.cnf_layer_3 = CNF_Netowrk(
            self.output_size, self.output_size, self.K, device)

        self.cnf_layers = nn.Sequential(self.cnf_layer_1)

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

        # Layers of CNF
        outs = self.cnf_layers(inputs)

        return outs.squeeze(-1)


class GCLN_CNF_Arch3(torch.nn.Module):
    literal_pairs = []

    def __init__(self, input_size, num_of_output_var, K, device):
        super(GCLN_CNF_Arch3, self).__init__()
        self.device = device
        self.K = K
        self.input_size = input_size
        self.output_size = num_of_output_var

        self.cnf_layer_1 = CNF_Netowrk3(
            self.input_size, self.output_size, self.K*self.output_size, self.K, device)
        self.cnf_layer_2 = CNF_Netowrk3(
            self.output_size, self.output_size, self.K*self.output_size, self.K, device)
        self.cnf_layer_3 = CNF_Netowrk3(
            self.output_size, self.output_size, self.K*self.output_size, self.K, device)

        self.cnf_layers = nn.Sequential(
            self.cnf_layer_1)

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

        # Layers of CNF
        outs = self.cnf_layers(inputs)

        return outs.squeeze(-1)

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

        # Weights and Biases
        # self.G1.shape: 2 * no_input_var x num_of_output_var * K
        self.cnf_layer_1 = DNF_Netowrk(
            self.input_size, 1, self.K, device)
        self.cnf_layer_2 = DNF_Netowrk(
            self.output_size, self.output_size, self.K, device)
        self.cnf_layers = nn.Sequential(
            self.cnf_layer_1)

    # FORWARD
    def forward(self, x):
        # x.shape and neg_x.shape: batch_size x no_input_vars
        x = x.to(self.device)
        neg_x = 1 - x

        # inputs.shape: batch_size x 2*no_input_vars x 1
        inputs = torch.cat((x, neg_x), dim=1).unsqueeze(-1)

        outs = self.cnf_layers(inputs)
        return outs


class GCLN_DNF_Arch2(torch.nn.Module):
    literal_pairs = []

    def __init__(self, input_size, num_of_output_var, K, device):
        super(GCLN_DNF_Arch2, self).__init__()
        self.device = device
        self.K = K
        self.input_size = input_size
        self.output_size = num_of_output_var

        # Weights and Biases
        self.cnf_layer_1 = DNF_Netowrk(
            self.input_size, self.output_size, self.K, device)
        self.cnf_layer_2 = DNF_Netowrk(
            self.output_size, self.output_size, self.K, device)
        self.cnf_layers = nn.Sequential(
            self.cnf_layer_1)

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

        outs = self.cnf_layers(inputs)
        return outs


class GCLN_DNF_Arch3(torch.nn.Module):
    literal_pairs = []

    def __init__(self, input_size, num_of_output_var, K, device):
        super(GCLN_DNF_Arch3, self).__init__()
        self.device = device
        self.K = K
        self.input_size = input_size
        self.output_size = num_of_output_var

        # Weights and Biases
        self.cnf_layer_1 = DNF_Netowrk3(
            self.input_size, self.output_size, self.K*self.output_size, self.K, device)
        self.cnf_layer_2 = DNF_Netowrk3(
            self.output_size, self.output_size, self.K*self.output_size, self.K, device)
        self.cnf_layers = nn.Sequential(
            self.cnf_layer_1)

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

        outs = self.cnf_layers(inputs)
        return (outs)
