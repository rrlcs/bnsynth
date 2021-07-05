import numpy as np
import torch

# Utilities class
class utils():
    def __init__(self):
        super(utils, self).__init__()

    # Continuous AND
    def tnorm(self, t, u, name):
        if name == "luka":
            return max(0, t + u - 1)
        elif name == "godel":
            return min(t, u)
        elif name == "product":
            return t * u
        else:
            print("Wrong Name!")

    def tnorm_vectorized(self, t, u, name):
        if name == "luka":
            return max(0, t + u - 1)
        elif name == "godel":
            return torch.minimum(t, u)
        elif name == "product":
            return torch.multiply(t, u)
        else:
            print("Wrong Name!")

    # Continuous AND with n inputs
    def tnorm_n_inputs(self, inp, name):
        if name == "luka":
            return torch.max(torch.tensor(0), torch.sum(inp) - 1)
        elif name == "godel":
            # print(inp)
            out, inds = torch.min(inp, dim=-2)
            return out
        elif name == "product":
            # print("inp: ", inp.shape)
            return torch.prod(inp, -2)
        else:
            print("Wrong Name!")

    # Continuous xor definition
    def continuous_xor(self, x, y, name):
        t = self.tnorm_vectorized(1-x, y, name)
        u = self.tnorm_vectorized(x, 1-y, name)
        return 1 - self.tnorm_vectorized(1-t, 1-u, name)
    
    def continuous_xor_vectorized(self, inp_vars, name):
        op1 = inp_vars[0, :]
        for i in range(inp_vars.shape[0]-1):
            op2 = inp_vars[i+1, :]
            t = self.tnorm_vectorized(1-op1, op2, name)
            u = self.tnorm_vectorized(op1, 1-op2, name)
            res = 1-self.tnorm_vectorized(1-t, 1-u, name)
            op1 = res
        return res
    
    def spec1(self, inp_vars, name):
        return self.continuous_xor_vectorized(inp_vars, name)
    
    def spec2(self, inp_vars, name):
        res1 = self.continuous_xor(inp_vars[0, :], inp_vars[1, :], name)
        res2 = self.tnorm_vectorized(inp_vars[1, :], inp_vars[2, :], name)
        return self.continuous_xor(res1, res2, name)

    def spec3(self, inp_vars, name):
        res1 = self.tnorm_vectorized(inp_vars[0, :], inp_vars[1, :], name)
        res2 = self.continuous_xor(inp_vars[2, :], inp_vars[3, :], name)
        return 1-self.tnorm_vectorized(1-res1, 1-res2, name)
    
    def spec4(self, inp_vars, name):
        return self.continuous_xor_vectorized(inp_vars, name)

    def spec5(self, inp_vars, name):
        res1 = 1 - self.tnorm_vectorized(1-inp_vars[0, :], 1-inp_vars[1, :], name)
        res2 = self.tnorm_vectorized(inp_vars[2, :], inp_vars[3, :], name)
        res3 = self.continuous_xor(res1, res2, name)
        return self.continuous_xor(res3, inp_vars[4, :], name)

    # Fractional Sampling
    def fractional_sampling(self, no_of_samples, name, threshold, no_of_input_var, spec):
        inp_vars = torch.from_numpy(np.random.uniform(0, 1, (no_of_input_var+1, no_of_samples)))
        if spec == 1:
            res = self.spec1(inp_vars, name)
        elif spec == 2:
            res = self.spec2(inp_vars, name)
        elif spec == 3:
            res = self.spec3(inp_vars, name)
        elif spec == 4:
            res = self.spec4(inp_vars, name)
        elif spec == 5:
            res = self.spec5(inp_vars, name)
        
        samples = inp_vars[:, res >= threshold].T
        print("Train Data Generated: ", samples.shape)

        return samples
    
    # Fractional Sampling
    def fractional_sampling_pos_and_neg(self, no_of_samples, name, threshold, no_of_input_var, spec):
        inp_vars = torch.from_numpy(np.random.uniform(0, 1, (no_of_input_var+1, no_of_samples)))
        if spec == 1:
            res = self.spec1(inp_vars, name)
        elif spec == 2:
            res = self.spec2(inp_vars, name)
        elif spec == 3:
            res = self.spec3(inp_vars, name)
        elif spec == 4:
            res = self.spec4(inp_vars, name)
        elif spec == 5:
            res = self.spec5(inp_vars, name)
        samples = inp_vars[:no_of_input_var, :]
        outs = (res > threshold).double()
        train_samples = torch.cat((samples.T, outs.reshape(-1, 1)), dim=1)
        sorted_data = torch.stack(sorted(train_samples, key=lambda train_samples: train_samples[-1], reverse=True))
        train_samples = sorted_data[:2*(outs == 1).sum(), :]
        print("Train Data Generated: ", train_samples.shape)

        return train_samples

        # Fractional Sampling
    def correlated_fractional_sampling(self, no_of_samples, name, threshold, no_of_input_var):
        inp_vars = torch.from_numpy(np.random.uniform(0, 1, (no_of_input_var, no_of_samples)))
        if no_of_input_var == 1:
            data = []
            for i in range(inp_vars.shape[1]):
                if inp_vars[0, i] > threshold:
                    t1 = torch.cat([inp_vars[0, i].unsqueeze(-1), 1-inp_vars[0, i].unsqueeze(-1)], dim=0)
                    data.append(t1)
                    t2 = torch.cat([inp_vars[0, i].unsqueeze(-1), inp_vars[0, i].unsqueeze(-1)], dim=0)
                    data.append(t2)
            train_samples = torch.stack(data)
            res = self.continuous_xor_vectorized(train_samples.T, name)
            outs = (res > threshold).double()
            # print(outs)
            train_samples = torch.cat((train_samples[:, :no_of_input_var], outs.reshape(-1, 1)), dim=1)
            print("Corr Train Data Generated: ", train_samples.shape)
            return train_samples
        res = self.continuous_xor_vectorized(inp_vars, name)
        # print("res", res.shape)
        data = []
        for i in range(res.shape[0]):
            if res[i] > threshold:
                t1 = torch.cat([inp_vars[:,i], 1-res[i].unsqueeze(-1)], dim=0)
                data.append(t1)
                t2 = torch.cat([inp_vars[:,i], res[i].unsqueeze(-1)], dim=0)
                data.append(t2)
        train_samples = torch.stack(data)
        print(train_samples.shape)
        res = self.continuous_xor_vectorized(train_samples.T, name)
        outs = (res > threshold).double()
        train_samples = torch.cat((train_samples[:, :no_of_input_var], outs.reshape(-1, 1)), dim=1)
        print("Corr Train Data Generated: ", train_samples.shape)
        return train_samples