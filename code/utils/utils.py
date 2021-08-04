import numpy as np
import torch
# from func_spec import F
import func_spec

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
    
    def tconorm_vectorized(self, t, u, name):
        return 1 - self.tnorm_vectorized(1-t, 1-u, name)

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
        return self.tconorm_vectorized(t, u, name)
    
    def continuous_xor_vectorized(self, inp_vars, name):
        op1 = inp_vars[0, :]
        for i in range(inp_vars.shape[0]-1):
            op2 = inp_vars[i+1, :]
            t = self.tnorm_vectorized(1-op1, op2, name)
            u = self.tnorm_vectorized(op1, 1-op2, name)
            res = self.tconorm_vectorized(t, u, name)
            op1 = res
        return res
    
    def cov(x, rowvar=False, bias=False, ddof=None, aweights=None):
        """Estimates covariance matrix like numpy.cov"""
        # ensure at least 2D
        if x.dim() == 1:
            x = x.view(-1, 1)

        # treat each column as a data point, each row as a variable
        if rowvar and x.shape[0] != 1:
            x = x.t()

        if ddof is None:
            if bias == 0:
                ddof = 1
            else:
                ddof = 0

        w = aweights
        if w is not None:
            if not torch.is_tensor(w):
                w = torch.tensor(w, dtype=torch.float)
            w_sum = torch.sum(w)
            avg = torch.sum(x * (w/w_sum)[:,None], 0)
        else:
            avg = torch.mean(x, 0)

        # Determine the normalization
        if w is None:
            fact = x.shape[0] - ddof
        elif ddof == 0:
            fact = w_sum
        elif aweights is None:
            fact = w_sum - ddof
        else:
            fact = w_sum - ddof * torch.sum(w * w) / w_sum

        xm = x.sub(avg.expand_as(x))

        if w is None:
            X_T = xm.t()
        else:
            X_T = torch.mm(torch.diag(w), xm).t()

        c = torch.mm(X_T, xm)
        c = c / fact

        return c.squeeze()
    
    def spec1(self, inp_vars, name):
        return self.continuous_xor_vectorized(inp_vars, name)
    
    def spec2(self, inp_vars, name):
        res1 = self.continuous_xor(inp_vars[0, :], inp_vars[1, :], name)
        res2 = self.tnorm_vectorized(inp_vars[1, :], inp_vars[2, :], name)
        return self.continuous_xor(res1, res2, name)

    def spec3(self, inp_vars, name):
        res1 = self.tnorm_vectorized(inp_vars[0, :], inp_vars[1, :], name)
        res2 = self.continuous_xor(inp_vars[2, :], inp_vars[3, :], name)
        return self.tconorm_vectorized(res1, res2, name)
    
    def spec4(self, inp_vars, name):
        return self.continuous_xor_vectorized(inp_vars, name)

    def spec5(self, inp_vars, name):
        res1 = self.tconorm_vectorized(inp_vars[0, :], inp_vars[1, :], name)
        res2 = self.tnorm_vectorized(inp_vars[2, :], inp_vars[3, :], name)
        res3 = self.continuous_xor(res1, res2, name)
        return self.continuous_xor(res3, inp_vars[4, :], name)
    
    def spec6(self, inp_vars, name):
        res1 = self.tconorm_vectorized(inp_vars[0, :], inp_vars[1, :], name)
        res2 = self.tnorm_vectorized(inp_vars[2, :], inp_vars[3, :], name)
        res3 = self.tconorm_vectorized(res1, res2, name)
        return self.continuous_xor(res3, inp_vars[4, :], name)
    
    def spec7(self, inp_vars, name):
        a1 = 1 - self.continuous_xor(self.continuous_xor(inp_vars[0, :], inp_vars[5, :], name), inp_vars[7, :], name)
        a2 = 1 - self.continuous_xor(self.continuous_xor(self.continuous_xor(
            self.tnorm_vectorized(
                inp_vars[0, :], inp_vars[5, :], name), inp_vars[4, :], name), inp_vars[6, :], name), inp_vars[8, :]
                , name)
        t1 = self.tnorm_vectorized(inp_vars[4, :], inp_vars[6, :], name)
        t2 = self.tnorm_vectorized(inp_vars[0, :], inp_vars[5, :], name)
        t3 = self.continuous_xor(inp_vars[4, :], inp_vars[6, :], name)
        t4 = self.tnorm_vectorized(t2, t3, name)
        t5 = 1 - self.tnorm_vectorized(1-t1, 1-t4, name)
        a3 = 1 - self.continuous_xor(t5, inp_vars[3, :], name)
        a4 = 1 - self.continuous_xor(1 - inp_vars[9, :], inp_vars[1, :], name)
        a5 = 1 - self.continuous_xor(inp_vars[10, :], 1 - self.tnorm_vectorized(1 - inp_vars[0, :], 1 - inp_vars[12, :], name), name)
        a6 = 1 - self.continuous_xor(inp_vars[11, :], self.tnorm_vectorized(inp_vars[4, :], inp_vars[10, :], name), name)
        a7 = 1 - self.continuous_xor(inp_vars[12, :], 1 - self.tnorm_vectorized(1 - inp_vars[5, :], 1 - inp_vars[11, :], name), name)
        res1 = self.tnorm_vectorized(a1, a2, name)
        res2 = self.tnorm_vectorized(res1, a3, name)
        res3 = self.tnorm_vectorized(res2, a4, name)
        res4 = self.tnorm_vectorized(res3, a5, name)
        res5 = self.tnorm_vectorized(res4, a6, name)
        return self.tnorm_vectorized(res5, a7, name)

    # Fractional Sampling
    def fractional_sampling(self, no_of_samples, util, name, threshold, no_of_input_var):
        inp_vars = torch.from_numpy(np.random.uniform(0, 1, (no_of_input_var+1, no_of_samples)))
        res = func_spec.F(inp_vars, name, util)
        samples = inp_vars[:, res >= threshold].T
        cov = torch.stack((samples[:, 0], samples[:, -1])).T
        # print("shape: ", torch.stack((samples[:, 2], samples[:, -1])).shape)
        print("covariance: ", cov)
        if torch.all(cov) > 0:
            print("Positively Correlated")
        x = samples[:, 0]
        y = samples[:, -1]
        print(x.shape, y.shape)
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        correlation_coefficient = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        print("Correlation Coefficient: ", correlation_coefficient)
        # samples = samples[samples[:, -1] < threshold, :]
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
    def correlated_fractional_sampling(self, no_of_samples, name, threshold, no_of_input_var, spec):
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
            if spec == 1:
                res = self.spec1(train_samples.T, name)
            elif spec == 2:
                res = self.spec2(train_samples.T, name)
            elif spec == 3:
                res = self.spec3(train_samples.T, name)
            elif spec == 4:
                res = self.spec4(train_samples.T, name)
            elif spec == 5:
                res = self.spec5(train_samples.T, name)
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