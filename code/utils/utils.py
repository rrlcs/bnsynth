import numpy as np
import torch
import func_spec

# Utilities class
class utils():
    def __init__(self):
        self.name = "product"
        super(utils, self).__init__()

    def negation(self, x):
        return 1 - x

    # Continuous AND
    def tnorm_vectorized(self, t, u):
        if self.name == "luka":
            return max(0, t + u - 1)
        elif self.name == "godel":
            return torch.minimum(t, u)
        elif self.name == "product":
            return torch.multiply(t, u)
        else:
            print("Wrong Name!")
    
    def tconorm_vectorized(self, t, u):
        return 1 - self.tnorm_vectorized(1-t, 1-u)

    # Continuous AND with n inputs
    def tnorm_n_inputs(self, inp):
        if self.name == "luka":
            return torch.max(torch.tensor(0), torch.sum(inp) - 1)
        elif self.name == "godel":
            out, _ = torch.min(inp, dim=-2)
            return out
        elif self.name == "product":
            return torch.prod(inp, -2)
        else:
            print("Wrong Name!")

    # Continuous xor definition
    def continuous_xor(self, x, y):
        t = self.tnorm_vectorized(1-x, y)
        u = self.tnorm_vectorized(x, 1-y)
        return self.tconorm_vectorized(t, u)
    
    def continuous_xor_vectorized(self, XY_vars):
        op1 = XY_vars[0, :]
        for i in range(XY_vars.shape[0]-1):
            op2 = XY_vars[i+1, :]
            t = self.tnorm_vectorized(1-op1, op2)
            u = self.tnorm_vectorized(op1, 1-op2)
            res = self.tconorm_vectorized(t, u)
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

    # Fractional Sampling
    def fractional_sampling(self, no_of_samples, util, threshold, num_of_vars):
        first_interval = np.array([0, 0.1])
        second_interval = np.array([0.9, 1])
        total_length = np.ptp(first_interval)+np.ptp(second_interval)
        n = (num_of_vars, no_of_samples)
        numbers = np.random.random(n)*total_length
        numbers += first_interval.min()
        numbers[numbers > first_interval.max()] += second_interval.min()-first_interval.max()
        XY_vars = torch.from_numpy(numbers)
        res = func_spec.F(XY_vars, util)
        samples = XY_vars[:, res >= threshold].T
        print("Train Data Generated: ", samples.shape)

        return samples
    
    # Fractional Sampling
    def fractional_sampling_pos_and_neg(self, no_of_samples, util, threshold, num_of_vars):
        first_interval = np.array([0, 0.3])
        second_interval = np.array([0.7, 1])
        
        total_length = np.ptp(first_interval)+np.ptp(second_interval)
        n = (num_of_vars, no_of_samples)
        numbers = np.random.random(n)*total_length
        numbers += first_interval.min()
        numbers[numbers > first_interval.max()] += second_interval.min()-first_interval.max()

        XY_vars = torch.from_numpy(numbers)
        res = func_spec.F(XY_vars, util)
        samples = XY_vars[:num_of_vars, :]
        outs = (res > threshold).double()
        train_samples = torch.cat((samples.T, outs.reshape(-1, 1)), dim=1)
        sorted_data = torch.stack(sorted(train_samples, key=lambda train_samples: train_samples[-1], reverse=True))
        train_samples = sorted_data[:2*(outs == 1).sum(), :]
        print("Train Data Generated: ", train_samples.shape)

        return train_samples

        # Fractional Sampling
    def correlated_fractional_sampling(self, no_of_samples, util, threshold, num_of_vars):
        first_interval = np.array([0, 0.3])
        second_interval = np.array([0.7, 1])
        total_length = np.ptp(first_interval)+np.ptp(second_interval)
        n = (num_of_vars, no_of_samples)
        numbers = np.random.random(n)*total_length
        numbers += first_interval.min()
        numbers[numbers > first_interval.max()] += second_interval.min()-first_interval.max()
        XY_vars = torch.from_numpy(numbers)
        if num_of_vars == 1:
            data = []
            for i in range(XY_vars.shape[1]):
                if XY_vars[0, i] > threshold:
                    t1 = torch.cat([XY_vars[0, i].unsqueeze(-1), 1-XY_vars[0, i].unsqueeze(-1)], dim=0)
                    data.append(t1)
                    t2 = torch.cat([XY_vars[0, i].unsqueeze(-1), XY_vars[0, i].unsqueeze(-1)], dim=0)
                    data.append(t2)
            train_samples = torch.stack(data)
            res = func_spec.F(XY_vars, util)
            outs = (res > threshold).double()
            train_samples = torch.cat((train_samples[:, :num_of_vars], outs.reshape(-1, 1)), dim=1)
            print("Train Data Generated: ", train_samples.shape)
            return train_samples
        res = self.continuous_xor_vectorized(XY_vars)
        data = []
        for i in range(res.shape[0]):
            if res[i] > threshold:
                t1 = torch.cat([XY_vars[:,i], 1-res[i].unsqueeze(-1)], dim=0)
                data.append(t1)
                t2 = torch.cat([XY_vars[:,i], res[i].unsqueeze(-1)], dim=0)
                data.append(t2)
        train_samples = torch.stack(data)
        res = self.continuous_xor_vectorized(train_samples.T)
        outs = (res > threshold).double()
        train_samples = torch.cat((train_samples[:, :num_of_vars], outs.reshape(-1, 1)), dim=1)
        print("Train Data Generated: ", train_samples.shape)
        
        return train_samples