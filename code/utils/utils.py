import copy

import numpy as np
import pandas as pd
import torch

# Utilities class


class utils():
    def __init__(self):
        self.name = "product"
        self.res = []
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

    def generateAllBinaryStrings(self, n, arr, i):
        if i == n:
            a = copy.deepcopy(arr)
            self.res.append(a)
            return
        arr[i] = 0.0
        self.generateAllBinaryStrings(n, arr, i + 1)
        arr[i] = 1.0
        self.generateAllBinaryStrings(n, arr, i + 1)

    def proc(self, dat, range):
        return torch.where(
            dat > 0.5,
            torch.rand(dat.shape)*range + (1 - range),
            torch.rand(dat.shape)*range
        )

    def add_noise(self, samples, range=0.2):
        return self.proc(samples, range)

    # Seed based sampling from truth table
    def seed_sampling(self, no_of_samples, util, py_spec, threshold, num_of_vars):
        arr = [0 for i in range(num_of_vars)]
        self.generateAllBinaryStrings(num_of_vars, arr, 0)
        XY_vars = torch.from_numpy(np.array(self.res).T)
        res = py_spec.F(XY_vars, util)
        samples = XY_vars[:, res >= threshold].T
        df = pd.DataFrame(samples.numpy())
        samples = torch.tensor(df.values)
        samples.type(torch.DoubleTensor)
        gen_new_data = torch.cat(
            [(self.add_noise(samples.T)).T for _ in range(100)])
        # gen_new_data = (gen_new_data * 10**2).round() / (10**2)
        samples = gen_new_data.type(torch.DoubleTensor)
        print("Train Data Generated: ", samples.shape, samples.dtype)
        return samples

    # Fractional Sampling
    def fractional_sampling(self, no_of_samples, util, py_spec, threshold, num_of_vars):
        first_interval = np.array([0, 0.01])
        second_interval = np.array([0.99, 1])
        total_length = np.ptp(first_interval)+np.ptp(second_interval)
        n = (num_of_vars, no_of_samples)
        np.random.seed(0)
        numbers = np.random.random(n)*total_length
        numbers += first_interval.min()
        numbers[numbers > first_interval.max()] += second_interval.min() - \
            first_interval.max()
        XY_vars = torch.from_numpy(numbers)
        res = py_spec.F(XY_vars, util)
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
        numbers[numbers > first_interval.max()] += second_interval.min() - \
            first_interval.max()

        XY_vars = torch.from_numpy(numbers)
        res = func_spec.F(XY_vars, util)
        samples = XY_vars[:num_of_vars, :]
        outs = (res > threshold).double()
        train_samples = torch.cat((samples.T, outs.reshape(-1, 1)), dim=1)
        sorted_data = torch.stack(
            sorted(train_samples, key=lambda train_samples: train_samples[-1], reverse=True))
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
        numbers[numbers > first_interval.max()] += second_interval.min() - \
            first_interval.max()
        XY_vars = torch.from_numpy(numbers)
        if num_of_vars == 1:
            data = []
            for i in range(XY_vars.shape[1]):
                if XY_vars[0, i] > threshold:
                    t1 = torch.cat([XY_vars[0, i].unsqueeze(-1),
                                   1-XY_vars[0, i].unsqueeze(-1)], dim=0)
                    data.append(t1)
                    t2 = torch.cat([XY_vars[0, i].unsqueeze(-1),
                                   XY_vars[0, i].unsqueeze(-1)], dim=0)
                    data.append(t2)
            train_samples = torch.stack(data)
            res = func_spec.F(XY_vars, util)
            outs = (res > threshold).double()
            train_samples = torch.cat(
                (train_samples[:, :num_of_vars], outs.reshape(-1, 1)), dim=1)
            print("Train Data Generated: ", train_samples.shape)
            return train_samples
        res = self.continuous_xor_vectorized(XY_vars)
        data = []
        for i in range(res.shape[0]):
            if res[i] > threshold:
                t1 = torch.cat([XY_vars[:, i], 1-res[i].unsqueeze(-1)], dim=0)
                data.append(t1)
                t2 = torch.cat([XY_vars[:, i], res[i].unsqueeze(-1)], dim=0)
                data.append(t2)
        train_samples = torch.stack(data)
        res = self.continuous_xor_vectorized(train_samples.T)
        outs = (res > threshold).double()
        train_samples = torch.cat(
            (train_samples[:, :num_of_vars], outs.reshape(-1, 1)), dim=1)
        print("Train Data Generated: ", train_samples.shape)

        return train_samples
