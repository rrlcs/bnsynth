from imports import np, torch, reduce, op

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
            return np.minimum(t, u)
        elif name == "product":
            return np.multiply(t, u)
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
        t = self.tnorm(1-x, y, name)
        u = self.tnorm(x, 1-y, name)
        return 1 - self.tnorm(1-t, 1-u, name)
    
    def continuous_xor_vectorized(self, inp_vars, name):
        op1 = inp_vars[0, :]
        print(inp_vars.shape[0])
        for i in range(inp_vars.shape[0]-1):
            op2 = inp_vars[i+1, :]
            t = self.tnorm_vectorized(1-op1, op2, name)
            u = self.tnorm_vectorized(op1, 1-op2, name)
            res = 1-self.tnorm_vectorized(1-t, 1-u, name)
            op1 = res
        return res

    # Calculate combinations
    def ncr(self, n, r):
        r = min(r, n-r)
        numer = reduce(op.mul, range(n, n-r, -1), 1)
        denom = reduce(op.mul, range(1, r+1), 1)
        return numer // denom

    # Fractional Sampling
    def fractional_sampling(self, no_of_samples, name, threshold, no_of_input_var):
        inp_vars = np.random.uniform(0, 1, (no_of_input_var+1, no_of_samples))
        # res = 1-self.tnorm_vectorized(1-inp_vars[0, :], 1-inp_vars[1, :], name)
        # res = self.continuous_xor(res, inp_vars[2, :], name)
        # res = self.continuous_xor(res, inp_vars[3, :], name)
        res = self.continuous_xor_vectorized(inp_vars, name)
        # res = self.tnorm_vectorized(inp_vars[0, :], inp_vars[1, :], name)
        
        samples =[inp_vars[i, res >= threshold] for i in range(no_of_input_var+1)]
        outs = res[[res[i] >= threshold for i in range(len(res))]]
        
        return np.array(samples), np.array(outs)

# Initialize utilities
util = utils()