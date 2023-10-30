import numpy as np
import torch

class GraphFilter(torch.nn.Module):
    def __init__(self,
                 n_features_in,
                 n_features_out,
                 n_filter_taps,
                 n_edge_features,
                 bias):
        super().__init__()
        self.N = None
        self.S = None
        self.F_in = n_features_in
        self.F_out = n_features_out
        self.K = n_filter_taps
        self.E = n_edge_features
        self.H = torch.nn.Parameter(torch.zeros((self.F_out, self.E, self.K, self.F_in)))
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros((self.F_out, 1)))
        else:
            self.register_parameter('bias', None)
    
    def add_GSO(self, S: np.ndarray):
        assert len(S.shape) == 5
        assert S.shape[2] == self.E
        self.N = self.S.shape[3]
        assert self.N == self.S.shape[4]
        self.S = torch.tensor(S)
        self.B = self.S.shape[0]
        self.T = self.S.shape[1]
    
    def forward(self, x: torch.Tensor):
        assert len(x.shape) == 4
        assert x.shape[0] == self.B # Check that batch sizes match
        assert x.shape[1] == self.T # Check that number of timesteps match
        assert self.F_in == x.shape[2]
        assert self.N == x.shape[3]

        x = x.reshape(self.B, self.T, 1, self.F_in, self.N).repeat(1, 1, self.E, 1, 1)
        z = x.reshape(self.B, self.T, 1, self.E, self.F_in, self.N)
        for k in range(self.K):
            x, _ = torch.split(x, [self.T - 1, 1], dim=1)
            zeroRow = torch.zeros((self.B, 1, self.E, self.F_in, self.N))
            x = torch.cat((zeroRow, x), dim=1)
            x = torch.matmul(x, self.S) # Recursing the shift operator
            xS = x.reshape(self.B, self.T, 1, self.E, self.F_in, self.N)
            z = torch.cat((z, xS), dim=2)
        # By the end of this, z becomes a B x T x K x E x F_in x N size Tensor
        # The filter coefficients H will be applied along the K axis

        z = z.permute(0, 1, 5, 3, 2, 4)
        z = z.reshape(self.B, self.T, self.N, self.E * self.K * self.F_in)
        H = self.H.reshape(self.F_out, self.E * self.K * self.F_in)
        H = H.permute(1, 0)
        y = torch.matmul(z, H)
        y = y.permute(0, 1, 3, 2) # Transform the output into a B x T x F_out x N size Tensor
        if self.bias:
            y = y + self.bias
        return y

class AggregateGNN(torch.nn.Module):
    def __init__(self,
                 n_signals,
                 n_filter_taps,
                 bias,
                 nonlinearity,
                 n_readout,
                 n_edge_features,
                 n_timesteps=64):
        super().__init__()
        self.L = len(n_filter_taps)
        self.F = n_signals
        self.K = n_filter_taps
        self.E = n_edge_features
        self.bias = bias
        self.sigma = nonlinearity
        self.n_readout = n_readout
        self.n_timesteps = n_timesteps

        gfl = []
        for l in range(self.L):
            gfl.append(GraphFilter(self.F[l], self.F[l+1], self.K[l], self.E, self.bias)) # append graph filter
            gfl.append(self.sigma())
        self.GFL = torch.nn.Sequential(*gfl)

        fc = []
        if len(self.n_readout) > 0:
            fc.append(torch.nn.Linear(self.F[-1], self.n_readout[0], bias=self.bias))
            for l in range(len(self.n_readout) - 1):
                fc.append(self.sigma())
                fc.append(torch.nn.Linear(self.n_readout[l], self.n_readout[l+1], bias=self.bias))
        self.readout = torch.nn.Sequential(*fc)

    def split_forward(self, x, S):
        assert len(S.shape) == 4 or len(S.shape) == 5
        if len(S.shape) == 4:
            S = S.unsqueeze(2)
        
        for l in range(self.L):
            self.GFL[2*l].addGSO(S)
        z = self.GFL(x)
        y = z.permute(0, 1, 3, 2)
        y = self.readout(y).permute(0, 1, 3, 2)
        return y, z
    
    def forward(self, x, S):
        output, _ = self.split_forward(x, S)
        return output