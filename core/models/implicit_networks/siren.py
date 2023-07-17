import torch
import numpy as np
from torch import nn
from .positional_encoding import PE
from ...utils.functions import BinaryQuantize

class SirenNetwork(nn.Module):
    def __init__(self, in_features, hidden_features, layers, out_features, cfg, w0=30, w1=30, binarize=True,
                 outermost_linear=False, exp_freq_map=False, use_pe=True, device="cpu", *args, **kwargs):
        super().__init__()
        
        self.net = []
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.layers = layers
        self.out_features = out_features
        self.w0 = w0
        self.w1 = w1
        self.exp_freq_map = exp_freq_map
        self.use_pe = use_pe
        self.device = device
        self.cfg = cfg
        self.act = BinaryQuantize().apply if binarize else nn.Identity()
        self.N_freq_by_dim = self.cfg.PE.FREQ
        
        if self.use_pe:
            self.net.append(PE(device=self.device, cfg=self.cfg))
            self.net.append(SineLayer(in_features=2*sum(self.N_freq_by_dim), out_features=self.hidden_features, device=self.device, bias=True,
                                          is_first=True, w0=self.w0, name="Sine 0"))
        else:
            self.net.append(SineLayer(in_features=self.in_features,  out_features=self.hidden_features, device=self.device, bias=True,
                                          is_first=True, w0=self.w0, name="Sine 0"))

        for i in range(self.layers):
            self.net.append(SineLayer(in_features=self.hidden_features,  out_features=self.hidden_features, device=self.device, bias=True,
                                          is_first=False, w0=self.w1, name=f"Sine {i}"))

        if outermost_linear:
            final_linear = nn.Linear(
                in_features=self.hidden_features,  out_features=self.out_features, device=self.device)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / self.hidden_features) / self.w1,
                                             np.sqrt(6 / self.hidden_features) / self.w1)

            self.net.append(final_linear)
            # self.net.append(nn.ReLU())
        else:
            self.net.append(SineLayer(in_features=self.hidden_features,  out_features=self.out_features, bias=True, device=self.device,
                                          name="Final Sine", is_first=False, w0=self.w1))

        self.net.append(nn.BatchNorm1d(256).to(self.device))
        self.net = nn.ModuleList(self.net)

    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        if x.ndim == 1:
            x = x.reshape(1, x.shape[0])
        for l in self.net:
            x = l(x)
        return self.act(torch.sin(self.w0 * x))
    
    def get_real_phi(self, x):
        if x.ndim == 1:
            x = x.reshape(1, x.shape[0])
        for l in self.net:
            x = l(x)
        return torch.sin(self.w0 * x)

class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias, name,
                 use_pe=True, is_first=False, w0=30.0, device="cpu", *args, **kwargs):
        super().__init__()
        self.name = name
        self.w0 = w0
        self.is_first = is_first
        self.use_pe = use_pe
        self.in_features = in_features
        self.device = device
        self.linear_sin = nn.Linear(
            in_features, out_features, bias=bias, device=device)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first and not self.use_pe:
                # If not PE we use the following initialization
                self.linear_sin.weight.uniform_(-1 / self.in_features,
                                                1 / self.in_features)
            else:
                # If PE we use the following initialization
                self.linear_sin.weight.uniform_(-np.sqrt(6 / self.in_features) / self.w0,
                                                np.sqrt(6 / self.in_features) / self.w0)

    def forward(self, input):
        return torch.sin(self.w0 * self.linear_sin(input))
