import torch
import torch.nn as nn
from .positional_encoding import PE
from ...utils.functions import BinaryQuantize

class WireNetwork(nn.Module):
    def __init__(self, in_features, hidden_features, layers, out_features, w0=10, w1=10, s0=40., binarize=True,
                 outermost_linear=False, exp_freq_map=False, use_pe=True, pe_info={}, device="cpu", *args, **kwargs) -> None:
        super().__init__()
        # All results in the paper were with the default complex 'gabor' nonlinearity
        self.net = []
        self.nonlin = ComplexGaborLayer
        self.act = BinaryQuantize().apply if binarize else nn.Identity()
        self.complex = True
        self.wavelet = 'gabor'    
        self.N_freq_by_dim = pe_info.get("N_freq_by_dim")
        # Since complex numbers are two real numbers, reduce the number of 
        # hidden parameters by 2
        hidden_features = int(hidden_features/torch.sqrt(torch.Tensor([2.0])))
        dtype = torch.cfloat
        
        # Legacy parameter
        self.pe_info = pe_info
        
        if use_pe:
            self.net.append(PE(device=device, **pe_info))
            self.net.append(self.nonlin(2*sum(self.N_freq_by_dim),
                                        hidden_features, 
                                        omega0=w0,
                                        sigma0=s0,
                                        is_first=True,
                                        trainable=False))
        else:        
            self.net.append(self.nonlin(in_features,
                                        hidden_features, 
                                        omega0=w0,
                                        sigma0=s0,
                                        is_first=True,
                                        trainable=False))

        for _ in range(layers):
            self.net.append(self.nonlin(hidden_features,
                                        hidden_features, 
                                        omega0=w1,
                                        sigma0=s0))
        if outermost_linear:
            final_linear = nn.Linear(hidden_features,
                                    out_features,
                                    dtype=dtype)     
        else:
            final_linear = self.nonlin(hidden_features,
                                        out_features, 
                                        omega0=w1,
                                        sigma0=s0)
        self.net.append(final_linear)
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        output = self.net(coords)
        
        if self.wavelet == 'gabor':
            return self.act(output.real)
         
        return self.act(output)
    
    def get_real_phi(self, coords):
        output = self.net(coords)
        
        if self.wavelet == 'gabor':
            return output.real
        
        return output
    
    
class ComplexGaborLayer(nn.Module):
    '''
        Implicit representation with complex Gabor nonlinearity
        
        Inputs;
            in_features: Input features
            out_features; Output features
            bias: if True, enable bias for the linear operation
            is_first: Legacy SIREN parameter
            omega_0: Legacy SIREN parameter
            omega0: Frequency of Gabor sinusoid term
            sigma0: Scaling of Gabor Gaussian term
            trainable: If True, omega and sigma are trainable parameters
    '''
    
    def __init__(self, in_features, out_features, bias=True,
                 use_pe=True, is_first=False, w0=10.0, s0=40.0, trainable=False,  device="cpu", *args, **kwargs):
        
        super().__init__()
        self.w0 = w0
        self.s0 = s0
        self.is_first = is_first
        
        self.in_features = in_features
        
        if self.is_first:
            dtype = torch.float
        else:
            dtype = torch.cfloat
            
        # Set trainable parameters if they are to be simultaneously optimized
        self.w0 = nn.Parameter(self.w0*torch.ones(1), trainable)
        self.s0 = nn.Parameter(self.s0*torch.ones(1), trainable)
        
        self.linear = nn.Linear(in_features,
                                out_features,
                                bias=bias,
                                dtype=dtype)
    
    def forward(self, input):
        lin = self.linear(input)
        omega = self.w0 * lin
        scale = self.s0 * lin
        
        return torch.exp(1j*omega - scale.abs().square())
