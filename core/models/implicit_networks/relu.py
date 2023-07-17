import math
import torch

from torch              import nn
from ...utils.functions import BinaryQuantize

class ReLULayer(nn.Module):
    '''
        Drop in replacement for SineLayer but with ReLU non linearity
    '''

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, input):
        return nn.functional.relu(self.linear(input))


class PosEncoding(nn.Module):
    '''Module to add positional encoding as in NeRF [Mildenhall et al. 2020].'''

    def __init__(self, in_features, sidelength=None, fn_samples=None, use_nyquist=True):
        super().__init__()

        self.in_features = in_features

        if self.in_features == 3:
            self.num_frequencies = 10
        elif self.in_features == 2:
            assert sidelength is not None
            if isinstance(sidelength, int):
                sidelength = (sidelength, sidelength)
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(
                    min(sidelength[0], sidelength[1]))
        elif self.in_features == 1:
            #assert fn_samples is not None
            fn_samples = sidelength
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(
                    fn_samples)
        else:
            self.num_frequencies = 4

        self.out_dim = in_features + 2 * in_features * self.num_frequencies

    def get_num_frequencies_nyquist(self, samples):
        nyquist_rate = 1 / (2 * (2 * 1 / samples))
        return int(math.floor(math.log(nyquist_rate, 2)))

    def forward(self, coords):
        coords = coords.view(coords.shape[0], -1, self.in_features)

        coords_pos_enc = coords
        for i in range(self.num_frequencies):
            for j in range(self.in_features):
                c = coords[..., j]

                sin = torch.unsqueeze(torch.sin((2 ** i) * math.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((2 ** i) * math.pi * c), -1)

                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)

        return coords_pos_enc.reshape(coords.shape[0], -1, self.out_dim)


class ReLUNetwork(nn.Module):
    def __init__(self, in_features, hidden_features, layers,
                 out_features, outermost_linear=True, binarize=False,
                 use_pe=True, exp_freq_map=False, pe_info={},
                 sidelength=512, fn_samples=None, model="relu",
                 use_nyquist=True, device="cpu"):
        """
        exp_freq_map: Not needed
        pe_info: Not needed
        """
        super().__init__()
        self.act = BinaryQuantize().apply if binarize else nn.Identity()
        self.nonlin = ReLULayer
        self.device = device
        self.use_pe = use_pe
        if use_pe:
            self.positional_encoding = PosEncoding(in_features=in_features,
                                                   sidelength=sidelength,
                                                   fn_samples=fn_samples,
                                                   use_nyquist=use_nyquist)

            in_features = self.positional_encoding.out_dim

        self.net = []
        self.net.append(self.nonlin(in_features, hidden_features))

        for i in range(layers):
            self.net.append(self.nonlin(hidden_features, hidden_features))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features,
                                     out_features)

            self.net.append(final_linear)
        else:
            self.net.append(self.nonlin(hidden_features, out_features))
        self.net.append(nn.BatchNorm1d(256).to(self.device))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        if self.use_pe:
            coords = self.positional_encoding(coords)
        
        output = self.net(coords)
        return self.act(output)

    def get_real_phi(self, coords):
        if self.use_pe:
            coords = self.positional_encoding(coords)

        output = self.net(coords)
        return output