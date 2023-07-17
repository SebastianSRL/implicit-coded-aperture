import torch
from torch import nn
import numpy as np


class PE(nn.Module):
    def __init__(
        self,
        N_freq_by_dim,
        cfg,
        deterministic=True,
        exp_freq_map=False,
        std=1.0,
        device="cpu",
        *args,
        **kwargs
    ):
        super().__init__()
        self.device = device
        self.cfg = cfg
        self.N = self.cfg.PE.FREQ

        # If exp_freq_map is True, then the frequency map will be exponential, otherwise it will be linear.
        if self.cfg.PE.EXP:
            self.freq_map = lambda x: torch.pi * torch.pow(2, x - 1)
        else:
            self.freq_map = lambda x: x * torch.pi / 2

        if self.cfg.PE.DETERMINISTIC:
            self.freqs = [
                self.freq_map(torch.arange(1, n + 1).to(device).reshape(1, 1, n))
                for n in self.N
            ]
        else:
            self.freqs = [
                torch.randn((1, 1, n), device=device) * 2 * np.pi * std for n in self.N
            ]

    def pe_cod(self, input, idx_freq):
        input = input.unsqueeze(-1)
        freqs = self.freqs[idx_freq].to(self.device)
        pe = torch.cat((torch.cos(freqs * input), torch.sin(freqs * input)), -1)
        return pe

    def forward(self, input):
        output = torch.cat(
            [
                self.pe_cod(x, n)
                for x, n in zip(
                    torch.unbind(input=input, dim=-1), list(range(len(self.N)))
                )
            ],
            -1,
        )
        return output
