import random
import scipy.io as sio

import torch
import torch.nn as nn

from einops                    import repeat
from ...utils.functions        import dims2coords
from ..implicit_networks.wire  import WireNetwork
from ..implicit_networks.relu  import ReLUNetwork
from ..implicit_networks.siren import SirenNetwork

# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
# optical encoders
# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#


class CASSI(nn.Module):
    def __init__(self, input_shape, stride, patch_size=None, mask_path=None, trainable_mask=False,
                 mask_seed=None, is_patch=True, y_norm=False, use_inr=True, inr_info={}, device='cpu', **kwargs):
        super(CASSI, self).__init__()
        self.input_shape = input_shape
        self.stride = stride
        self.patch_size = patch_size
        self.mask_path = mask_path
        self.trainable_mask = trainable_mask
        self.mask_seed = mask_seed
        self.is_patch = is_patch
        self.y_norm = y_norm
        self.use_inr = use_inr
        self.inr_info = inr_info
        self.device = device
        self.build(self.input_shape)

    def build(self, input_shape):
        M, N, L = self.input_shape
        M = M if M > self.patch_size else self.patch_size
        N = M
        if self.use_inr:
            self.coords = torch.from_numpy(dims2coords(
                (M, N))).float().to(self.device)
            if self.inr_info['model'] == 'siren':
                self.inr = SirenNetwork(device=self.device, **self.inr_info)
            elif self.inr_info['model'] == 'wire':
                self.inr = WireNetwork(device=self.device, **self.inr_info)
            elif self.inr_info['model'] == 'relu':
                self.inr = ReLUNetwork(device=self.device, **self.inr_info)
        else:
            if self.mask_path is None:
                if self.mask_seed is not None:
                    torch.manual_seed(self.mask_seed)
                    print('mask seed established: {}'.format(self.mask_seed))

                phi = (1 + torch.sign(torch.randn((M, N)))) / 2
                # phi = torch.sign(torch.randn((M, N)))
                # phi = repeat(phi, 'm n -> l m n', l=self.input_shape[-1])

            else:
                try:
                    phi = sio.loadmat(
                        f'datasets/masks/{self.mask_path}.mat')['mask']
                except:
                    phi = sio.loadmat(
                        f'datasets/masks/{self.mask_path}.mat')['CA']

                if phi.ndim == 2:
                    phi = repeat(phi, 'm n -> l m n', l=self.input_shape[-1])

                phi = torch.from_numpy(phi.astype('float32'))

            self.phi = torch.nn.Parameter(
                phi, requires_grad=self.trainable_mask)

    def set_is_patch(self, is_patch):
        self.is_patch = is_patch

    def get_phi_patch(self, phi, x_shape):
        phi_shape = phi.shape

        if x_shape[-2] <= phi_shape[-2]:  # self.is_patch:
            pxm = random.randint(0, phi_shape[1] - self.patch_size)
            pym = random.randint(0, phi_shape[2] - self.patch_size)

            return phi[:, pxm:pxm + self.patch_size:1, pym:pym + self.patch_size:1]

        else:
            patches = int(x_shape[-2] / self.patch_size)
            return repeat(phi, 'l m n -> l (rm m) (rn n)', rm=patches, rn=patches)

    def get_measurement(self, x, phi):
        b, L, M, N = x.shape
        y1 = torch.einsum('blmn,lmn->blmn', x, phi)

        # shift and sum
        y2 = torch.zeros((b, 1, M, N + self.stride * (L - 1)), device=x.device)
        for l in range(L):
            y2 += nn.functional.pad(y1[:, l, None],
                                    (self.stride * l, self.stride * (L - l - 1)))

        return y2 / L * self.stride if self.y_norm else y2

    def get_transpose(self, y, phi, mask_mul=True):
        x = torch.cat([y[..., self.stride * l:self.stride * l + y.shape[-2]]
                      for l in range(self.input_shape[-1])], dim=1)
        x = torch.einsum('blmn,lmn->blmn', x, phi) if mask_mul else x
        return x

    def get_phi(self,):
        if self.use_inr:
            return self.inr(self.coords)
        return self.phi
    
    def get_real_phi(self,):
        if self.use_inr:
            return self.inr.get_real_phi(self.coords)
        return self.phi

    def forward(self, x, only_measurement=False, only_transpose=False, mask_mul=True):
        if self.use_inr:
            phi = (self.inr(self.coords) + 1) / 2
            phi = torch.squeeze(phi)
        else:
            phi = self.phi

        phi = repeat(phi, 'm n -> l m n', l=self.input_shape[-1])

        if only_transpose:
            return self.get_transpose(x, phi, mask_mul=mask_mul)

        if only_measurement:
            return self.get_measurement(x, phi)

        return self.get_transpose(self.get_measurement(x, phi), phi, mask_mul=mask_mul)
