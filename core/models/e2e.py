import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path

from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import MeanSquaredError, MeanAbsoluteError, StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

from .optical_encoders.cassi import CASSI
from .computational_decoders.dgsmp import DGSMP
from .computational_decoders.dssp import DSSP
from .computational_decoders.unet import UNet

from ..utils.spec2rgb import SpectralSensitivity
from ..utils.functions import AverageMeter


# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
# baseline
# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#

class M2S:
    def __init__(self, model, input_shape, param=0.2, lr=1e-4, save_path=None, optical_info={}, device='cpu', *args, **kwargs):
        self.model = model
        self.input_shape = input_shape
        self.writer = SummaryWriter(save_path)
        self.save_path = save_path
        self.device = device
        self.num_blocking = torch.cuda.is_available()
        self.param = param
        self.optical_info = optical_info

        self.MSE = MeanSquaredError().to(device)
        self.MAE = MeanAbsoluteError().to(device)
        self.SSIM = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.PSNR = PeakSignalNoiseRatio().to(device)

        self.spec_sensitivity = SpectralSensitivity(
            'Canon60D', bands=input_shape[-1], device=device)

        self.save_image_path = f'{save_path}/images'
        Path(self.save_image_path).mkdir(parents=True, exist_ok=True)

        # Model

        y_norm = True if model == 'dgsmp' else False

        self.optical_encoder = CASSI(
            input_shape=input_shape, y_norm=y_norm, device=self.device, **optical_info)

        if model == 'unet':
            L = input_shape[-1]
            self.computational_decoder = UNet(
                in_channels=L, out_channels=L, nfilters=32, bilinear=False)
            self.criterion = nn.MSELoss()
            self.optimizer = torch.optim.Adam(
             #   self.optical_encoder.parameters(), lr=lr)
                list(self.computational_decoder.parameters()) + list(self.optical_encoder.parameters()), lr=lr)
            self.scheduler = None
            
        if model == 'dssp':
            L = input_shape[-1]
            self.computational_decoder = DSSP(in_channels=L, out_channels=L, stages=8,
                                              optical_encoder=self.optical_encoder)

            self.criterion = nn.MSELoss()
            self.optimizer = torch.optim.Adam(
                list(self.computational_decoder.parameters()) + list(self.optical_encoder.parameters()), lr=lr)
            # self.optimizer = torch.optim.Adam(
            #     self.computational_decoder.parameters(), lr=lr)
            # torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)
            self.scheduler = None

        elif model == 'dgsmp':
            self.computational_decoder = DGSMP(
                Ch=input_shape[-1], stages=4, device=device)

            self.criterion = nn.L1Loss()
            self.optimizer = torch.optim.Adam(
                list(self.computational_decoder.parameters()) + list(self.optical_encoder.parameters()), lr=lr)
            # self.optimizer = torch.optim.Adam(self.computational_decoder.parameters(), lr=lr, betas=(0.9, 0.999),
            #                                   eps=1e-8)
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=[], gamma=0.1)

        # else:
        #     raise NotImplementedError(f"Model not found: {model}")

        self.optical_encoder.to(device)
        self.computational_decoder.to(device)
        self.criterion.to(device)

    def physics_informed_ca(self, param):
        x_ones = torch.ones([1,] + self.input_shape[::-1]).to(self.device)
        y_ones = self.optical_encoder(x_ones)
        return torch.std(y_ones) * param

    def train(self, train_loader, init_epoch, epochs, val_loader=None):

        print("Beginning training")

        time_begin = time.time()
        for epoch in range(init_epoch, epochs):
            self.optical_encoder.set_is_patch(True)
            train_metrics = self.train_step(train_loader, epoch, epochs)

            for key, value in train_metrics.items():
                self.writer.add_scalar(f'train_{key}', value, epoch)

            if val_loader is not None:
                self.optical_encoder.set_is_patch(False)
                test_metrics = self.test_step(val_loader, epoch, epochs)

                for key, value in test_metrics.items():
                    self.writer.add_scalar(f'test_{key}', value, epoch)

            # lr and reg scheduler
            self.writer.add_scalar(
                "lr", self.optimizer.param_groups[0]['lr'], epoch)

            if self.scheduler is not None:
                self.scheduler.step(epoch)

            self.save_coded_aperture(epoch)
            self.save_images(val_loader, epoch)
            self.save_checkpoint(f'{self.save_path}/checkpoints', 0)

        print("Ending training")

        total_mins = (time.time() - time_begin) / 60
        print(f'Script finished in {total_mins:.2f} minutes')

    def train_step(self, dataloader, epoch, max_epochs):
        self.optical_encoder.train()
        self.computational_decoder.train()

        return self.forward(dataloader, epoch, max_epochs, kind='train', colour='red')

    def test_step(self, dataloader, epoch, max_epochs):
        with torch.no_grad():
            self.optical_encoder.eval()
            self.computational_decoder.eval()

            return self.forward(dataloader, epoch, max_epochs, kind='test', colour='green')

    def forward(self, dataloader, epoch, max_epochs, kind, colour):
        losses = AverageMeter()

        mse_x_losses = AverageMeter()
        mae_x_losses = AverageMeter()
        ssim_x_losses = AverageMeter()
        psnr_x_losses = AverageMeter()
        pi_x_losses = AverageMeter()
        dict_metrics = dict()
        data_loop = tqdm(enumerate(dataloader), total=len(
            dataloader), colour=colour)

        for _, data in data_loop:
            _, x = data

            x = Variable(x.to(self.device, non_blocking=self.num_blocking))
            y = self.optical_encoder(x, only_measurement=True)

            if self.model == 'dgsmp':
                x_hat = self.computational_decoder(y)

            else:
                #x_hat = self.optical_encoder(x)
                x0 = self.optical_encoder(x)
                x_hat = self.computational_decoder(x0, y)


            # pi_phi = self.physics_informed_ca(self.param)
            loss = self.criterion(x_hat, x) # + pi_phi
            losses.update(loss.item(), x.size(0))

            mse_x = self.MSE(x_hat, x)
            mae_x = self.MAE(x_hat, x)
            ssim_x = self.SSIM(x_hat, x)
            psnr_x = self.PSNR(x_hat, x)

            # pi_x_losses.update(pi_phi.item(), x.size(0))

            mse_x_losses.update(mse_x.item(), x.size(0))
            mae_x_losses.update(mae_x.item(), x.size(0))
            ssim_x_losses.update(ssim_x.item(), x.size(0))
            psnr_x_losses.update(psnr_x.item(), x.size(0))

            dict_metrics = dict(loss=losses.avg,
                                mse_x=mse_x_losses.avg, mae_x=mae_x_losses.avg,
                                ssim_x=ssim_x_losses.avg, psnr_x=psnr_x_losses.avg, pi_phi=pi_x_losses.avg)

            if kind == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            lr = format(self.optimizer.param_groups[0]['lr'], '.1e')
            data_loop.set_description(
                f'{kind.capitalize()}: Epoch [{epoch + 1} / {max_epochs}] lr: {lr}')
            data_loop.set_postfix(**dict_metrics)

        return dict_metrics

    def save_images(self, val_loader, epoch, save=True, show=False):
        self.optical_encoder.eval()
        self.computational_decoder.eval()

        for i, data in enumerate(val_loader):
            _, spec = data

            if i == 5:
                break

        # data = [(rgb, spec) for i, (rgb, spec) in enumerate(val_loader) if i < 5]
        # _, spec = next(iter(val_loader))
        spec = spec.to(self.device, non_blocking=self.num_blocking)

        with torch.no_grad():
            y = self.optical_encoder(spec, only_measurement=True)

            if self.model == 'dgsmp':
                reconstructed = self.computational_decoder(y)

            else:
                x0 = self.optical_encoder(spec)
                reconstructed = self.computational_decoder(x0, y)

        y = y.permute(0, 2, 3, 1).cpu().numpy()

        if spec.shape[1] == 3:
            rgb_reconstructed = reconstructed.permute(0, 2, 3, 1).cpu().numpy()
            rgb_spec = spec.permute(0, 2, 3, 1).cpu().numpy()

        else:
            rgb_reconstructed = self.spec_sensitivity.get_rgb_01(
                reconstructed).permute(0, 2, 3, 1).cpu().numpy()
            rgb_spec = self.spec_sensitivity.get_rgb_01(
                spec).permute(0, 2, 3, 1).cpu().numpy()

        indices = np.linspace(0, len(rgb_spec) - 1, 4).astype(int)

        plt.figure(figsize=(30, 40))

        count = 1
        for idx in indices:
            plt.subplot(4, 3, count)
            plt.imshow(y[idx], cmap='gray')
            plt.title('Measurement')
            plt.axis('off')

            plt.subplot(4, 3, count + 1)
            plt.imshow(np.clip(
                rgb_reconstructed[idx], a_min=0., a_max=1.) / np.max(rgb_reconstructed[idx]))
            plt.title(f'ssim: {self.SSIM(reconstructed[idx:idx + 1], spec[idx:idx + 1]).item():.4f},'
                      f'psnr: {self.PSNR(reconstructed[idx:idx + 1], spec[idx:idx + 1]).item():.4f}')
            plt.axis('off')

            plt.subplot(4, 3, count + 2)
            plt.imshow(rgb_spec[idx] / np.max(rgb_spec[idx]))
            plt.title('GT')
            plt.axis('off')

            count += 3

        if save:
            plt.savefig(
                '{}/recons_{:03d}.png'.format(self.save_image_path, epoch))

        if show:
            plt.show()

        plt.close()

    def save_full_images(self, val_loader, epoch, save=True, show=False):
        self.optical_encoder.eval()
        self.computational_decoder.eval()

        _, spec = next(iter(val_loader))
        spec = spec.to(self.device, non_blocking=self.num_blocking)

        with torch.no_grad():
            y = self.optical_encoder(spec, only_measurement=True)

            if self.model == 'dgsmp':
                reconstructed = self.computational_decoder(y)

            else:
                x0 = self.optical_encoder(spec)
                reconstructed = self.computational_decoder(x0, y)

        y = y.permute(0, 2, 3, 1).cpu().numpy()

        if spec.shape[1] == 3:
            rgb_reconstructed = reconstructed.permute(0, 2, 3, 1).cpu().numpy()
            rgb_spec = spec.permute(0, 2, 3, 1).cpu().numpy()

        else:
            rgb_reconstructed = self.spec_sensitivity.get_rgb_01(
                reconstructed).permute(0, 2, 3, 1).cpu().numpy()
            rgb_spec = self.spec_sensitivity.get_rgb_01(
                spec).permute(0, 2, 3, 1).cpu().numpy()

        # save images

        indices = np.linspace(0, len(rgb_spec) - 1, 4).astype(int)

        plt.figure(figsize=(30, 40))

        count = 1
        for idx in indices:
            plt.subplot(4, 3, count)
            plt.imshow(y[idx], cmap='gray')
            plt.title('Measurement')
            plt.axis('off')

            plt.subplot(4, 3, count + 1)
            plt.imshow(np.clip(
                rgb_reconstructed[idx], a_min=0., a_max=1.) / np.max(rgb_reconstructed[idx]))
            plt.title(f'ssim: {self.SSIM(reconstructed[idx:idx + 1], spec[idx:idx + 1]).item():.4f},'
                      f'psnr: {self.PSNR(reconstructed[idx:idx + 1], spec[idx:idx + 1]).item():.4f}')
            plt.axis('off')

            plt.subplot(4, 3, count + 2)
            plt.imshow(rgb_spec[idx] / np.max(rgb_spec[idx]))
            plt.title('GT')
            plt.axis('off')

            count += 3

        if save:
            plt.savefig(
                '{}/big_recons_{:03d}.png'.format(self.save_image_path, epoch))

        if show:
            plt.show()

        plt.close()

    def save_checkpoint(self, path, epoch=None):
        if epoch is None:
            torch.save(self.optical_encoder.state_dict(),
                       '{}/optical_encoder.pth'.format(path))
            torch.save(self.computational_decoder.state_dict(),
                       '{}/computational_decoder.pth'.format(path))
        else:
            torch.save(self.optical_encoder.state_dict(),
                       '{}/optical_encoder_{}.pth'.format(path, epoch))
            torch.save(self.computational_decoder.state_dict(),
                       '{}/computational_decoder_{}.pth'.format(path, epoch))

    def load_checkpoint(self, path, epoch=None, is_encoder=True, is_decoder=True):
        if epoch is None:
            if is_encoder:
                self.optical_encoder.load_state_dict(
                    torch.load('{}/optical_encoder.pth'.format(path)))

            if is_decoder:
                self.computational_decoder.load_state_dict(
                    torch.load('{}/computational_decoder.pth'.format(path)))

        else:
            if is_encoder:
                self.optical_encoder.load_state_dict(torch.load(
                    '{}/optical_encoder_{}.pth'.format(path, epoch)))

            if is_decoder:
                self.computational_decoder.load_state_dict(
                    torch.load('{}/computational_decoder_{}.pth'.format(path, epoch)))

    def save_coded_aperture(self, epoch):
        phi = self.optical_encoder.get_phi().detach().cpu().numpy()
        real_phi = self.optical_encoder.get_real_phi().detach().cpu().numpy()
        phi = np.squeeze(phi)
        real_phi = np.squeeze(real_phi)
        _, ax = plt.subplots(1,2, figsize=(10,5)) 
        ax[0].imshow(phi, cmap='gray')
        ax[0].set_title('Coded Aperture')
        im = ax[1].imshow(real_phi, cmap='gray')
        ax[1].set_title('Coded Aperture real')
        plt.colorbar(im, ax=ax[1])
        path = Path(f"{self.save_image_path}/phi")
        
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{path}/phi_{epoch}.png")
        plt.close()
