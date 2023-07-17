import h5py
import torch
import random

import numpy as np
import torch.utils.data as data


# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
# dataset
# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#


class Dataset:
    def __init__(self, dataset_path, batch_size, patch_size, workers):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.workers = workers

    def get_arad_dataset(self, gen_dataset_path=None):
        train_dataset = AradDataset(f'{self.dataset_path}/train', gen_dataset_path=gen_dataset_path,
        # train_dataset = AradDataset(f'{self.dataset_path}/train', gen_dataset_path=gen_dataset_path,
                                    patch_size=self.patch_size, is_train=True)
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                       num_workers=self.workers, pin_memory=True)

        test_dataset = AradDataset(f'{self.dataset_path}/val', patch_size=self.patch_size, is_train=False)
        test_loader = data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False,
                                      num_workers=self.workers, pin_memory=True)

        big_test_dataset = AradDataset(f'{self.dataset_path}/val', patch_size=512, is_train=False)
        big_test_loader = data.DataLoader(big_test_dataset, batch_size=self.batch_size, shuffle=False,
                                          num_workers=self.workers, pin_memory=True)

        return train_loader, (test_loader, big_test_loader)


# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
# arad
# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#


class AradDataset(data.Dataset):
    def __init__(self, dataset_path, gen_dataset_path=None, patch_size=64, is_train=True):
        super(AradDataset, self).__init__()
        self.dataset_path = f'{dataset_path}_full_256x256x31.h5'
        self.gen_dataset_path = gen_dataset_path

        with h5py.File(self.dataset_path, 'r') as hf:
            self.base_dataset_len = int(hf['spec'].shape[0])

        self.dataset_len = self.base_dataset_len

        if gen_dataset_path is not None:
            print(f'Load generated dataset: {gen_dataset_path}')
            with h5py.File(self.gen_dataset_path, 'r') as hf:
                self.gen_dataset_len = int(hf['spec'].shape[0])

            self.dataset_len += self.gen_dataset_len

        self.is_train = is_train

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        if index < self.base_dataset_len:
            with h5py.File(self.dataset_path, 'r') as hf:
                spec = hf['spec'][index]
                rgb = hf['rgb'][index]

                spec = spec.astype(np.float32) / 65535.0
                rgb = rgb.astype(np.float32) / 65535.0

        else:
            with h5py.File(self.gen_dataset_path, 'r') as hf:
                spec = hf['spec'][index - self.base_dataset_len]
                rgb = hf['rgb'][index - self.base_dataset_len]

                spec = spec.astype(np.float32) / 65535.0
                rgb = rgb.astype(np.float32) / 65535.0

        if self.is_train == True:
            rotTimes = random.randint(0, 3)
            vFlip = random.randint(0, 1)
            hFlip = random.randint(0, 1)

            # Random rotation
            for j in range(rotTimes):
                spec = np.rot90(spec)
                rgb = np.rot90(rgb)

            # Random vertical Flip
            for j in range(vFlip):
                spec = spec[:, ::-1, :].copy()
                rgb = rgb[:, ::-1, :].copy()

            # Random horizontal Flip
            for j in range(hFlip):
                spec = spec[::-1, :, :].copy()
                rgb = rgb[::-1, :, :].copy()

        spec = torch.from_numpy(spec.copy()).permute(2, 0, 1)
        rgb = torch.from_numpy(rgb.copy()).permute(2, 0, 1)

        return rgb, spec
