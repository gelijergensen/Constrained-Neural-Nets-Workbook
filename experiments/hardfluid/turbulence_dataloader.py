"""Wraps around the turbulence dataset in order to provide proper preprocessing

Modified from the work of Max Jiang (maxjiang93)"""

from glob import glob
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader


def get_turbulence_dataloaders(dataroot, ires=32, ores=128, augment=True, frac_valid=0.2, frac_test=0.2, batch_size=8):
    """Grabs all three dataloaders, given the location of the data, options, and
    fraction of testing and validation data

    :param dataroot: location of the data files
    :param ires: input resolution (must be power of 2)
    :param ores: output resolution (must be power of 2)
    :param augment: flag to augment training data on the fly
    :param frac_valid: fraction of data to set as validation
    :param frac_test: fraction of data to set as testing
    :param batch_size: number of items in a batch
    :return: training, validation, testing
    """

    assert frac_valid + frac_test < 1.0, "Validation and Testing data" + \
        "should account for less than 1.0 of the data"

    all_files = glob("%s/*.npy" % dataroot)
    length = len(all_files)

    permuted_files = np.random.permutation(all_files)
    train_idx = int((1.0 - frac_test - frac_valid) * length)
    valid_idx = train_idx + int(frac_valid * length)
    training_files = permuted_files[:train_idx]
    validation_files = permuted_files[train_idx:valid_idx]
    testing_files = permuted_files[valid_idx:]

    train_dl = DataLoader(
        TurbulenceDataset(
            training_files, ires=ires, ores=ores, augment=augment, mode="train"
        ), shuffle=True, batch_size=batch_size
    )
    valid_dl = DataLoader(
        TurbulenceDataset(
            validation_files, ires=ires, ores=ores, augment=False, mode="val"),
        batch_size=batch_size
    )
    test_dl = DataLoader(
        TurbulenceDataset(
            testing_files, ires=ires, ores=ores, augment=False, mode="val"),
        batch_size=batch_size
    )

    return train_dl, valid_dl, test_dl


class TurbulenceDataset(Dataset):
    def __init__(self, filelist, ires=32, ores=128, augment=True, mode="train"):
        """
        :param filelist: list of files in this dataset
        :param ires: input resolution (must be power of 2)
        :param ores: output resolution (must be power of 2)
        :param augment: flag to augment data on the fly
        """
        # self.dataroot = dataroot
        # assert(mode in ["train", "test", "val"])
        # self.mode = mode
        # self.filelist = sorted(
        #     list(np.genfromtxt("data_{}.txt".format(mode), dtype=str)))
        # self.filelist = [os.path.join(dataroot, f) for f in self.filelist]

        self.filelist = filelist
        self.aug = augment
        assert(np.log2(ires) % 1 == 0 and np.log2(ores) % 1 == 0)
        self.ires = ires
        self.ores = ores
        self.idx = int(128/ires)
        self.odx = int(128/ores)

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        dat = np.load(self.filelist[idx]).astype(np.float32)
        vel = np.moveaxis(dat, -1, 0)  # [3, res, res, res]
        vel = torch.from_numpy(vel)
        n = 128
        if self.aug:
            dx, dy, dz = np.random.randint(0, n, 3)
            t = torch.arange(n)
            xx = (t+dx) % n
            yy = (t+dy) % n
            zz = (t+dz) % n
            vel = vel[:, xx, :, :][:, :, yy, :][:, :, :, zz]
        ivel = vel[:, ::self.idx, ::self.idx, ::self.idx]
        ovel = vel[:, ::self.odx, ::self.odx, ::self.odx]
        return ivel, ovel
