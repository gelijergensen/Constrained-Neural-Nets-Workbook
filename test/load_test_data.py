import gzip
from pathlib import Path
import pickle
import torch
from torch.utils.data import TensorDataset, DataLoader

DATA_DIR = Path(".temp")


def get_mnist_dataloaders(batch_size=32):
    filename = Path("mnist.pkl.gz")
    with gzip.open(DATA_DIR / filename, "rb") as f:
        ((x_train, y_train), (x_valid, y_valid),
         _) = pickle.load(f, encoding="latin-1")

    x_train, y_train, x_valid, y_valid = map(
        torch.as_tensor, (x_train, y_train, x_valid, y_valid)
    )

    train_dl = DataLoader(TensorDataset(
        x_train, y_train), batch_size=batch_size, shuffle=True)
    valid_dl = DataLoader(TensorDataset(
        x_valid, y_valid), batch_size=batch_size * 2)
    return train_dl, valid_dl


def get_mnist_gan_dataloaders(batch_size=32):
    # Here, we swap inputs and targets because we want to produce images from
    # the class labels
    filename = Path("mnist.pkl.gz")
    with gzip.open(DATA_DIR / filename, "rb") as f:
        ((y_train, x_train), (y_valid, x_valid),
         _) = pickle.load(f, encoding="latin-1")

    x_train, y_train, x_valid, y_valid = map(
        torch.as_tensor, (x_train, y_train, x_valid, y_valid)
    )

    x_train = x_train.type(torch.float)
    x_valid = x_valid.type(torch.float)

    train_dl = DataLoader(TensorDataset(
        x_train, y_train), batch_size=batch_size, shuffle=True)
    valid_dl = DataLoader(TensorDataset(
        x_valid, y_valid), batch_size=batch_size * 2)
    return train_dl, valid_dl
