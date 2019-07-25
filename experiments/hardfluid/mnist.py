from functools import reduce
import numpy as np
import os

import torch
import torch.nn as nn
import torch.optim as optim

from pyinsulate.utils.train_gan_pair import train_gan_pair
from pyinsulate.utils.callbacks import OutputLogger, Callback, LossLogger
from mnist_dataloader import get_mnist_gan_dataloaders
from visualization.two_dimensional import plot_slice, plot_slice_comparison
from visualization.visualize_data import (
    plot_quiver_of_3D_tensor,
    animation_of_3D_tensor,
    animation_of_slice_comparison,
)


class MnistGen(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(1, 49)
        self.conv1 = nn.ConvTranspose2d(1, 1, 2, stride=2)
        self.conv2 = nn.ConvTranspose2d(1, 1, 2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, xb):
        return self.relu(
            self.conv2(
                self.relu(
                    self.conv1(
                        self.relu(self.lin(xb.view(-1, 1)).view(-1, 1, 7, 7))
                    )
                )
            )
        )


class MnistDis(nn.Module):
    def __init__(self, in_res=128):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, 2, stride=2)
        self.conv2 = nn.Conv2d(1, 1, 2, stride=2)
        self.lin = nn.Linear(49, 1)
        self.relu = nn.ReLU()

    def forward(self, xb):
        y = self.relu(self.conv2(self.relu(self.conv1(xb.view(-1, 1, 28, 28)))))
        x = self.relu(self.lin(y.view(-1, 1, 49)))

        return torch.sigmoid(x)


if __name__ == "__main__":

    train_dl, valid_dl = get_mnist_gan_dataloaders()

    print(len(train_dl))
    print(len(valid_dl))

    gener = MnistGen()
    discr = MnistDis()
    print(gener)
    print(discr)
    if torch.cuda.is_available():
        print("Ooh, a gpu!")
        device = torch.device("cuda")
        gener.to(device)
        discr.to(device)
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            print("Many gpus!")
            gener = nn.DataParallel(gener, list(range(num_gpus)))
            discr = nn.DataParallel(discr, list(range(num_gpus)))

    gener_opt = optim.SGD(gener.parameters(), lr=0.01)
    discr_opt = optim.SGD(discr.parameters(), lr=0.01)

    def should_stop(context):
        epoch = context.get("epoch")
        return epoch == 20

    output_logger = OutputLogger(train=True, valid=True, model_type="gener")
    gen_loss_logger = LossLogger(train=True, valid=True, model_type="gener")
    dis_loss_logger = LossLogger(train=True, valid=True, model_type="discr")

    class EpochPrinter(Callback):
        def __init__(self):
            super().__init__()
            self.epoch = None

        def on_train_epoch_end(self, context):
            self.epoch = context.get("epoch")
            context.get("log")(self.epoch)

    epoch_printer = EpochPrinter()

    train_gan_pair(
        gener,
        discr,
        train_dl,
        valid_dl,
        gener_opt,
        discr_opt,
        should_stop,
        callbacks=[
            output_logger,
            epoch_printer,
            gen_loss_logger,
            dis_loss_logger,
        ],
    )

    with open("output_logger.pkl", "wb") as f:
        torch.save(output_logger, f)
    with open("gen_loss_logger.pkl", "wb") as f:
        torch.save(gen_loss_logger, f)
    with open("dis_loss_logger.pkl", "wb") as f:
        torch.save(dis_loss_logger, f)

    # with open('output_logger.pkl', 'rb') as f:
    #     output_logger = torch.load(
    #         f, map_location='cpu')
    # with open('gen_loss_logger.pkl', 'rb') as f:
    #     gen_loss_logger = torch.load(
    #         f, map_location='cpu')
    # with open('dis_loss_logger.pkl', 'rb') as f:
    #     dis_loss_logger = torch.load(
    #         f, map_location='cpu')

    print(gen_loss_logger.train_losses)
    print(gen_loss_logger.valid_losses)
    print(dis_loss_logger.train_losses)
    print(dis_loss_logger.valid_losses)

    def fix_image(data):
        return data.view(1, 28, 28, 1)

    train_xb_cpu = output_logger.train_xb.cpu()

    train_yb_cpu = output_logger.train_yb.cpu()
    print(fix_image(train_yb_cpu).size())
    data = [
        [fix_image(output), fix_image(train_yb_cpu)]
        for output in output_logger.train_outputs
    ]

    animation_of_slice_comparison(
        data,
        "mnist_training_comparison.gif",
        idx=0,
        plot_uvw="u",
        color_scheme="gray",
    )

    valid_xb_cpu = output_logger.valid_xb.cpu()
    valid_yb_cpu = output_logger.valid_yb.cpu()
    data = [
        [fix_image(output), fix_image(valid_yb_cpu)]
        for output in output_logger.valid_outputs
    ]

    animation_of_slice_comparison(
        data,
        "mnist_validation_comparison.gif",
        idx=0,
        plot_uvw="u",
        color_scheme="gray",
    )

    # xb, yb = next(iter(train_dl))

    # data = list()
    # for i in range(200):
    #     fake_data = np.float(i) / 200 * yb[0]
    #     data.append([xb[0], fake_data, yb[0]])

    # plot_slice(yb[0], "hue-value.png")
    # plot_slice_comparison([xb[0], yb[0]], "comparison.png")
    # animation_of_slice_comparison(data, "comp.gif")

    # plot_quiver_of_3D_tensor(xb[0], "quiver.png", color_scheme="rgb")
    # plot_slice_of_3D_tensor(yb[0], "temp.png")
    # animation_of_3D_tensor(xb[0], "animated.gif")
    # plot_slice_of_3D_tensor(train_data[0][1], "temp-rbg.png", hsv=False)

    print("done!")
