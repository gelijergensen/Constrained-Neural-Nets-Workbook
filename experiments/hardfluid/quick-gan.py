from functools import reduce
import numpy as np
import os

import torch
import torch.nn as nn
import torch.optim as optim

from pyinsulate.utils.train_gan_pair import train_gan_pair
from pyinsulate.utils.callbacks import OutputLogger, Callback, LossLogger
from turbulence_dataloader import get_turbulence_dataloaders
from visualization.two_dimensional import plot_slice, plot_slice_comparison
from visualization.visualize_data import (
    plot_quiver_of_3D_tensor,
    animation_of_3D_tensor,
    animation_of_slice_comparison,
)


class TurbGen(nn.Module):
    def __init__(self, in_res=32, out_res=128):
        super().__init__()
        self.conv = nn.ConvTranspose3d(3, 3, 2, stride=2)
        self.relu = nn.ReLU()
        # Intended reuse
        self.convs = [
            self.conv for i in range(int(np.log2(out_res) - np.log2(in_res)))
        ]

    def forward(self, xb):
        return reduce(lambda res, layer: self.relu(layer(res)), self.convs, xb)


class TurbDis(nn.Module):
    def __init__(self, in_res=128):
        super().__init__()
        self.conv = nn.Conv3d(3, 3, 2, stride=2)
        self.relu = nn.ReLU()
        # Intended reuse
        self.convs = [self.conv for i in range(0, int(np.log2(in_res)))]
        self.lin = nn.Linear(3, 1)

    def forward(self, xb):

        # def apl()
        x = reduce(
            lambda res, layer: self.relu(layer(res)), self.convs, xb
        ).view(1, -1)
        x = self.lin(x)

        return torch.sigmoid(x)


if __name__ == "__main__":

    path = os.path.expandvars("$SCRATCH/data/divfree-test")
    train_dl, valid_dl, test_dl = get_turbulence_dataloaders(path, batch_size=1)

    print(len(train_dl))
    print(len(valid_dl))
    print(len(test_dl))

    # gener = TurbGen()
    # discr = TurbDis()
    # print(gener)
    # print(discr)
    # if torch.cuda.is_available():
    #     print("Ooh, a gpu!")
    #     device = torch.device("cuda")
    #     gener.to(device)
    #     discr.to(device)
    #     num_gpus = torch.cuda.device_count()
    #     if num_gpus > 1:
    #         print("Many gpus!")
    #         gener = nn.DataParallel(gener, list(range(num_gpus)))
    #         discr = nn.DataParallel(discr, list(range(num_gpus)))

    # gener_opt = optim.SGD(gener.parameters(), lr=0.1)
    # discr_opt = optim.SGD(discr.parameters(), lr=0.1)

    # def should_stop(context):
    #     epoch = context.get('epoch')
    #     return epoch == 20

    # output_logger = OutputLogger(train=True, valid=True, model_type='gener')
    # gen_loss_logger = LossLogger(
    #     train=True, valid=True, model_type='gener'
    # )
    # dis_loss_logger = LossLogger(
    #     train=True, valid=True, model_type='discr'
    # )

    # class EpochPrinter(Callback):

    #     def __init__(self):
    #         super().__init__()
    #         self.epoch = None

    #     def on_train_epoch_end(self, context):
    #         self.epoch = context.get('epoch')
    #         context.get('log')(self.epoch)

    # epoch_printer = EpochPrinter()

    # train_gan_pair(
    #     gener, discr, train_dl, valid_dl, gener_opt, discr_opt, should_stop,
    #     callbacks=[output_logger, epoch_printer,
    #                gen_loss_logger, dis_loss_logger]
    # )

    # with open('output_logger.pkl', 'wb') as f:
    #     torch.save(output_logger, f)
    # with open('gen_loss_logger.pkl', 'wb') as f:
    #     torch.save(gen_loss_logger, f)
    # with open('dis_loss_logger.pkl', 'wb') as f:
    #     torch.save(dis_loss_logger, f)

    with open("output_logger.pkl", "rb") as f:
        output_logger = torch.load(f, map_location="cpu")
    with open("gen_loss_logger.pkl", "rb") as f:
        gen_loss_logger = torch.load(f, map_location="cpu")
    with open("dis_loss_logger.pkl", "rb") as f:
        dis_loss_logger = torch.load(f, map_location="cpu")

    print(gen_loss_logger.train_losses)
    print(gen_loss_logger.valid_losses)
    print(dis_loss_logger.train_losses)
    print(dis_loss_logger.valid_losses)

    train_xb_cpu = output_logger.train_xb[0].cpu()
    train_yb_cpu = output_logger.train_yb[0].cpu()
    data = [
        [train_xb_cpu, output[0], train_yb_cpu]
        for output in output_logger.train_outputs
    ]

    animation_of_slice_comparison(data, "training_comparison.gif")

    valid_xb_cpu = output_logger.valid_xb[0].cpu()
    valid_yb_cpu = output_logger.valid_yb[0].cpu()
    data = [
        [valid_xb_cpu, output[0], valid_yb_cpu]
        for output in output_logger.valid_outputs
    ]

    animation_of_slice_comparison(data, "validation_comparison.gif")

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
