
# import json
import logging
import numpy as np
from pathlib import Path
import urllib.request
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm
from typing import Callable

from .. import tools


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        nsteps: int,
    ):
        super().__init__()
        assert nsteps >= 0
        self.nsteps = nsteps

        # Encoder
        # In the encoder, convolutional layers with the Conv2d function are used to extract features from the input image.
        # Each block in the encoder consists of two convolutional layers followed by a max-pooling layer, with the exception of the last block which does not include a max-pooling layer.
        # -------

        conv_kw = {'kernel_size': 3, 'padding': 1, 'padding_mode': 'reflect'}
        ups_kw = {'kernel_size': 2, 'stride': 2}  # , 'padding_mode': 'reflect'}  # only zero-padding possible

        # input: 572x572x3
        self.e11 = nn.Conv2d(in_channels, 64, **conv_kw)  # output: 570x570x64
        self.e12 = nn.Conv2d(64, 64, **conv_kw)  # output: 568x568x64

        if self.nsteps >= 1:
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # output: 284x284x64

            # input: 284x284x64
            self.e21 = nn.Conv2d(64, 128, **conv_kw)  # output: 282x282x128
            self.e22 = nn.Conv2d(128, 128, **conv_kw)  # output: 280x280x128

        if self.nsteps >= 2:
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # output: 140x140x128

            # input: 140x140x128
            self.e31 = nn.Conv2d(128, 256, **conv_kw)  # output: 138x138x256
            self.e32 = nn.Conv2d(256, 256, **conv_kw)  # output: 136x136x256

        if self.nsteps >= 3:
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # output: 68x68x256

            # input: 68x68x256
            self.e41 = nn.Conv2d(256, 512, **conv_kw)  # output: 66x66x512
            self.e42 = nn.Conv2d(512, 512, **conv_kw)  # output: 64x64x512

        if self.nsteps >= 4:
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # output: 32x32x512

            # input: 32x32x512
            self.e51 = nn.Conv2d(512, 1024, **conv_kw)  # output: 30x30x1024
            self.e52 = nn.Conv2d(1024, 1024, **conv_kw)  # output: 28x28x1024

        # Decoder
        if self.nsteps >= 4:
            self.upconv1 = nn.ConvTranspose2d(1024, 512, **ups_kw)
            self.d11 = nn.Conv2d(1024, 512, **conv_kw)
            self.d12 = nn.Conv2d(512, 512, **conv_kw)

        if self.nsteps >= 3:
            self.upconv2 = nn.ConvTranspose2d(512, 256, **ups_kw)
            self.d21 = nn.Conv2d(512, 256, **conv_kw)
            self.d22 = nn.Conv2d(256, 256, **conv_kw)

        if self.nsteps >= 2:
            self.upconv3 = nn.ConvTranspose2d(256, 128, **ups_kw)
            self.d31 = nn.Conv2d(256, 128, **conv_kw)
            self.d32 = nn.Conv2d(128, 128, **conv_kw)

        if self.nsteps >= 1:
            self.upconv4 = nn.ConvTranspose2d(128, 64, **ups_kw)  # zero padding!
            self.d41 = nn.Conv2d(128, 64, **conv_kw)
            self.d42 = nn.Conv2d(64, 64, **conv_kw)

        # Output layer
        self.outconv = nn.Conv2d(64, out_channels, kernel_size=1, padding_mode='reflect')

    def forward(self, x_in):
        # Encoder
        xe11 = F.relu(self.e11(x_in))
        x = xe12 = F.relu(self.e12(xe11))
        if self.nsteps >= 1:
            xp1 = self.pool1(xe12)
            xe21 = F.relu(self.e21(xp1))
            x = xe22 = F.relu(self.e22(xe21))

        if self.nsteps >= 2:
            xp2 = self.pool2(xe22)
            xe31 = F.relu(self.e31(xp2))
            x = xe32 = F.relu(self.e32(xe31))

        if self.nsteps >= 3:
            xp3 = self.pool3(xe32)
            xe41 = F.relu(self.e41(xp3))
            x = xe42 = F.relu(self.e42(xe41))

        if self.nsteps >= 4:
            xp4 = self.pool4(xe42)
            xe51 = F.relu(self.e51(xp4))
            x = xe52 = F.relu(self.e52(xe51))

        # Decoder
        if self.nsteps >= 4:
            xu1 = self.upconv1(x)
            xu11 = torch.cat([xu1, xe42], dim=1)
            xd11 = F.relu(self.d11(xu11))
            x = xd12 = F.relu(self.d12(xd11))

        if self.nsteps >= 3:
            xu2 = self.upconv2(x)
            xu22 = torch.cat([xu2, xe32], dim=1)
            xd21 = F.relu(self.d21(xu22))
            x = xd22 = F.relu(self.d22(xd21))

        if self.nsteps >= 2:
            xu3 = self.upconv3(x)
            xu33 = torch.cat([xu3, xe22], dim=1)
            xd31 = F.relu(self.d31(xu33))
            x = xd32 = F.relu(self.d32(xd31))

        if self.nsteps >= 1:
            xu4 = self.upconv4(x)
            xu44 = torch.cat([xu4, xe12], dim=1)
            xd41 = F.relu(self.d41(xu44))
            x = F.relu(self.d42(xd41))

        # Output layer
        return F.sigmoid(self.outconv(x))


def infere_single(
    x: np.ndarray,
    model: Callable,
    device: torch.nn.Module = torch.device('cpu'),
) -> np.ndarray:
    # convert to torch

    transform = transforms.Compose([
        transforms.ToTensor(),  # to torch tensor
        transforms.CenterCrop(512),  # reduce large images to 512x512
        tools.networks.Grayscale(),  # convert to grayscale
    ])
    x_ = transform(x / 255.)[None].to(device)

    # infere
    y_ = model(x_)

    # convert back to numpy
    y = y_.detach().numpy()[0, 0, 1:-1, 1:-1] * 255.
    return y[..., None]


def pretrained(
    model_path: str = None,
    *,
    model_name: str = '240222160214-2804736-unet_2-alpha_0.400_grayscale_l1ws_0.25_lr_0.0001_.pt.tar',
    device: torch.nn.Module = torch.device('cpu')
):
    # model
    model = UNet(in_channels=1, out_channels=1, nsteps=2).to(device)

    # download if needed
    if model_path is None:
        model_url = f'https://github.com/uibk-uncover/sealwatch/releases/download/2025.05/{model_name}'
        cache_dir = Path(torch.hub.get_dir()) / 'sealwatch'
        resume_model_file = tools.networks.download_if_missing(model_url, cache_dir / model_name)
    else:
        resume_model_file = Path(model_path) / model_name

    # load weights
    checkpoint = torch.load(resume_model_file, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    logging.info(f'model {model_name} loaded from {resume_model_file}')
    return model


def unet_estimator(*args, **kw):
    # load model
    model = pretrained(*args, **kw)
    device = torch.device('cpu')

    def predict(x):
        y = infere_single(x, model=model, device=device)
        return y

    return predict
