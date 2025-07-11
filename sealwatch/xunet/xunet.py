"""Implementation of Xunet.

Based on IEEE Signal Processing Letter 2016 paper:
Guanshuo Xu, Han-Zhou Wu, Yun-Qing Shi
CNN tailored to steganalysis, with facilitated statistical modeling.

Inspired by implementation by Brijesh Singh.

Author: Max Ninow
Affiliation: University of Innsbruck
"""
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class ImageProcessing(nn.Module):
    """Computes convolution with KV filter over the input tensor."""

    def __init__(self) -> None:
        """Constructor"""
        super().__init__()
        self.kv_filter = (
            torch.tensor(
                [
                    [-1.0, 2.0, -2.0, 2.0, -1.0],
                    [2.0, -6.0, 8.0, -6.0, 2.0],
                    [-2.0, 8.0, -12.0, 8.0, -2.0],
                    [2.0, -6.0, 8.0, -6.0, 2.0],
                    [-1.0, 2.0, -2.0, 2.0, -1.0],
                ]
            ).view(1, 1, 5, 5)
            / 12.0
        )

    def forward(self, inp: Tensor) -> Tensor:
        """Returns tensor convolved with KV filter"""
        kv_filter = self.kv_filter.to(inp.device)
        return F.conv2d(inp, kv_filter, stride=1, padding=2)


class ConvBlock(nn.Module):
    """This class returns a building block for XuNet."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        activation: str = "relu",
        use_abs: bool = False,
    ) -> None:
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=False
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        if activation == "tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU(inplace=True)

        self.use_abs = use_abs
        self.pool = nn.AvgPool2d(kernel_size=5, stride=2, padding=2)

    def forward(self, inp: Tensor) -> Tensor:
        """Returns conv -> batch_norm -> activation -> pooling."""
        x = self.conv(inp)
        x = self.batch_norm(x)
        if self.use_abs:
            x = torch.abs(x)
        x = self.activation(x)
        x = self.pool(x)
        return x


class XuNet(nn.Module):
    """
    XuNet: A convolutional neural network for steganalysis.

    Architecture:
    - Preprocessing with high pass filter
    - 5 convolutional layers with batch normalization and activation
    - Global average pooling
    - Fully connected layers with softmax output

    Intended for binary classification of stego and cover images.
    """

    def __init__(self) -> None:
        super().__init__()
        self.image_processing = ImageProcessing()
        self.layer1 = ConvBlock(1, 8, kernel_size=5, activation="tanh", use_abs=True)
        self.layer2 = ConvBlock(8, 16, kernel_size=5, activation="tanh")
        self.layer3 = ConvBlock(16, 32, kernel_size=1)
        self.layer4 = ConvBlock(32, 64, kernel_size=1)
        self.layer5 = ConvBlock(64, 128, kernel_size=1)
        self.pooling = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, image: Tensor) -> Tensor:
        """Forward pass"""
        x = self.image_processing(image)

        # Pass through convolutional layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        # Global average pooling
        x = self.pooling(x)

        # Flatten and pass through fully connected layers
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
