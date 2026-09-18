from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Sequential):
    def __init__(
        self, input_channels: int, output_channels: int, dropout: float
    ) -> None:
        super().__init__(
            nn.Conv2d(
                input_channels, output_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                output_channels, output_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )


class BaselineCNN(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 64, dropout),
            ConvBlock(64, 128, dropout),
            ConvBlock(128, 256, dropout),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


class ResidualBlock(nn.Module):
    def __init__(
        self, input_channels: int, output_channels: int, stride: int = 1
    ) -> None:
        super().__init__()
        self.convolution_path = nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                output_channels, output_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(output_channels),
        )
        self.skip_path: nn.Module
        if stride != 1 or input_channels != output_channels:
            self.skip_path = nn.Sequential(
                nn.Conv2d(
                    input_channels,
                    output_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(output_channels),
            )
        else:
            self.skip_path = nn.Identity()
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.convolution_path(inputs) + self.skip_path(inputs))


class ResidualCNN(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.2) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.features = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 128, stride=2),
            nn.Dropout2d(dropout),
            ResidualBlock(128, 256, stride=2),
            nn.Dropout2d(dropout),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(self.stem(inputs)))


def build_model(name: str, dropout: float = 0.2) -> nn.Module:
    if name == "baseline":
        return BaselineCNN(dropout=dropout)
    if name == "residual":
        return ResidualCNN(dropout=dropout)
    raise ValueError(f"unknown model: {name}")
