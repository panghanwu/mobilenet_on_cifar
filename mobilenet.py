from collections import OrderedDict
from dataclasses import dataclass, asdict
from typing import Literal
from copy import deepcopy

import torch
from torch import nn, Tensor
from torchvision.ops import Conv2dNormActivation


@dataclass
class MobileNetConfig:
    in_channels: int
    expanded_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    se_ratio: float
    activation: Literal['ReLU', 'Hardswish']

MOBILENET_LARGE_CONFIG = [
    MobileNetConfig(16, 16, 16, 3, 1, 0, 'ReLU'), 
    MobileNetConfig(16, 64, 24, 3, 2, 0, 'ReLU'), 
    MobileNetConfig(24, 72, 24, 3, 1, 0, 'ReLU'), 
    MobileNetConfig(24, 72, 40, 5, 2, 0.25, 'ReLU'), 
    MobileNetConfig(40, 120, 40, 5, 1, 0.25, 'ReLU'), 
    MobileNetConfig(40, 120, 40, 5, 1, 0.25, 'ReLU'), 
    MobileNetConfig(40, 240, 80, 3, 2, 0, 'Hardswish'), 
    MobileNetConfig(80, 200, 80, 3, 1, 0, 'Hardswish'), 
    MobileNetConfig(80, 184, 80, 3, 1, 0, 'Hardswish'), 
    MobileNetConfig(80, 184, 80, 3, 1, 0, 'Hardswish'), 
    MobileNetConfig(80, 480, 112, 3, 1, 0.25, 'Hardswish'), 
    MobileNetConfig(112, 672, 112, 3, 1, 0.25, 'Hardswish'), 
    MobileNetConfig(112, 672, 160, 5, 2, 0.25, 'Hardswish'), 
    MobileNetConfig(160, 960, 160, 5, 1, 0.25, 'Hardswish'), 
    MobileNetConfig(160, 960, 160, 5, 1, 0.25, 'Hardswish'), 
]

def make_divisible(num: int, divider: int = 8):
    return max(num // divider * divider, divider)


class SqueezeExcitationBlock(nn.Module):
    def __init__(self, in_channels: int, squeeze_channels: int) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, squeeze_channels, 1, bias=False)
        self.fc2 = nn.Conv2d(squeeze_channels, in_channels, 1, bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w = self.avgpool(x)
        w = self.fc1(w)
        w = self.relu(w)
        w = self.fc2(w)
        w = self.sigmoid(w)
        return w * x


class MobileNetBlock(nn.Module):
    """
    MobileNetV3 block composed of: 
        (1) Depthwise Separable Convolution
        (2) Inverted Residual 
        (3) Squeeze-and-Excitation
    """
    def __init__(
        self,
        in_channels: int,
        expanded_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        se_ratio: float,
        activation: Literal['ReLU', 'Hardswish']
    ) -> None:
        super().__init__()
        self.stride = self.validate_stride(stride)
        self.use_shortcut = self.stride == 1 and in_channels == out_channels
        activation = self.get_activation(activation)

        layers: OrderedDict[str, nn.Module] = OrderedDict()

        if expanded_channels != in_channels:
            layers['pointwise_expansion'] = Conv2dNormActivation(
                in_channels,
                expanded_channels,
                kernel_size=1,
                norm_layer=nn.BatchNorm2d,
                activation_layer=activation,
            )
        layers['depthwise_conv'] = Conv2dNormActivation(
            expanded_channels,
            expanded_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=expanded_channels,
            norm_layer=nn.BatchNorm2d,
            activation_layer=activation,
        )

        if se_ratio != 0:
            squeeze_channels = make_divisible(int(se_ratio*expanded_channels), 8)
            layers['se_block'] = SqueezeExcitationBlock(expanded_channels, squeeze_channels)

        layers['pointwise_conv'] = Conv2dNormActivation(
            expanded_channels, 
            out_channels, 
            kernel_size=1,
            norm_layer=nn.BatchNorm2d,
            activation_layer=None
        )
        self.conv = nn.Sequential(layers)

    @staticmethod
    def validate_stride(stride: int) -> int:
        if stride not in (1, 2):
            raise ValueError(f'stride should be 1 or 2 instead of {stride}')
        return stride
    
    @staticmethod
    def get_activation(name: Literal['ReLU', 'Hardswish']):
        match name:
            case 'ReLU':
                return nn.ReLU
            case 'Hardswish':
                return nn.Hardswish
            case _:
                raise ValueError(f'activation should be "ReLU" or "Hardswish" instead of {name}')

    def forward(self, x: Tensor) -> Tensor:
        y = self.conv(x)
        if self.use_shortcut:
            y += x
        return y
    

class MobileNet(nn.Module):
    def __init__(self, configs: list[MobileNetConfig], input_channels: int = 3) -> None:
        super().__init__()

        layers: list[nn.Module] = []

        first_layer = Conv2dNormActivation(
            input_channels,
            configs[0].in_channels,
            kernel_size=3,
            stride=2,
            norm_layer=nn.BatchNorm2d,
            activation_layer=nn.Hardswish
        )
        layers.append(first_layer)

        factor = 1
        for cfg in configs:
            layers.append(MobileNetBlock(**asdict(cfg)))
            if cfg.stride == 2:
                factor += 1

        last_channels = 6 * configs[-1].out_channels
        last_layer = Conv2dNormActivation(
            configs[-1].out_channels,
            last_channels,
            kernel_size=1,
            norm_layer=nn.BatchNorm2d,
            activation_layer=nn.Hardswish
        )
        layers.append(last_layer)
        self.backbone = nn.Sequential(*layers)

        self.input_divider = 2 ** factor
        self.last_channels = last_channels

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)
    

class MobileNetForClassification(MobileNet):
    def __init__(
        self, 
        configs: list[MobileNetConfig], 
        output_dim: int, 
        input_channels: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__(configs, input_channels)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(self.last_channels, 1280),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(1280, output_dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x

def create_mobilenet_for_cifar(num_classes: int, dropout: float = 0.1) -> MobileNet:
    cfgs = deepcopy(MOBILENET_LARGE_CONFIG)
    cfgs[1].stride = 1
    cfgs[3].stride = 1
    return MobileNetForClassification(cfgs, num_classes, dropout=dropout)