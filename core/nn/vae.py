from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        conv_only=False,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.bn = nn.BatchNorm1d(out_channels, momentum=0.05)
        self.act = nn.LeakyReLU(0.2)
        self.conv_only = False

    def forward(self, x):
        x = self.conv(x)
        if self.conv_only is False:
            x = self.bn(x)
            x = self.act(x)
        return x


class ConvTransBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        conv_only=False,
    ):
        super().__init__()
        self.conv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=stride - 1,
            dilation=dilation,
            bias=bias,
        )
        self.bn = nn.BatchNorm1d(out_channels, momentum=0.05)
        self.act = nn.LeakyReLU(0.2)
        self.conv_only = conv_only

    def forward(self, x):
        x = self.conv(x)
        if self.conv_only is False:
            x = self.bn(x)
            x = self.act(x)
        return x


class VAE(nn.Module):
    def __init__(self, latent_dim: int, **kwargs) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.arch = {
            "conv_global": [
                (1, 16, 17, 4, 8),
                (16, 32, 9, 1, 4),
                (32, 32, 5, 1, 2),
                (32, 64, 15, 2, 7),
                (64, 64, 15, 2, 7),
                (64, 64, 7, 1, 3),
                (64, 64, 7, 2, 3),
                (64, 64, 8, 1, 0),
            ],
            "conv_local": [
                (1, 32, 3, 1, 1),
                (32, 32, 3, 2, 1),
                (32, 32, 3, 1, 1),
                (32, 64, 5, 1, 2),
                (64, 64, 3, 2, 1),
                (64, 64, 3, 1, 1),
                (64, 128, 3, 2, 1),
                (128, 128, 3, 1, 1),
                (128, 128, 3, 2, 1),
                (128, 128, 3, 1, 1),
                (128, 128, 3, 2, 1),
                (128, 128, 3, 1, 1),
                (128, 128, 8, 1, 0),
            ],
            "fc": [
                (128 + 64, 128),
                (128, 128),
                (128, 64),
                (64, 64),
                (64, 64),
                (64, latent_dim),
            ],
        }

        self.encode_global = nn.Sequential(
            *[
                ConvBlock(*args, conv_only=index == 0)
                for index, args in enumerate(self.arch["conv_global"])
            ]
        )
        self.encode_local = nn.Sequential(
            *[
                ConvBlock(*args, conv_only=index == 0)
                for index, args in enumerate(self.arch["conv_local"])
            ]
        )
        self.encode_fc = nn.Sequential(
            nn.Linear(*self.arch["fc"][0]),
            nn.Linear(*self.arch["fc"][1]),
            nn.LeakyReLU(0.2),
            nn.Linear(*self.arch["fc"][2]),
            nn.Linear(*self.arch["fc"][3]),
            nn.LeakyReLU(0.2),
        )

        self.mu_fc = nn.Linear(*self.arch["fc"][5])
        self.logvar_fc = nn.Linear(*self.arch["fc"][5])

        self.decode_fc = nn.Sequential(
            nn.Linear(*self.arch["fc"][5][::-1]),
            nn.Linear(*self.arch["fc"][4][::-1]),
            nn.LeakyReLU(0.2),
            nn.Linear(*self.arch["fc"][3][::-1]),
            nn.Linear(*self.arch["fc"][2][::-1]),
            nn.LeakyReLU(0.2),
            nn.Linear(*self.arch["fc"][1][::-1]),
            nn.Linear(*self.arch["fc"][0][::-1]),
        )
        self.decode_global = nn.Sequential(
            *[
                ConvTransBlock(
                    args[1],
                    args[0],
                    *args[2:],
                    conv_only=index == (len(self.arch["conv_local"]) - 1)
                )
                for index, args in enumerate(self.arch["conv_global"][::-1])
            ]
        )
        self.decode_local = nn.Sequential(
            *[
                ConvTransBlock(
                    args[1],
                    args[0],
                    *args[2:],
                    conv_only=index == (len(self.arch["conv_local"]) - 1)
                )
                for index, args in enumerate(self.arch["conv_local"][::-1])
            ]
        )

    def encode(self, inputs: Tensor) -> List[Tensor]:
        flocal = self.encode_local(inputs)
        fglobal = self.encode_global(inputs)

        concat = torch.cat((flocal, fglobal), dim=1)
        concat = concat.squeeze(2)
        ffc = self.encode_fc(concat)

        mu = self.mu_fc(ffc)
        logvar = self.logvar_fc(ffc)

        return [mu, logvar]

    def decode(self, z: Tensor) -> Tensor:
        ffc = self.decode_fc(z)
        ffc = ffc.unsqueeze(2)

        olocal = self.decode_local(ffc[:, : self.arch["conv_local"][-1][0]])
        oglobal = self.decode_global(ffc[:, self.arch["conv_local"][-1][0] :])
        output = olocal + oglobal

        return output

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, inputs: Tensor, **kwargs) -> List[Tensor]:
        mu, logvar = self.encode(inputs)
        z = self.reparameterize(mu, logvar)
        return [self.decode(z), mu, logvar]

    @property
    def device(self):
        return next(self.parameters()).device