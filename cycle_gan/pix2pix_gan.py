# Copyright (c) 2018-2021, RangerUFO
#
# This file is part of cycle_gan.
#
# cycle_gan is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# cycle_gan is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with cycle_gan.  If not, see <https://www.gnu.org/licenses/>.


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


def sn_conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))


class ResBlock(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReflectionPad2d(1),
            sn_conv2d(filters, filters, 3, 1, 0),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            sn_conv2d(filters, filters, 3, 1, 0),
        )

    def forward(self, x):
        return self.net(x) + x


class ClassActivationMapping(nn.Module):
    def __init__(self, in_channels, units, activation):
        super().__init__()
        self.act = activation
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gap_linear = nn.Conv2d(in_channels, units, 1, bias=False)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.gmp_linear = nn.Conv2d(in_channels, units, 1, bias=False)
        self.out = nn.Conv2d(units * 2, units, 1)

    def forward(self, x):
        gap_y = self.gap_linear(self.gap(x))
        gap_m = self.gap_linear(x)
        gmp_y = self.gmp_linear(self.gmp(x))
        gmp_m = self.gmp_linear(x)
        return self.act(self.out(torch.cat([gap_m, gmp_m], dim=1))), torch.cat([gap_y, gmp_y], dim=1)


class ResnetGenerator(nn.Module):
    def __init__(self, channels=3, filters=64, res_blocks=9, downsample_layers=2):
        super().__init__()
        enc_layers = [
            nn.ReflectionPad2d(3),
            sn_conv2d(channels, filters, 7, 1, 0),
            nn.ReLU(inplace=True),
        ]
        for i in range(downsample_layers):
            enc_layers += [
                sn_conv2d(2 ** i * filters, 2 ** (i + 1) * filters, 3, 2, 1),
                nn.ReLU(inplace=True),
            ]
        units = 2 ** downsample_layers * filters
        for _ in range(res_blocks):
            enc_layers.append(ResBlock(units))
        self.enc = nn.Sequential(*enc_layers)

        self.cam = ClassActivationMapping(units, units, nn.ReLU(inplace=True))

        dec_layers = []
        for i in range(downsample_layers):
            in_ch = 2 ** (downsample_layers - i) * filters
            out_ch = 2 ** (downsample_layers - i - 1) * filters
            dec_layers += [
                nn.Upsample(scale_factor=2, mode='nearest'),
                sn_conv2d(in_ch, out_ch, 3, 1, 1),
                nn.ReLU(inplace=True),
            ]
        dec_layers += [
            nn.ReflectionPad2d(3),
            sn_conv2d(filters, channels, 7, 1, 0),
            nn.Tanh(),
        ]
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        x, y = self.cam(self.enc(x))
        return self.dec(x), y


class PatchDiscriminator(nn.Module):
    def __init__(self, channels=3, filters=64, layers=3):
        super().__init__()
        enc_layers = [
            sn_conv2d(channels, filters, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        for i in range(1, layers):
            in_ch = min(2 ** (i - 1), 8) * filters
            out_ch = min(2 ** i, 8) * filters
            enc_layers += [
                sn_conv2d(in_ch, out_ch, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        units = min(2 ** layers, 8) * filters
        enc_layers += [
            sn_conv2d(min(2 ** (layers - 1), 8) * filters, units, 4, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        self.enc = nn.Sequential(*enc_layers)
        self.cam = ClassActivationMapping(units, units, nn.LeakyReLU(0.2, inplace=True))
        self.dec = sn_conv2d(units, 1, 4, 1, 1)

    def forward(self, x):
        x, y = self.cam(self.enc(x))
        return self.dec(x), y


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


if __name__ == "__main__":
    device = torch.device("cpu")
    net_g = ResnetGenerator()
    net_g.apply(init_weights)
    net_d = PatchDiscriminator()
    net_d.apply(init_weights)
    real_in = torch.zeros(4, 3, 256, 256)
    real_out = torch.ones(4, 3, 256, 256)
    real_y, real_cam_y = net_d(real_out)
    print("real_y: ", torch.sigmoid(real_y))
    print("real_cam_y: ", torch.sigmoid(real_cam_y))
    fake_out, gen_cam_y = net_g(real_in)
    print("fake_out: ", fake_out)
    print("gen_cam_y: ", torch.sigmoid(gen_cam_y))
    fake_y, fake_cam_y = net_d(fake_out)
    print("fake_y: ", torch.sigmoid(fake_y))
    print("fake_cam_y: ", torch.sigmoid(fake_cam_y))
