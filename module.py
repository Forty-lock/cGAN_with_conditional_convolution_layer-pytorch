import math

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import numpy as np
from ops import *

class Res_Block_up(nn.Module):

    def __init__(self, in_channels, out_channels, num_classes, dim_bal = True):
        super(Res_Block_up, self).__init__()

        self.conv1 = cConv(in_channels, out_channels, 3, num_classes, padding=1)
        self.conv2 = cConv(out_channels, out_channels, 3, num_classes, padding=1)

        self.cbn1 = ConditionalBatchNorm2d(num_classes, in_channels)
        self.cbn2 = ConditionalBatchNorm2d(num_classes, out_channels)

        self.dim_bal = dim_bal

        if dim_bal:
            self.bal_conv = cConv(in_channels, out_channels, 1, num_classes)

    def _upsample(self, x):
        h, w = x.size()[2:]
        return F.interpolate(x, size=(h * 2, w * 2), mode='bilinear')

    def forward(self, x, c):
        h = self.cbn1(x, c)
        h = F.relu(h)
        h = self._upsample(h)
        h = self.conv1(h, c)
        h = self.cbn2(h, c)
        h = F.relu(h)
        h = self.conv2(h, c)

        if self.dim_bal:
            bypass = self.bal_conv(x, c)
            bypass = self._upsample(bypass)
        else:
            bypass = self._upsample(x)

        return h + bypass

class Res_Block_Down(nn.Module):

    def __init__(self, in_channels, out_channels, down = True, dim_bal = True, First = False):
        super(Res_Block_Down, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)

        if First:
            self.model = nn.Sequential(
                SpectralNorm(self.conv1),
                nn.ReLU(),
                SpectralNorm(self.conv2),
                nn.AvgPool2d(2)
            )

            self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1)
            nn.init.xavier_uniform(self.bypass_conv.weight.data, np.sqrt(2))

            self.bypass = nn.Sequential(
                nn.AvgPool2d(2),
                SpectralNorm(self.bypass_conv),
            )
        else:
            if down:
                self.model = nn.Sequential(
                    nn.ReLU(),
                    SpectralNorm(self.conv1),
                    nn.ReLU(),
                    SpectralNorm(self.conv2),
                    nn.AvgPool2d(2)
                    )
            else:
                self.model = nn.Sequential(
                    nn.ReLU(),
                    SpectralNorm(self.conv1),
                    nn.ReLU(),
                    SpectralNorm(self.conv2)
                    )
            self.bypass = nn.Sequential()
            if dim_bal:

                self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1)
                nn.init.xavier_uniform(self.bypass_conv.weight.data, np.sqrt(2))

                self.bypass = nn.Sequential(
                    nn.AvgPool2d(2),
                    SpectralNorm(self.bypass_conv)
                )

    def forward(self, x):
        return self.model(x) + self.bypass(x)

class Generator(nn.Module):
    def __init__(self, z_dim, num_classes):
        super(Generator, self).__init__()
        self.z_dim = z_dim

        self.dense = nn.Linear(self.z_dim, 4 * 4 * 1024)
        self.final = nn.Conv2d(64, 3, 3, stride=1, padding=1)
        nn.init.xavier_uniform(self.dense.weight.data, 1.)
        nn.init.xavier_uniform(self.final.weight.data, 1.)

        self.block1 = Res_Block_up(1024, 1024, num_classes, dim_bal=False)
        self.block2 = Res_Block_up(1024, 512, num_classes)
        self.block3 = Res_Block_up(512, 256, num_classes)
        self.block4 = Res_Block_up(256, 128, num_classes)
        self.block5 = Res_Block_up(128, 64, num_classes)

        self.model = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            self.final,
            nn.Tanh()
        )

    def forward(self, z, c):
        h = self.dense(z).view(-1, 1024, 4, 4)
        h = self.block1(h, c)
        h = self.block2(h, c)
        h = self.block3(h, c)
        h = self.block4(h, c)
        h = self.block5(h, c)
        return self.model(h)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
                Res_Block_Down(3, 64, First=True),
                Res_Block_Down(64, 128),
                Res_Block_Down(128, 256),
                Res_Block_Down(256, 512),
                Res_Block_Down(512, 1024),
                Res_Block_Down(1024, 1024, down=False, dim_bal=False),
                nn.ReLU(),
            )
        self.fc = nn.Linear(1024, 1)
        nn.init.xavier_uniform(self.fc.weight.data, 1.)
        self.fc = nn.Sequential(SpectralNorm(self.fc))

    def forward(self, x):
        h = self.model(x)
        h = torch.sum(h, dim=(2, 3))
        h = self.fc(h)
        return h