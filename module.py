from torch.nn import init
from torch.nn import utils

import numpy as np
from ops import *

class Res_Block_up(nn.Module):

    def __init__(self, in_channels, out_channels, num_classes, dim_bal=True):
        super(Res_Block_up, self).__init__()

        self.conv1 = utils.spectral_norm(cConv2d(in_channels, out_channels, 3, num_classes, padding=1))
        self.conv2 = utils.spectral_norm(cConv2d(out_channels, out_channels, 3, num_classes, padding=1))

        # self.cbn1 = ConditionalBatchNorm2d(num_classes, in_channels)
        # self.cbn2 = ConditionalBatchNorm2d(num_classes, out_channels)

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.dim_bal = dim_bal

        if dim_bal:
            self.bal_conv = utils.spectral_norm(nn.Conv2d(in_channels, out_channels, 1))
            init.xavier_uniform_(self.bal_conv.weight.data, 1.)

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.conv1.weight.data, 1.)
        init.xavier_uniform_(self.conv2.weight.data, 1.)

    def _upsample(self, x):
        h, w = x.size()[2:]
        return F.interpolate(x, size=(h * 2, w * 2), mode='bilinear', align_corners=True)

    def shortcut(self, x, c):
        if self.dim_bal:
            h = self.bal_conv(x)
            h = self._upsample(h)
            return h
        else:
            return self._upsample(x)

    def model(self, x, c):
        h = self.bn1(x)
        h = F.relu(h, True)
        h = self._upsample(h)
        h = self.conv1(h, c)
        h = self.bn2(h)
        h = F.relu(h, True)
        h = self.conv2(h, c)
        return h

    def forward(self, x, c):
        return self.shortcut(x, c) + self.model(x, c)

class Res_Block_Down(nn.Module):

    def __init__(self, in_channels, out_channels, down=True, dim_bal=True):
        super(Res_Block_Down, self).__init__()

        self.conv1 = utils.spectral_norm(nn.Conv2d(in_channels, out_channels, 3, padding=1))
        self.conv2 = utils.spectral_norm(nn.Conv2d(out_channels, out_channels, 3, padding=1))
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)

        if dim_bal:
            self.bal_conv = utils.spectral_norm(nn.Conv2d(in_channels, out_channels, 1))
            nn.init.xavier_uniform_(self.bal_conv.weight.data, 1.)

        self.down = down
        self.dim_bal = dim_bal

    def shortcut(self, x):
        if self.dim_bal:
            x = F.avg_pool2d(x, 2)
            x = self.bal_conv(x)
        return x

    def model(self, x):
        h = F.relu(x)
        h = self.conv1(h)
        h = F.relu(h, True)
        h = self.conv2(h)
        if self.down:
            h = F.avg_pool2d(h, 2)
        return h

    def forward(self, x):
        return self.model(x) + self.shortcut(x)

class FirstBlock(nn.Module):

    def __init__(self, in_ch, out_ch, ksize=3, pad=1, activation=F.relu):
        super(FirstBlock, self).__init__()
        self.activation = activation

        self.c1 = utils.spectral_norm(nn.Conv2d(in_ch, out_ch, ksize, 1, pad))
        self.c2 = utils.spectral_norm(nn.Conv2d(out_ch, out_ch, ksize, 1, pad))
        self.c_sc = utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 1, 1, 0))

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.c1.weight.data, 1.)
        init.xavier_uniform_(self.c2.weight.data, 1.)
        init.xavier_uniform_(self.c_sc.weight.data)

    def forward(self, x):
        return self.shortcut(x) + self.residual(x)

    def shortcut(self, x):
        return self.c_sc(F.avg_pool2d(x, 2))

    def residual(self, x):
        h = self.activation(self.c1(x), True)
        return F.avg_pool2d(self.c2(h), 2)

class Generator(nn.Module):
    def __init__(self, z_dim, num_classes):
        super(Generator, self).__init__()
        self.z_dim = z_dim

        self.dense = utils.spectral_norm(nn.Linear(self.z_dim, 4 * 4 * 1024))
        self.final = utils.spectral_norm(nn.Conv2d(64, 3, 3, stride=1, padding=1))
        nn.init.xavier_uniform_(self.dense.weight.data, 1.)
        nn.init.xavier_uniform_(self.final.weight.data, 1.)

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

    def net(self, x, c):
        h = self.dense(x).view(-1, 1024, 4, 4)
        h = self.block1(h, c)
        h = self.block2(h, c)
        h = self.block3(h, c)
        h = self.block4(h, c)
        h = self.block5(h, c)
        return h

    def forward(self, z, c):
        h = self.net(z, c)
        return self.model(h)

class Discriminator(nn.Module):
    def __init__(self, num_classes):
        super(Discriminator, self).__init__()

        self.block1 = FirstBlock(3, 64)
        self.block2 = Res_Block_Down(64, 128)
        self.block3 = Res_Block_Down(128, 256)
        self.block4 = Res_Block_Down(256, 512)
        self.block5 = Res_Block_Down(512, 1024)
        self.block6 = Res_Block_Down(1024, 1024, down=False, dim_bal=False)

        self.fc = utils.spectral_norm(nn.Linear(1024, 1))
        self.l_y = utils.spectral_norm(nn.Linear(1024, num_classes))

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.fc.weight.data)
        init.xavier_uniform_(self.l_y.weight.data)

    def net(self, x):
        h = self.block1(x)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.block6(h)
        h = F.relu(h)
        return torch.sum(h, dim=(2, 3))

    def forward(self, x, c):
        h = self.net(x)
        return self.fc(h) + self.l_y(h).gather(1, c.unsqueeze(-1))
