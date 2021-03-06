from torch.nn import init
from torch.nn import utils

import numpy as np
from ops import *

class Res_Block_up(nn.Module):

    def __init__(self, in_channels, out_channels, num_classes, dim_bal=True):
        super(Res_Block_up, self).__init__()

        # self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        # self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.conv1 = cConv2d(in_channels, out_channels, 3, num_classes, padding=1)
        self.conv2 = cConv2d(out_channels, out_channels, 3, num_classes, padding=1)

        # self.bn1 = ConditionalBatchNorm2d(num_classes, in_channels)
        # self.bn2 = ConditionalBatchNorm2d(num_classes, out_channels)

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.dim_bal = dim_bal

        if dim_bal:
            # self.bal_conv = nn.Conv2d(in_channels, out_channels, 1)
            self.bal_conv = cConv2d(in_channels, out_channels, 1, num_classes)
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
            h = self.bal_conv(x, c)
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

class SelfAttention(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        self.query = utils.spectral_norm(nn.Conv1d(in_channel, in_channel // 8, 1))
        self.key = utils.spectral_norm(nn.Conv1d(in_channel, in_channel // 8, 1))
        self.value = utils.spectral_norm(nn.Conv1d(in_channel, in_channel, 1))

        self.gamma = nn.Parameter(torch.zeros(1), requires_grad=True)

    def attention(self, x):
        shape = x.shape
        flatten = x.view(shape[0], shape[1], -1)
        query_key = torch.bmm(self.query(flatten).permute(0, 2, 1), self.key(flatten))
        attn = F.softmax(query_key, 1)

        return torch.bmm(self.value(flatten), attn).view(*shape)

    def forward(self, x):

        return self.gamma * self.attention(x) + x

class Generator(nn.Module):
    def __init__(self, z_dim, num_classes, SA=False):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.SA = SA

        self.dense = nn.Linear(self.z_dim, 4 * 4 * 1024)
        self.final = nn.Conv2d(64, 3, 3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.dense.weight.data, 1.)
        nn.init.xavier_uniform_(self.final.weight.data, 1.)

        self.block1 = Res_Block_up(1024, 1024, num_classes, dim_bal=False)
        self.block2 = Res_Block_up(1024, 512, num_classes)
        self.block3 = Res_Block_up(512, 256, num_classes)
        self.block4 = Res_Block_up(256, 128, num_classes)
        self.block5 = Res_Block_up(128, 64, num_classes)

        if SA:
            self.SelfAttn = SelfAttention(256)

        self.model = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            self.final,
            nn.Tanh()
        )

    def net(self, x, c):
        h = self.dense(x).view(-1, 1024, 4, 4)
        h = self.block1(h, c)
        h = self.block2(h, c)
        h = self.block3(h, c)
        if self.SA:
            h = self.SelfAttn(h)
        h = self.block4(h, c)
        h = self.block5(h, c)
        return h

    def forward(self, z, c):
        h = self.net(z, c)
        return self.model(h)

class Discriminator(nn.Module):
    def __init__(self, num_classes, SA=False):
        super(Discriminator, self).__init__()

        self.block1 = FirstBlock(3, 64)
        self.block2 = Res_Block_Down(64, 128)
        self.block3 = Res_Block_Down(128, 256)
        self.block4 = Res_Block_Down(256, 512)
        self.block5 = Res_Block_Down(512, 1024)
        self.block6 = Res_Block_Down(1024, 1024, down=False, dim_bal=False)

        if SA:
            self.SelfAttn = SelfAttention(128)
            self.model = nn.Sequential(
                self.block1,
                self.block2,
                self.SelfAttn,
                self.block3,
                self.block4,
                self.block5,
                self.block6,
                nn.ReLU(True)
            )
        else:
            self.model = nn.Sequential(
                self.block1,
                self.block2,
                self.block3,
                self.block4,
                self.block5,
                self.block6,
                nn.ReLU(True)
            )

        self.fc = utils.spectral_norm(nn.Linear(1024, 1))
        self.l_y = utils.spectral_norm(nn.Linear(1024, num_classes))

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.fc.weight.data)
        init.xavier_uniform_(self.l_y.weight.data)

    def projection(self, h, c):
        return self.fc(h) + self.l_y(h).gather(1, c.unsqueeze(-1))

    def forward(self, x, c):
        h = self.model(x)
        h = torch.sum(h, dim=(2, 3))
        return self.projection(h, c)
