import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn import utils

class ConditionalBatchNorm2d(nn.Module):

    """Conditional Batch Normalization"""

    def __init__(self, num_classes, num_features):
        super(ConditionalBatchNorm2d, self).__init__()

        self.weights = nn.Embedding(num_classes, num_features)
        self.biases = nn.Embedding(num_classes, num_features)

        self.b1 = nn.BatchNorm2d(num_features, affine=False)

        self._initialize()

    def _initialize(self):
        torch.nn.init.ones_(self.weights.weight.data)
        torch.nn.init.zeros_(self.biases.weight.data)

    def forward(self, x, c):

        return self.weights(c).unsqueeze(-1).unsqueeze(-1) * self.b1(x) + self.biases(c).unsqueeze(-1).unsqueeze(-1)

class _cConvNd(nn.Module):

    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size']

    def __init__(self, in_channels, out_channels, kernel_size, num_classes, stride,
                 padding, dilation, groups, padding_mode):
        super(_cConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self.weight = Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
        self.bias = Parameter(torch.Tensor(1, out_channels, 1, 1))

        self.scales = nn.Embedding(num_classes, out_channels)
        self.shifts = nn.Embedding(num_classes, in_channels)
        torch.nn.init.ones_(self.scales.weight.data)
        torch.nn.init.zeros_(self.shifts.weight.data)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn. init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(_cConvNd, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'

class cConv2d(_cConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, num_classes, stride=1,
                 padding=0, dilation=1, groups=1, padding_mode='zeros'):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        kernel_size = [kernel_size, kernel_size]
        stride = [stride, stride]
        padding = [padding, padding]
        dilation = [dilation, dilation]
        super(cConv2d, self).__init__(
            in_channels, out_channels, kernel_size, num_classes, stride, padding, dilation, groups, padding_mode)

    def conv2d_forward(self, input, weight, c):
        b_size, c_size, height, width = input.shape

        scale = self.scales(c).view(b_size, self.out_channels, 1, 1, 1)
        shift = self.shifts(c).view(b_size, 1, self.in_channels, 1, 1) / self.kernel_size[0] / self.kernel_size[1]

        weight = (weight * scale + shift)

        return F.conv2d(input.view(1, b_size*c_size, height, width),
                        weight.view(-1, self.in_channels, self.kernel_size[0], self.kernel_size[1]),
                        None, self.stride, self.padding,
                        self.dilation, b_size).view(-1, self.out_channels, height, width) + self.bias

    def forward(self, x, c):
        return self.conv2d_forward(x, self.weight, c)

