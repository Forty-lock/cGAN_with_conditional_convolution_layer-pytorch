import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.autograd import Variable

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

class ConditionalBatchNorm2d(nn.Module):

    """Conditional Batch Normalization"""

    def __init__(self, num_classes, num_features):
        super(ConditionalBatchNorm2d, self).__init__()

        self.weights = nn.Embedding(num_classes, num_features)
        self.biases = nn.Embedding(num_classes, num_features)

        self.b1 = nn.BatchNorm2d(num_features, affine = False)

        self._initialize()

    def _initialize(self):
        torch.nn.init.ones_(self.weights.weight.data)
        torch.nn.init.zeros_(self.biases.weight.data)

    def forward(self, input, c):

        output = self.b1(input)

        weight = self.weights(c).unsqueeze(-1).unsqueeze(-1)
        bias = self.biases(c).unsqueeze(-1).unsqueeze(-1)
        return weight * output + bias

class cConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_classes, stride=1,
                 padding=0, dilation=1):
        super(cConv, self).__init__()

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.kernel_size = kernel_size

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = Parameter(torch.Tensor(1, out_channels, in_channels, kernel_size, kernel_size))
        self.bias = Parameter(torch.Tensor(1, out_channels, 1, 1))

        self.scales = nn.Embedding(num_classes, out_channels)
        self.shifts = nn.Embedding(num_classes, in_channels)

        self._initialize()

    def _initialize(self):
        torch.nn.init.kaiming_normal(self.weight, math.sqrt(5))
        torch.nn.init.zeros_(self.bias)
        torch.nn.init.ones_(self.scales.weight.data)
        torch.nn.init.zeros_(self.shifts.weight.data)

    def forward(self, x, c):

        b_size, c_size, height, width = x.shape

        scale = self.scales(c)
        shift = self.shifts(c)

        weight = self.weight * scale.view(b_size, self.out_channels, 1, 1, 1) + \
                 shift.view(b_size, 1, self.in_channels, 1, 1)

        out = x.view(1, b_size * c_size, height, width)
        weight = weight.view(b_size * self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)

        out = F.conv2d(out, weight=weight, bias=None, stride=self.stride, dilation=self.dilation,
                       groups=b_size, padding=self.padding)

        out = out.view(b_size, self.out_channels, out.shape[-2], out.shape[-1]) + self.bias

        return out