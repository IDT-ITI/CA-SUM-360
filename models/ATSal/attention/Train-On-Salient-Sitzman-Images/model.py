import torch
from torchvision.models import vgg16
from torch import nn, sigmoid
from torch.nn.modules.upsampling import Upsample
from torch.nn.functional import interpolate, dropout2d
from torch.autograd import Variable
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.activation import Sigmoid, ReLU
from torch.nn import MaxPool2d
import numpy as np
import torch
import torch.cuda
import torch.nn as nn


class Downsample(nn.Module):
    # specify the kernel_size for downsampling
    def __init__(self, kernel_size):
        super(Downsample, self).__init__()
        self.pool = MaxPool2d(kernel_size)

    def forward(self, x):
        x = self.pool(x)
        return x


def make_conv_layers(cfg):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M_2':
            layers += [Downsample(kernel_size=2)]
        elif v == 'M_4':
            layers += [Downsample(kernel_size=4)]
        else:
            conv = Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv, ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'global_attention': [64, 64, 'M_2', 128, 128, 'M_2', 256, 256, 256, 'M_4', 512, 512, 512, 'M_4', 512, 512, 512],
    'based_AM': [64, 64, 'M_2', 128, 128, 'M_2', 256, 256, 256, 'M_4', 512, 512, 512, 512, 512, 512]

}
global_attention = make_conv_layers(cfg['global_attention'])
based_AM = make_conv_layers(cfg['based_AM'])


# from Encoders import  based_AM

# create pooling layer
class Downsample(nn.Module):
    # specify the kernel_size for downsampling
    def __init__(self, kernel_size):
        super(Downsample, self).__init__()
        self.pool = MaxPool2d(kernel_size)

    def forward(self, x):
        x = self.pool(x)
        return x


# create unpooling layer
class Upsample(nn.Module):
    # specify the scale_factor for upsampling
    def __init__(self, scale_factor, mode):
        super(Upsample, self).__init__()
        self.interp = interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


# create Add layer , support backprop
class Multiply(nn.Module):
    def __init__(self):
        super(Multiply, self).__init__()

    def forward(self, tensors):
        result = torch.ones(tensors[0].shape).cuda()
        for t in tensors:
            result *= t
        return result


# create Multiply layer , supprot backprop
class Add(nn.Module):
    def __init__(self):
        super(Add, self).__init__()

    def forward(self, tensors):
        result = torch.zeros(tensors[0].shape).cuda()
        for t in tensors:
            result += t
        return result


# reshape vectors layer
class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class Sal_based_Attention_module(nn.Module):

    def __init__(self):
        super(Sal_based_Attention_module, self).__init__()

        Based_Attention_Module = based_AM

        self.encoder = torch.nn.Sequential(*Based_Attention_Module)
        self.attention_module = torch.nn.Sequential(*[
            Downsample(kernel_size=2),
            Conv2d(512, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Downsample(kernel_size=2),
            Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(128, 1, kernel_size=(1, 1), stride=(1, 1), padding=0),
            Sigmoid(),
            Upsample(scale_factor=4, mode='nearest')

        ])

        # self.reshape = Reshape(-1,512,40,20)

        # define decoder based on VGG16 (inverse order and Upsampling layers , chose nearest mode)
        decoder_list = [
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Upsample(scale_factor=2, mode='nearest'),

            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Upsample(scale_factor=2, mode='nearest'),

            Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Upsample(scale_factor=2, mode='nearest'),

            Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Upsample(scale_factor=2, mode='nearest'),

            Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1), padding=0),
            Sigmoid(),
        ]

        self.decoder = torch.nn.Sequential(*decoder_list)

        print("Model initialized, Sal_based_Attention_module")

    def forward(self, input):
        x = self.encoder(input)

        y = self.attention_module(x)

        repeted = y.repeat(1, 512, 1, 1)

        product = x * repeted

        added = x + product

        x = self.decoder(added)

        return x, y


class Upsample(nn.Module):
    def __init__(self, scale_factor, mode):
        super(Upsample, self).__init__()
        self.interp = interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class SalEMA(nn.Module):
    """
    In this model, we pick a Convolutional layer from the bottleneck and apply EMA as a simple temporal regularizer.
    The smaller the alpha, the less each newly added frame will impact the outcome. This way the temporal information becomes most relevant.
    """

    def __init__(self):
        super(SalEMA, self).__init__()

        self.dropout = False
        self.residual = False
        self.use_gpu = True

        self.alpha = nn.Parameter(torch.Tensor([0.1]))
        self.ema_loc = 30  # 30 = bottleneck

        original_vgg16 = vgg16()

        # select only convolutional layers
        encoder = torch.nn.Sequential(*list(original_vgg16.features)[:30])

        # define decoder based on VGG16 (inverse order and Upsampling layers)
        decoder_list = [
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Upsample(scale_factor=2, mode='nearest'),

            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Upsample(scale_factor=2, mode='nearest'),

            Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Upsample(scale_factor=2, mode='nearest'),

            Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Upsample(scale_factor=2, mode='nearest'),

            Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1), padding=0),
            Sigmoid(),
        ]

        decoder = torch.nn.Sequential(*decoder_list)

        self.salgan = torch.nn.Sequential(*(list(encoder.children()) + list(decoder.children())))

        print("Model initialized, SalEMA")

    def forward(self, input_, prev_state=None):
        x = self.salgan[:self.ema_loc](input_)
        residual = x
        batch_size = x.data.size()[0]

        spatial_size = x.data.size()[2:]

        if self.dropout == True:
            x = dropout2d(x)
        if prev_state is None:
            current_state = self.salgan[self.ema_loc](x)
        else:
            current_state = sigmoid(self.alpha) * self.salgan[self.ema_loc](x) + (1 - sigmoid(self.alpha)) * prev_state

        if self.residual == True:
            x = current_state + residual
        else:
            x = current_state

        if self.ema_loc < len(self.salgan) - 1:
            x = self.salgan[self.ema_loc + 1:](x)

        return current_state, x


import sys

from torch.nn.functional import interpolate


def _pointwise_loss(lambd, input, target, size_average=True, reduce=True):
    d = lambd(input, target)
    if not reduce:
        return d
    return torch.mean(d) if size_average else torch.sum(d)


class KLDLoss(nn.Module):
    def __init__(self):
        super(KLDLoss, self).__init__()

    def KLD(self, inp, trg):
        inp = inp / torch.sum(inp)
        trg = trg / torch.sum(trg)
        eps = sys.float_info.epsilon

        return torch.sum((trg + eps) * torch.log(eps + torch.div(trg, (inp + eps))))

    def forward(self, inp, trg):
        return _pointwise_loss(lambda a, b: self.KLD(a, b), inp, trg)


# inn this implementation we consider standatrd normalization

'''class NSSLoss(nn.Module):
    def __init__(self):
        super(NSSLoss, self).__init__()

    # normalize saliency map
    def stand_normlize(self, x):
        # res = (x - np.mean(x)) / np.std(x)
        # x should be float tensor
        return (x - x.mean()) / x.std()

    def forward(self, sal_map, fix):
        if sal_map.size() != fix.size():
            sal_map = interpolate(sal_map, size=(fix.size()[1], fix.size()[0]))
            print(sal_map.size())
            print(fix.size())
        # bool tensor
        fix = fix > 0.1
        # Normalize saliency map to have zero mean and unit std
        sal_map = self.stand_normlize(sal_map)
        return sal_map[fix].mean()'''


class NSSLoss(nn.Module):
    def __init__(self):
        super(NSSLoss, self).__init__()

    # normalize saliency map
    def stand_normlize(self, x):
        # res = (x - np.mean(x)) / np.std(x)
        # x should be float tensor
        return (x - x.mean()) / x.std()

    def forward(self, sal_map, fix):
        if sal_map.size() != fix.size():
            sal_map = interpolate(sal_map, size=(fix.size()[1], fix.size()[0]))
            print(sal_map.size())
            print(fix.size())
        # bool tensor
        # fix = fix > 0.1
        # Normalize saliency map to have zero mean and unit std
        sal_map = self.stand_normlize(sal_map)
        N = torch.sum(fix)
        loss = -1 / N * torch.sum(sal_map * fix)

        return loss


class LOSS(nn.Module):
    def __init__(self):
        super(LOSS, self).__init__()
        self.KLDLoss = KLDLoss()
        self.NSSLoss = NSSLoss()

    def forward(self, saliency_map, gtruth, fix, fiaxtion_module):
        if fiaxtion_module.max() == 0.0:
            print('output zero')
        attention = 0.1 * (- self.NSSLoss(saliency_map, fix))
        # cc = 1.0-self.CC_Loss(saliency_map,gtruth)

        last = 0.8 * self.KLDLoss(saliency_map, gtruth) + 0.1 * self.KLDLoss(fiaxtion_module,
                                                                             Downsample(16)(gtruth)) + attention
        return last, attention