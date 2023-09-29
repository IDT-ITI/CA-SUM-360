
import sys
import torch
import torch.cuda
from torch import nn
from torch.nn import MaxPool2d
from torch.nn.functional import interpolate


class Downsample(nn.Module):
    # specify the kernel_size for downsampling
    def __init__(self, kernel_size):
        super(Downsample, self).__init__()
        self.pool = MaxPool2d(kernel_size)

    def forward(self, x):
        x = self.pool(x)
        return x
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

        return torch.sum(trg * torch.log(eps + torch.div(trg, (inp + eps))))

    def forward(self, inp, trg):
        return _pointwise_loss(lambda a, b: self.KLD(a, b), inp, trg)



class NSSLoss(nn.Module):
    def __init__(self):
        super(NSSLoss, self).__init__()

    #normalize saliency map
    def stand_normlize(self, x) :
       # res = (x - np.mean(x)) / np.std(x)
       # x should be float tensor
       return (x - x.mean())/x.std()

    def forward(self, sal_map, fix):
        if sal_map.size() != fix.size():
           sal_map = interpolate(sal_map, size= (fix.size()[1],fix.size()[0]))
           print(sal_map.size())
           print(fix.size())
        # bool tensor
        fix = fix > 0.1
        # Normalize saliency map to have zero mean and unit std
        sal_map = self.stand_normlize(sal_map)
        return sal_map[fix].mean()


class LOSS(nn.Module):
    def __init__(self):
        super(LOSS, self).__init__()
        self.KLDLoss = KLDLoss()
        self.NSSLoss = NSSLoss()

    def forward(self, saliency_map, gtruth,fix,fiaxtion_module):
        if fiaxtion_module.max() == 0.0:
            print('output zero')
        attention = 0.1 * (- self.NSSLoss(saliency_map, fix))
        #attention =0.1*(self.NSSLoss(gtruth,gtruth)- self.NSSLoss(saliency_map,gtruth))

        last = 0.8 * self.KLDLoss(saliency_map, gtruth) +0.1*self.KLDLoss(fiaxtion_module,Downsample(16)(gtruth)) + attention
        return last