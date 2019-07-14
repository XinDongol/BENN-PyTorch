import torch.nn as nn
import numpy
import torch.nn.functional as F
import torch
from torch.autograd import Variable

def sto_quant(tensor):
    return tensor.add(1.).div(2.).add(torch.rand(tensor.size()).cuda()\
        .add(-0.5)).clamp(0.,1.).round().mul(2.).add(-1.)   

class BinOp():
    def __init__(self, model, mode='allbin'):
        # count the number of Conv2d
        count_Conv2d = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                count_Conv2d = count_Conv2d + 1

        assert mode in ['allbin','nin'], 'No such a mode!'
        if mode == 'allbin':
            start_range = 0
            end_range = count_Conv2d-1
        elif mode == 'nin':
            start_range = 1
            end_range = count_Conv2d-2
            
        self.bin_range = numpy.linspace(start_range,
                end_range, end_range-start_range+1)\
                        .astype('int').tolist()
        self.num_of_params = len(self.bin_range)
        self.saved_params = []
        self.target_params = []
        self.target_modules = []
        index = -1
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                index = index + 1
                if index in self.bin_range:
                    tmp = m.weight.data.clone()
                    self.saved_params.append(tmp)
                    self.target_modules.append(m.weight)

    def binarization(self, quant_mode='det'):
        self.meancenterConvParams()
        self.clampConvParams()
        self.save_params()
        self.binarizeConvParams(quant_mode)

    def meancenterConvParams(self):
        for index in range(self.num_of_params):
            s = self.target_modules[index].data.size()
            negMean = self.target_modules[index].data.mean(1, keepdim=True).\
                    mul(-1).expand_as(self.target_modules[index].data)
            self.target_modules[index].data = self.target_modules[index].data.add(negMean)

    def clampConvParams(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data = self.target_modules[index].data.clamp(min=-1.0, max=1.0)

    def save_params(self):
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data) 

    def binarizeConvParams(self, quant_mode):
        assert quant_mode in ['det', 'sto'], 'No such a quant_mode'
        for index in range(self.num_of_params):
            n = self.target_modules[index].data[0].nelement()
            s = self.target_modules[index].data.size()
            m = self.target_modules[index].data.norm(1, 3, keepdim=True)\
                    .sum(2, keepdim=True).sum(1, keepdim=True).div(n)
            if quant_mode=='det':
                self.target_modules[index].data = self.target_modules[index].data.sign().mul(m.expand(s))
                #self.target_modules[index].data.sign()\
                #        .mul(m.expand(s), out=self.target_modules[index].data)
            elif quant_mode=='sto':
                self.target_modules[index].data = sto_quant(self.target_modules[index].data).mul(m.expand(s))
                #sto_quant(self.target_modules[index].data)\
                #        .mul(m.expand(s), out=self.target_modules[index].data)        

    # copy the full precision value back to weight
    def restore(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])

    def updateBinaryGradWeight(self):
        for index in range(self.num_of_params):
            weight = self.target_modules[index].data
            n = weight[0].nelement()
            s = weight.size()
            m = weight.norm(1, 3, keepdim=True)\
                    .sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            m[weight.lt(-1.0)] = 0 
            m[weight.gt(1.0)] = 0
            # m = m.add(1.0/n).mul(1.0-1.0/s[1]).mul(n)
            # self.target_modules[index].grad.data = \
            #         self.target_modules[index].grad.data.mul(m)
            m = m.mul(self.target_modules[index].grad.data)
            m_add = weight.sign().mul(self.target_modules[index].grad.data)
            m_add = m_add.sum(3, keepdim=True)\
                    .sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            m_add = m_add.mul(weight.sign())
            self.target_modules[index].grad.data = m.add(m_add).mul(1.0-1.0/s[1]).mul(n)

 
class WeightedLoss(nn.Module):
    """
    Cross entropy with instance-wise weights. Leave `aggregate` to None to obtain a loss
    vector of shape (batch_size,).
    """
    def __init__(self, aggregate='mean'):
        super(WeightedLoss, self).__init__()
        assert aggregate in ['normal_ce_mean', 's_ce_mean', 'sc_ce_mean'], 'No such a mode'
        self.aggregate = aggregate

    def forward(self, input, target, weights=None):
        if self.aggregate == 'normal_ce_mean':

            return F.cross_entropy(input, target)
        elif self.aggregate == 's_ce_mean':
            sep_loss = F.cross_entropy(input, target, reduce=False)
            weights.squeeze_()
            assert sep_loss.size() == weights.size(), \
            'Required size: %r, but got: %r' % (str(sep_loss.size()),str(weights.size()))

            return (sep_loss*Variable(weights.cuda().float())).mean()
        elif self.aggregate == 'sc_ce_mean':

            batch_size = target.data.nelement()
            #print('weights',weights)
            #print('target',target)
            oned_weights = weights[:,target.data.cpu().numpy()].diag()
            #print('oned_weights', oned_weights)
            sep_loss = F.cross_entropy(input, target, reduce=False)
            assert sep_loss.size() == oned_weights.size(), \
            'Required size: %r, but got: %r' % (str(sep_loss.size()),str(oned_weights.size()))

            return (sep_loss*Variable(oned_weights.cuda().float())).mean()
