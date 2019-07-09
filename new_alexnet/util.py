import torch.nn as nn
import numpy

class BinOp():
    def __init__(self, model, quant_mode='FL_Full'):
        # count the number of Conv2d and Linear
        count_targets = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                count_targets = count_targets + 1

        assert quant_mode in ['FL_Binary', 'FL_Full'], 'No such mode!'
        if quant_mode == 'FL_Full':
            start_range = 1
            end_range = count_targets-2
        elif quant_mode == 'FL_Binary':
            start_range = 0
            end_range = count_targets-1

        self.bin_range = numpy.linspace(start_range,
                end_range, end_range-start_range+1)\
                        .astype('int').tolist()
        self.num_of_params = len(self.bin_range)
        self.saved_params = []
        self.target_params = []
        self.target_modules = []
        index = -1
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                index = index + 1
                if index in self.bin_range:
                    tmp = m.weight.data.clone()
                    self.saved_params.append(tmp)
                    self.target_modules.append(m.weight)

    def binarization(self):
        self.meancenterConvParams()
        self.clampConvParams()
        self.save_params()
        self.binarizeConvParams()

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

    def binarizeConvParams(self):
        for index in range(self.num_of_params):
            n = self.target_modules[index].data[0].nelement()
            s = self.target_modules[index].data.size()
            if len(s) == 4:
                m = self.target_modules[index].data.norm(1, 3, keepdim=True)\
                        .sum(2, keepdim=True).sum(1, keepdim=True).div(n)
            elif len(s) == 2:
                m = self.target_modules[index].data.norm(1, 1, keepdim=True).div(n)
            self.target_modules[index].data = self.target_modules[index].data.sign()\
                    .mul(m.expand(s))

    def restore(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])

    def updateBinaryGradWeight(self):
        for index in range(self.num_of_params):
            weight = self.target_modules[index].data
            n = weight[0].nelement()
            s = weight.size()
            if len(s) == 4:
                m = weight.norm(1, 3, keepdim=True)\
                        .sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            elif len(s) == 2:
                m = weight.norm(1, 1, keepdim=True).div(n).expand(s)
            m[weight.lt(-1.0)] = 0 
            m[weight.gt(1.0)] = 0
            m = m.mul(self.target_modules[index].grad.data)
            m_add = weight.sign().mul(self.target_modules[index].grad.data)
            if len(s) == 4:
                m_add = m_add.sum(3, keepdim=True)\
                        .sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            elif len(s) == 2:
                m_add = m_add.sum(1, keepdim=True).div(n).expand(s)
            m_add = m_add.mul(weight.sign())
            self.target_modules[index].grad.data = m.add(m_add).mul(1.0-1.0/s[1]).mul(n)
            self.target_modules[index].grad.data = self.target_modules[index].grad.data.mul(1e+9)



class WeightedLoss(nn.Module):
    """
    Cross entropy with instance-wise weights. Leave `aggregate` to None to obtain a loss
    vector of shape (batch_size,).
    """
    def __init__(self, aggregate='mean'):
        super(WeightedLoss, self).__init__()
        assert aggregate in ['normal_ce_mean', 's_ce_mean', 's_ce_means_softmax','sc_ce_mean'], 'No such a mode!'
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


        elif self.aggregate == 's_ce_means_softmax':
            sep_loss = F.cross_entropy(input, target, reduce=False)

            softmax_weights = F.softmax(input)[:,target.data.cpu().numpy()].diag()
            softmax_weights.squeeze_()
            weights.squeeze_()
            print('softmax_weights:',softmax_weights)
            print('weights:', weights)
            assert sep_loss.size() == weights.size(), \
            'Required size: %r, but got: %r' % (str(sep_loss.size()),str(weights.size()))
            assert sep_loss.size() == softmax_weights.size(), \
            'Required size: %r, but got: %r' % (str(sep_loss.size()),str(softmax_weights.size()))

            return (sep_loss*Variable(weights.cuda().float())*softmax_weights).mean()            
        
