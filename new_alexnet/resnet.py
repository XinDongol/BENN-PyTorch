import torch
import os
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

__all__ = ['resnet', 'ResNet']

class BinActive(torch.autograd.Function):
    '''
    Binarize the input activations and calculate the mean across channel dimension.
    '''
    def forward(self, input):
        self.save_for_backward(input)
        #size = input.size()
        input = input.sign()
        return input

    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        #grad_input[input.ge(1)] = 0
        #grad_input[input.le(-1)] = 0
        return grad_input * input.le(1).float() * input.ge(-1).float()


class BinConv2d(nn.Module): # change the name of BinConv2d
    def __init__(self, input_channels, output_channels,
            kernel_size=-1, stride=1, padding=0, groups=1, dropout=0,
            Linear=False, is_ac=True):
        super(BinConv2d, self).__init__()
        self.layer_type = 'BinConv2d'
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout_ratio = dropout

        if dropout!=0:
            self.dropout = nn.Dropout(dropout)
        self.Linear = Linear
        if not self.Linear:
            self.bn = nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True)
            self.conv = nn.Conv2d(input_channels, output_channels,
                    kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
        else:
            self.bn = nn.BatchNorm1d(input_channels, eps=1e-4, momentum=0.1, affine=True)
            self.linear = nn.Linear(input_channels, output_channels)
        self.relu = nn.ReLU(inplace=True)
        self.is_ac = is_ac
    
    def forward(self, x):
        x = self.bn(x)
        x = BinActive()(x)
        if self.dropout_ratio!=0:
            x = self.dropout(x)
        if not self.Linear:
            x = self.conv(x)
        else:
            x = self.linear(x)
        if self.is_ac:
            x = self.relu(x)
        return x


class RealConv2d(nn.Module):
    def __init__(self, input_channels, output_channels,
            kernel_size=-1, stride=-1, padding=-1, groups=1, dropout=0,
            Linear=False):
        super(RealConv2d, self).__init__()
        self.layer_type = 'RealConv2d'
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout_ratio = dropout

        if dropout!=0:
            self.dropout = nn.Dropout(dropout)
        self.Linear = Linear
        if not self.Linear:
            self.bn = nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True)
            self.conv = nn.Conv2d(input_channels, output_channels,
                    kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
        else:
            self.bn = nn.BatchNorm1d(input_channels, eps=1e-4, momentum=0.1, affine=True)
            self.linear = nn.Linear(input_channels, output_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.bn(x)
        #x = BinActive()(x)
        if self.dropout_ratio!=0:
            x = self.dropout(x)
        if not self.Linear:
            x = self.conv(x)
        else:
            x = self.linear(x)
        x = self.relu(x)
        return x

def Binaryconv3x3(in_planes, out_planes, stride=1, is_ac=True):
    "3x3 convolution with padding"
    return BinConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, is_ac=is_ac)

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1)

def init_model(model):
    for m in model.modules():
        if isinstance(m, BinConv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, do_bntan=True):
        super(BasicBlock, self).__init__()

        self.conv1 = Binaryconv3x3(inplanes, planes, stride, is_ac=True)
        self.conv2 = Binaryconv3x3(planes, planes, is_ac=False)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.do_bntan=do_bntan;
        self.stride = stride

    def forward(self, x):

        residual = x.clone()

        out = self.conv1(x)
        out = self.conv2(out)


        if self.downsample is not None:
            #if residual.data.max()>1:
            #    import pdb; pdb.set_trace()
            residual = self.downsample(residual)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = BinConv2d(inplanes, planes, kernel_size=1, is_ac=True)

        self.conv2 = BinConv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, is_ac=True)
        
        self.conv3 = BinConv2d(planes, planes * 4, kernel_size=1, is_ac=False)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        #import pdb; pdb.set_trace()
        out = self.conv1(x)

        out = self.conv2(out)

        out = self.conv3(out)


        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = BinConv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, is_ac=False)
    
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks-1):
            layers.append(block(self.inplanes, planes))
        layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.bn1(x)
        x = self.tanh1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.bn2(x)
        x = self.tanh2(x)
        x = self.fc(x)
        #x = self.bn3(x)
        #x = self.logsoftmax(x)

        return x


class ResNet_imagenet(ResNet):

    def __init__(self, num_classes=1000,
                 block=Bottleneck, layers=[3, 4, 23, 3]):
        super(ResNet_imagenet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=True)
        self.tanh1 = nn.ReLU(inplace=True)
        self.tanh2 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.bn2 = nn.BatchNorm1d(512 * block.expansion, eps=1e-4, momentum=0.1, affine=True)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        #self.bn3 = nn.BatchNorm1d(1000)
        #self.logsoftmax = nn.LogSoftmax()

        #init_model(self)
        self.regime = {
            0: {'optimizer': 'SGD', 'lr': 1e-1,
                'weight_decay': 1e-4, 'momentum': 0.9},
            30: {'lr': 1e-2},
            60: {'lr': 1e-3, 'weight_decay': 0},
            90: {'lr': 1e-4}
        }


class ResNet_cifar10(ResNet):

    def __init__(self, num_classes=10,
                 block=BasicBlock, depth=18):
        super(ResNet_cifar10, self).__init__()
        self.inflate = 5
        self.inplanes = 16*self.inflate
        n = int((depth - 2) / 6)
        self.conv1 = BinarizeConv2d(3, 16*self.inflate, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.maxpool = lambda x: x
        self.bn1 = nn.BatchNorm2d(16*self.inflate)
        self.tanh1 = nn.Hardtanh(inplace=True)
        self.tanh2 = nn.Hardtanh(inplace=True)
        self.layer1 = self._make_layer(block, 16*self.inflate, n)
        self.layer2 = self._make_layer(block, 32*self.inflate, n, stride=2)
        self.layer3 = self._make_layer(block, 64*self.inflate, n, stride=2,do_bntan=False)
        self.layer4 = lambda x: x
        self.avgpool = nn.AvgPool2d(8)
        self.bn2 = nn.BatchNorm1d(64*self.inflate)
        self.bn3 = nn.BatchNorm1d(10)
        self.logsoftmax = nn.LogSoftmax()
        self.fc = BinarizeLinear(64*self.inflate, num_classes)

        init_model(self)
        #self.regime = {
        #    0: {'optimizer': 'SGD', 'lr': 1e-1,
        #        'weight_decay': 1e-4, 'momentum': 0.9},
        #    81: {'lr': 1e-4},
        #    122: {'lr': 1e-5, 'weight_decay': 0},
        #    164: {'lr': 1e-6}
        #}
        self.regime = {
            0: {'optimizer': 'Adam', 'lr': 5e-3},
            101: {'lr': 1e-3},
            142: {'lr': 5e-4},
            184: {'lr': 1e-4},
            220: {'lr': 1e-5}
        }


def resnet_binary(**kwargs):
    num_classes, depth, dataset = map(
        kwargs.get, ['num_classes', 'depth', 'dataset'])
    if dataset == 'imagenet':
        num_classes = num_classes or 1000
        depth = depth or 50
        if depth == 18:
            return ResNet_imagenet(num_classes=num_classes,
                                   block=BasicBlock, layers=[2, 2, 2, 2])
        if depth == 34:
            return ResNet_imagenet(num_classes=num_classes,
                                   block=BasicBlock, layers=[3, 4, 6, 3])
        if depth == 50:
            return ResNet_imagenet(num_classes=num_classes,
                                   block=Bottleneck, layers=[3, 4, 6, 3])
        if depth == 101:
            return ResNet_imagenet(num_classes=num_classes,
                                   block=Bottleneck, layers=[3, 4, 23, 3])
        if depth == 152:
            return ResNet_imagenet(num_classes=num_classes,
                                   block=Bottleneck, layers=[3, 8, 36, 3])

    elif dataset == 'cifar10':
        num_classes = num_classes or 10
        depth = depth or 18
        return ResNet_cifar10(num_classes=num_classes,
                              block=BasicBlock, depth=depth)



'''
class RealAlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(RealAlexNet, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96, eps=1e-4, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            RealConv2d(96, 256, kernel_size=5, stride=1, padding=2, groups=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            RealConv2d(256, 384, kernel_size=3, stride=1, padding=1),
            RealConv2d(384, 384, kernel_size=3, stride=1, padding=1, groups=1),
            RealConv2d(384, 256, kernel_size=3, stride=1, padding=1, groups=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            RealConv2d(256 * 6 * 6, 4096, Linear=True),
            RealConv2d(4096, 4096, dropout=0.5, Linear=True),
            nn.BatchNorm1d(4096, eps=1e-3, momentum=0.1, affine=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


class AllBinAlexNet(nn.Module):
    def __init__(self):
        super(AllBinNet, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            BinConv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=2),
            BinConv2d(96, 256, kernel_size=5, stride=1, padding=2, groups=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            BinConv2d(256, 384, kernel_size=3, stride=1, padding=1),
            BinConv2d(384, 384, kernel_size=3, stride=1, padding=1, groups=1),
            BinConv2d(384, 256, kernel_size=3, stride=1, padding=1, groups=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            BinConv2d(256 * 6 * 6, 4096, Linear=True),
            BinConv2d(4096, 4096, dropout=0.5, Linear=True),
            BinConv2d(4096, num_classes, dropout=0.5, Linear=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


'''

def resnet(model_name, pretrained=None, **kwargs):
    """AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    assert model_name in ['ResNet_imagenet','ResNet_imagenet',\
    'AllBinAlexNet'], 'No such model!'
    
    if model_name == 'ResNet_imagenet':
        model = resnet_binary(**kwargs)
    elif model_name == 'ResNet_cifar10':
        model = ResNet_cifar10(**kwargs)
    #elif model_name == 'AllBinResNet':
        #model = AllBinAlexNet(**kwargs)

    if pretrained is not None:
        pretrained_model = torch.load(pretrained)
        model.load_state_dict(pretrained_model['state_dict'])
    return model
