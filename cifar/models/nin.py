import torch.nn as nn
import torch
import torch.nn.functional as F

class BinActive(torch.autograd.Function):
    '''
    Binarize the input activations and calculate the mean across channel dimension.
    '''
    def forward(self, input):
        self.save_for_backward(input)
        size = input.size()
        mean = torch.mean(input.abs(), 1, keepdim=True)
        input = input.sign()
        return input, mean

    def backward(self, grad_output, grad_output_mean):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input

class BinConv2d(nn.Module):
    '''
    Conv layer with vinarized weights and input
    '''
    def __init__(self, input_channels, output_channels,
            kernel_size=-1, stride=-1, padding=-1, dropout=0):
        super(BinConv2d, self).__init__()
        self.layer_type = 'BinConv2d'
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout_ratio = dropout

        self.bn = nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True)
        if dropout!=0:
            self.dropout = nn.Dropout(dropout)
        self.conv = nn.Conv2d(input_channels, output_channels,
                kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.bn(x)
        x, mean = BinActive()(x)
        if self.dropout_ratio!=0:
            x = self.dropout(x)
        x = self.conv(x)
        x = self.relu(x)
        return x


class RealConv2d(nn.Module):
    '''
    Float conv layer with the same architecture with class::BinConv2d
    '''
    def __init__(self, input_channels, output_channels,
            kernel_size=-1, stride=-1, padding=-1, dropout=0):
        super(RealConv2d, self).__init__()
        self.layer_type = 'RealConv2d'
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout_ratio = dropout

        self.bn = nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True)
        if dropout!=0:
            self.dropout = nn.Dropout(dropout)
        self.conv = nn.Conv2d(input_channels, output_channels,
                kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.bn(x)
        #x, mean = BinActive()(x)
        if self.dropout_ratio!=0:
            x = self.dropout(x)
        x = self.conv(x)
        x = self.relu(x)
        return x



class Net(nn.Module):
    '''
    The original binarized XNOR-NIN model
    '''
    def __init__(self):
        super(Net, self).__init__()
        self.xnor = nn.Sequential(
                nn.Conv2d(3, 192, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=False),
                nn.ReLU(inplace=True),
                #BinConv2d(192, 160, kernel_size=1, stride=1, padding=0),
                BinConv2d(192, 96, kernel_size=1, stride=1, padding=0), # new by simon
                #BinConv2d(160,  96, kernel_size=1, stride=1, padding=0),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

                BinConv2d( 96, 192, kernel_size=5, stride=1, padding=2, dropout=0.5),
                #BinConv2d(192, 192, kernel_size=1, stride=1, padding=0),
                BinConv2d(192, 192, kernel_size=1, stride=1, padding=0),
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1),

                BinConv2d(192, 192, kernel_size=3, stride=1, padding=1, dropout=0.5),
                BinConv2d(192, 192, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=False),
                nn.Conv2d(192,  10, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(kernel_size=8, stride=1, padding=0),
                )

    def forward(self, x):
        x = self.xnor(x)
        x = x.view(x.size(0), 10)
        return x


####################################################
#
# model variants
#
####################################################


class Net_Cut(nn.Module):
    '''
    The 'narrower' XNOR-NIN model
    '''
    def __init__(self, cut_ratio=0.5):
        super(Net_Cut, self).__init__()
        self.xnor = nn.Sequential(
                nn.Conv2d(3, int(192*cut_ratio), kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(int(192*cut_ratio), eps=1e-4, momentum=0.1, affine=False),
                nn.ReLU(inplace=True),
                #BinConv2d(192, 160, kernel_size=1, stride=1, padding=0),
                BinConv2d(int(192*cut_ratio), int(96*cut_ratio), kernel_size=1, stride=1, padding=0), # new by simon
                #BinConv2d(160,  96, kernel_size=1, stride=1, padding=0),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

                BinConv2d(int(96*cut_ratio), int(192*cut_ratio), kernel_size=5, stride=1, padding=2, dropout=0.5),
                #BinConv2d(192, 192, kernel_size=1, stride=1, padding=0),
                BinConv2d(int(192*cut_ratio), int(192*cut_ratio), kernel_size=1, stride=1, padding=0),
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1),

                BinConv2d(int(192*cut_ratio), int(192*cut_ratio), kernel_size=3, stride=1, padding=1, dropout=0.5),
                BinConv2d(int(192*cut_ratio), int(192*cut_ratio), kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(int(192*cut_ratio), eps=1e-4, momentum=0.1, affine=False),
                nn.Conv2d(int(192*cut_ratio),  10, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(kernel_size=8, stride=1, padding=0),
                )

    def forward(self, x):
        x = self.xnor(x)
        x = x.view(x.size(0), 10)
        return x



class RealNet(nn.Module):
    '''
    The float NIN model
    '''
    def __init__(self):
        super(RealNet, self).__init__()
        self.xnor = nn.Sequential(
                nn.Conv2d(3, 192, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=False),
                nn.ReLU(inplace=True),
                #BinConv2d(192, 160, kernel_size=1, stride=1, padding=0),
                RealConv2d(192, 96, kernel_size=1, stride=1, padding=0), # new by simon
                #BinConv2d(160,  96, kernel_size=1, stride=1, padding=0),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

                RealConv2d( 96, 192, kernel_size=5, stride=1, padding=2, dropout=0.5),
                #BinConv2d(192, 192, kernel_size=1, stride=1, padding=0),
                RealConv2d(192, 192, kernel_size=1, stride=1, padding=0),
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1),

                RealConv2d(192, 192, kernel_size=3, stride=1, padding=1, dropout=0.5),
                RealConv2d(192, 192, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=False),
                nn.Conv2d(192,  10, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(kernel_size=8, stride=1, padding=0),
                )

    def forward(self, x):
        x = self.xnor(x)
        x = x.view(x.size(0), 10)
        return x


class AllBinNet(nn.Module):
    '''
    binarize the last and first layers of the original XNOR-NIN model
    '''
    def __init__(self):
        super(AllBinNet, self).__init__()
        self.xnor = nn.Sequential(
                #nn.Conv2d(3, 192, kernel_size=5, stride=1, padding=2),
                #nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=False),
                #nn.ReLU(inplace=True),
                BinConv2d(3, 192, kernel_size=5, stride=1, padding=2),
                #BinConv2d(192, 160, kernel_size=1, stride=1, padding=0),
                BinConv2d(192, 96, kernel_size=1, stride=1, padding=0), # new by simon
                #BinConv2d(160,  96, kernel_size=1, stride=1, padding=0),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

                BinConv2d( 96, 192, kernel_size=5, stride=1, padding=2, dropout=0.5),
                #BinConv2d(192, 192, kernel_size=1, stride=1, padding=0),
                BinConv2d(192, 192, kernel_size=1, stride=1, padding=0),
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1),

                BinConv2d(192, 192, kernel_size=3, stride=1, padding=1, dropout=0.5),
                BinConv2d(192, 192, kernel_size=1, stride=1, padding=0),
                #nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=False),
                #nn.Conv2d(192,  10, kernel_size=1, stride=1, padding=0),
                #nn.ReLU(inplace=True),
                BinConv2d(192,  10, kernel_size=1, stride=1, padding=0),
                nn.AvgPool2d(kernel_size=8, stride=1, padding=0),
                )

    def forward(self, x):
        x = self.xnor(x)
        x = x.view(x.size(0), 10)
        return x



class NotAllBinNet(nn.Module):
    '''
    Only binarize the last layer of the original XNOR-NIN model
    '''
    def __init__(self):
        super(NotAllBinNet, self).__init__()
        self.xnor = nn.Sequential(
                nn.Conv2d(3, 192, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=False),
                nn.ReLU(inplace=True),
                #BinConv2d(3, 192, kernel_size=5, stride=1, padding=2),
                #BinConv2d(192, 160, kernel_size=1, stride=1, padding=0),
                BinConv2d(192, 96, kernel_size=1, stride=1, padding=0), # new by simon
                #BinConv2d(160,  96, kernel_size=1, stride=1, padding=0),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

                BinConv2d( 96, 192, kernel_size=5, stride=1, padding=2, dropout=0.5),
                #BinConv2d(192, 192, kernel_size=1, stride=1, padding=0),
                BinConv2d(192, 192, kernel_size=1, stride=1, padding=0),
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1),

                BinConv2d(192, 192, kernel_size=3, stride=1, padding=1, dropout=0.5),
                BinConv2d(192, 192, kernel_size=1, stride=1, padding=0),
                #nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=False),
                #nn.Conv2d(192,  10, kernel_size=1, stride=1, padding=0),
                #nn.ReLU(inplace=True),
                BinConv2d(192,  10, kernel_size=1, stride=1, padding=0),
                nn.AvgPool2d(kernel_size=8, stride=1, padding=0),
                )

    def forward(self, x):
        x = self.xnor(x)
        x = x.view(x.size(0), 10)
        return x





class AllBinNet_Cut(nn.Module):
    def __init__(self, cut_ratio=0.5):
        super(AllBinNet_Cut, self).__init__()
        self.xnor = nn.Sequential(
                #nn.Conv2d(3, 192, kernel_size=5, stride=1, padding=2),
                #nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=False),
                #nn.ReLU(inplace=True),
                BinConv2d(3, int(192*cut_ratio), kernel_size=5, stride=1, padding=2),
                #BinConv2d(192, 160, kernel_size=1, stride=1, padding=0),
                BinConv2d(int(192*cut_ratio), int(96*cut_ratio), kernel_size=1, stride=1, padding=0, dropout=0.0), # new by simon
                #BinConv2d(160,  96, kernel_size=1, stride=1, padding=0),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

                BinConv2d( int(96*cut_ratio), int(192*cut_ratio), kernel_size=5, stride=1, padding=2, dropout=0.5),
                #BinConv2d(192, 192, kernel_size=1, stride=1, padding=0),
                BinConv2d(int(192*cut_ratio), int(192*cut_ratio), kernel_size=1, stride=1, padding=0, dropout=0.0),
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1),

                BinConv2d(int(192*cut_ratio), int(192*cut_ratio), kernel_size=3, stride=1, padding=1, dropout=0.5),
                BinConv2d(int(192*cut_ratio), int(192*cut_ratio), kernel_size=1, stride=1, padding=0, dropout=0.0),
                #nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=False),
                #nn.Conv2d(192,  10, kernel_size=1, stride=1, padding=0),
                #nn.ReLU(inplace=True),
                BinConv2d(int(192*cut_ratio),  10, kernel_size=1, stride=1, padding=0),
                nn.AvgPool2d(kernel_size=8, stride=1, padding=0),
                )

    def forward(self, x):
        x = self.xnor(x)
        x = x.view(x.size(0), 10)
        return x


class NotAllBinNet_Cut(nn.Module):
    def __init__(self, cut_ratio=0.5):
        super(NotAllBinNet_Cut, self).__init__()
        self.xnor = nn.Sequential(
                nn.Conv2d(3, int(192*cut_ratio), kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(int(192*cut_ratio), eps=1e-4, momentum=0.1, affine=False),
                nn.ReLU(inplace=True),
                #BinConv2d(3, int(192*cut_ratio), kernel_size=5, stride=1, padding=2),
                #BinConv2d(192, 160, kernel_size=1, stride=1, padding=0),
                BinConv2d(int(192*cut_ratio), int(96*cut_ratio), kernel_size=1, stride=1, padding=0, dropout=0.0), # new by simon
                #BinConv2d(160,  96, kernel_size=1, stride=1, padding=0),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

                BinConv2d( int(96*cut_ratio), int(192*cut_ratio), kernel_size=5, stride=1, padding=2, dropout=0.5),
                #BinConv2d(192, 192, kernel_size=1, stride=1, padding=0),
                BinConv2d(int(192*cut_ratio), int(192*cut_ratio), kernel_size=1, stride=1, padding=0, dropout=0.0),
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1),

                BinConv2d(int(192*cut_ratio), int(192*cut_ratio), kernel_size=3, stride=1, padding=1, dropout=0.5),
                BinConv2d(int(192*cut_ratio), int(192*cut_ratio), kernel_size=1, stride=1, padding=0, dropout=0.0),
                #nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=False),
                #nn.Conv2d(192,  10, kernel_size=1, stride=1, padding=0),
                #nn.ReLU(inplace=True),
                BinConv2d(int(192*cut_ratio),  10, kernel_size=1, stride=1, padding=0),
                nn.AvgPool2d(kernel_size=8, stride=1, padding=0),
                )

    def forward(self, x):
        x = self.xnor(x)
        x = x.view(x.size(0), 10)
        return x




class RealNet_Cut(nn.Module):
    def __init__(self, cut_ratio=0.5):
        super(RealNet_Cut, self).__init__()
        self.xnor = nn.Sequential(
                nn.Conv2d(3, int(192*cut_ratio), kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(int(192*cut_ratio), eps=1e-4, momentum=0.1, affine=False),
                nn.ReLU(inplace=True),
                #BinConv2d(192, 160, kernel_size=1, stride=1, padding=0),
                RealConv2d(int(192*cut_ratio), int(96*cut_ratio), kernel_size=1, stride=1, padding=0), # new by simon
                #BinConv2d(160,  96, kernel_size=1, stride=1, padding=0),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

                RealConv2d( int(96*cut_ratio), int(192*cut_ratio), kernel_size=5, stride=1, padding=2, dropout=0.5),
                #BinConv2d(192, 192, kernel_size=1, stride=1, padding=0),
                RealConv2d(int(192*cut_ratio), int(192*cut_ratio), kernel_size=1, stride=1, padding=0),
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1),

                RealConv2d(int(192*cut_ratio), int(192*cut_ratio), kernel_size=3, stride=1, padding=1, dropout=0.5),
                RealConv2d(int(192*cut_ratio), int(192*cut_ratio), kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(int(192*cut_ratio), eps=1e-4, momentum=0.1, affine=False),
                nn.Conv2d(int(192*cut_ratio),  10, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(kernel_size=8, stride=1, padding=0),
                )

    def forward(self, x):
        x = self.xnor(x)
        x = x.view(x.size(0), 10)
        return x
