import math

import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _pair
from functions import deform_conv_function
import torch.nn.functional as F
from .modulated_dcn import *

class CNN(nn.Module):
    def __init__(self,inC,num_group,dataset):
        super(CNN, self).__init__()
        
        # conv1
        self.conv1 = nn.Conv2d(inC, 8, 3, stride=1,padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)

        # conv2
        self.conv2 = nn.Conv2d(8, 20, 3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(20)

        # # conv3
        # self.conv3 = nn.Conv2d(64, 40, 3, padding=1, stri de=1)
        # self.bn3 = nn.BatchNorm2d(40)


        # dconv1
        self.conv3 = nn.Conv2d(20, 40 , kernel_size=(3,3), padding= (1,1), stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(40)

        # dconv2
        self.conv4 = nn.Conv2d(40, 64, 3, padding=1, stride=1)
        self.bn4 = nn.BatchNorm2d(64)

        # # dconv3
        # self.dconv_3 = DeformConv(120, 120, (3,3),stride=1,padding=1, num_deformable_groups=num_group)
        # self.conv_5 = nn.Conv2d(120, num_group * 18, 3, padding=1, stride=1)
        # self.bn5 = nn.BatchNorm2d(120)

        # out
        self.fc = nn.Linear(256, 10)


    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))

        x = self.bn3(F.relu(self.conv3(x)))        
        x = self.bn4(F.relu(self.conv4(x)))

        em = x.clone()  
        x = F.avg_pool2d(x,kernel_size=[x.size(2)//2,x.size(3)//2])
        x = x.view((x.size(0),-1))
        x = self.fc(x)
        x = F.softmax(x)

        return x,em

    def parameters(self):
        return filter(lambda p: p.requires_grad, super(CNN, self).parameters())


class DCNv2(nn.Module):
    def __init__(self,inC,num_group,dataset):
        super(DCNv2, self).__init__()
        
        # conv1
        self.conv1 = nn.Conv2d(inC, 8, 3, stride=1,padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)

        # conv2
        self.conv2 = nn.Conv2d(8, 20, 3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(20)

        # # conv3
        # self.conv3 = nn.Conv2d(64, 40, 3, padding=1, stri de=1)
        # self.bn3 = nn.BatchNorm2d(40)


        # dconv1
        self.dconv_1 = ModulatedDeformConvPack(20, 40, (3,3),stride=1,padding=1, deformable_groups=num_group)
        self.bn3 = nn.BatchNorm2d(40)

        # dconv2
        self.dconv_2 = ModulatedDeformConvPack(40, 64, (3,3),stride=1,padding=1, deformable_groups=num_group)
        self.bn4 = nn.BatchNorm2d(64)

        # # dconv3
        # self.dconv_3 = DeformConv(120, 120, (3,3),stride=1,padding=1, num_deformable_groups=num_group)
        # self.conv_5 = nn.Conv2d(120, num_group * 18, 3, padding=1, stride=1)
        # self.bn5 = nn.BatchNorm2d(120)

        # out
        self.fc = nn.Linear(256, 10)


    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))

        x = self.bn3(F.relu(self.dconv_1(x)))        
        x = self.bn4(F.relu(self.dconv_2(x)))
        
        em = x.clone()
        x = F.avg_pool2d(x,kernel_size=[x.size(2)//2,x.size(3)//2])
        x = x.view((x.size(0),-1))
        x = self.fc(x)
        x = F.softmax(x)

        return x,em

    def parameters(self):
        return filter(lambda p: p.requires_grad, super(DCNv2, self).parameters())

# MNIST
class DCN(nn.Module):
    def __init__(self,inC,num_group,dataset):
        super(DCN, self).__init__()
        
        # conv1
        self.conv1 = nn.Conv2d(inC, 8, 3, stride=1,padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)

        # conv2
        self.conv2 = nn.Conv2d(8, 20, 3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(20)

        # # conv3
        # self.conv3 = nn.Conv2d(64, 40, 3, padding=1, stri de=1)
        # self.bn3 = nn.BatchNorm2d(40)


        # dconv1
        self.dconv_1 = DeformConv(20, 40, (3,3),stride=1,padding=1, num_deformable_groups=num_group)
        self.conv_3 = nn.Conv2d(20, num_group * 18 , kernel_size=(3,3), padding= (1,1), stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(40)

        # dconv2
        self.dconv_2 = DeformConv(40, 64, (3,3),stride=1,padding=1, num_deformable_groups=num_group)
        self.conv_4 = nn.Conv2d(40, num_group * 18, 3, padding=1, stride=1)
        self.bn4 = nn.BatchNorm2d(64)

        # # dconv3
        # self.dconv_3 = DeformConv(120, 120, (3,3),stride=1,padding=1, num_deformable_groups=num_group)
        # self.conv_5 = nn.Conv2d(120, num_group * 18, 3, padding=1, stride=1)
        # self.bn5 = nn.BatchNorm2d(120)

        # out
        self.fc = nn.Linear(256, 10)


    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))

        offset1 = self.conv_3(x)
        x = self.bn3(F.relu(self.dconv_1(x,offset1)))
        
        offset2 = self.conv_4(x)
        x = self.bn4(F.relu(self.dconv_2(x,offset2)))

        # offset3 = self.conv_5(x)
        # x = self.bn5(F.relu(self.dconv_3(x,offset3)))        
        em = x.clone() 
        x = F.avg_pool2d(x,kernel_size=[x.size(2)//2,x.size(3)//2])
        # print("Layer 4 -"+str(x.shape))
        x = x.view((x.size(0),-1))
        x = self.fc(x)
        x = F.softmax(x)
        # print("Output Layer  -"+str(x.shape))

        return x,em

    def parameters(self):
        return filter(lambda p: p.requires_grad, super(DCN, self).parameters())


class DeformConv(Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 num_deformable_groups=1):
        super(DeformConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.num_deformable_groups = num_deformable_groups

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, *self.kernel_size))

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, offset):
        return deform_conv_function(input, offset, self.weight, self.stride,
                             self.padding, self.dilation,
                             self.num_deformable_groups)
