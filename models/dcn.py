import sys,os

sys.path.insert(0,os.path.abspath(os.path.join('..')))

from deform_cnn.modules.deform_conv import *
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, rate=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=rate, padding=rate, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.rate = rate

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, nInputChannels, block, layers, os=16, pretrained=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        if os == 16:
            strides = [1, 2, 2, 1]
            rates = [1, 1, 1, 2]
            blocks = [1, 2, 4]
        elif os == 8:
            strides = [1, 2, 1, 1]
            rates = [1, 1, 2, 2]
            blocks = [1, 2, 1]
        else:
            raise NotImplementedError

        # Modules
        self.conv1 = nn.Conv2d(nInputChannels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # layers = [3, 4, 23, 3]
        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], rate=rates[0])#64， 3
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], rate=rates[1])#128 4
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], rate=rates[2])#256 23
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], rate=rates[3])

        self.conv2 = nn.Conv2d(2052, 19, kernel_size=3, stride=1, padding=1, bias=False)
        self.upsample = nn.Upsample(scale_factor=16,mode='bilinear')
        
        self._init_weight()

        if pretrained:
            self._load_pretrained_model()

    def _make_layer(self, block, planes, blocks, stride=1, rate=1):
        """
        Input:
            block class: uninitialized bottleneck class
            planes: number of output layers
            blocks: number of block in the layer
        Output:

        """

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, rate, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks=[1,2,4], stride=1, rate=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, rate=blocks[0]*rate, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1, rate=blocks[i]*rate))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x, low_level_feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


def ResNet101(nInputChannels=3, os=16, pretrained=False):
    model = ResNet(nInputChannels, Bottleneck, [3, 4, 23, 3], os, pretrained=pretrained)
    return model

#layers.append(block(self.inplanes, planes, stride=1, rate=blocks[i]*rate))

class deform_bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, rate=1, downsample=None):
        super(deform_bottleneck, self).__init__()
        num_class = 19
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # deformable conv
        self.conv2 = nn.Conv2d(planes, 18*num_class, kernel_size=3, stride=stride, dilation=rate, padding=rate, bias=False)
        self.dconv = DeformConv(planes, planes, kernel_size=3, stride=stride, dilation=rate, padding=rate,num_deformable_groups=num_class)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.rate = rate

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        offset = self.conv2(out)


        out = self.dconv(out, offset)
        out = self.relu(out)
        out = self.bn2(out)


        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class deform_ResNet(nn.Module):
    def __init__(self, nInputChannels, block,deform_block, layers, os=16, pretrained=False):
        self.inplanes = 64
        super(deform_ResNet, self).__init__()
        if os == 16:
            strides = [1, 2, 2, 1]
            rates = [1, 1, 1, 2]
            blocks = [1, 2, 4]
        elif os == 8:
            strides = [1, 2, 1, 1]
            rates = [1, 1, 2, 2]
            blocks = [1, 2, 1]
        else:
            raise NotImplementedError

        # Modules
        self.conv1 = nn.Conv2d(nInputChannels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1 = nn.DataParallel(self.conv1)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn1 = nn.DataParallel(self.bn1)

        self.relu = nn.ReLU(inplace=True)
        self.relu = nn.DataParallel(self.relu)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.DataParallel(self.maxpool)
        # layers = [3, 4, 23, 3]
        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], rate=rates[0])#64， 3
        self.layer1 = nn.DataParallel(self.layer1)

        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], rate=rates[1])#128 4
        self.layer2 = nn.DataParallel(self.layer2)
        
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], rate=rates[2])#256 23
        self.layer3 = nn.DataParallel(self.layer3)

        self.layer4 = self._make_deform_layer(deform_block, 513, blocks=blocks, stride=strides[3], rate=rates[3])
        self.layer4 = nn.DataParallel(self.layer4)


        self.conv2 = nn.Conv2d(2052, 19, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.DataParallel(self.conv2)

        self.upsample = nn.Upsample(scale_factor=16,mode='bilinear')
        self.upsample = nn.DataParallel(self.upsample)

        self._init_weight()

        if pretrained:
            self._load_pretrained_model()

    def _make_layer(self, block, planes, blocks, stride=1, rate=1):
        """
        Input:
            block class: uninitialized bottleneck class
            planes: number of output layers
            blocks: number of block in the layer
        Output:
            resblock consisting of 'block' modules 
        """

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, rate, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_deform_layer(self, block, planes, blocks=[1,2,4], stride=1, rate=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, rate=blocks[0]*rate, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1, rate=blocks[i]*rate))

        return nn.Sequential(*layers)

    def forward(self, input):
        
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        
        low_level_feat = x
        x = self.layer2(x)
        
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv2(x)
        x = self.upsample(x)

        return x, low_level_feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def deform_ResNet101(nInputChannels=3, os=16, pretrained=False):
    model = deform_ResNet(nInputChannels, Bottleneck,deform_bottleneck, [3, 4, 23, 3], os, pretrained=pretrained)
    return model
