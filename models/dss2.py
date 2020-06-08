from dss.Dconv import *
import torch
import sys
from dcn2 import functions
#import DCN
class deform_dconv(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size,io_scales,stride, padding,num_cls,bias=True):
        '''
        Input:
              in_channels ... 
              out_channels ...
              kernel_size ...
              io_scales ...
              stride ...
              padding ...
              num_group: number of classes
        '''
        super(deform_dconv,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Kernel sizes
        self.kernel_size = kernel_size
        self.io_scales = io_scales
        self.stride  = stride
        self.padding = padding
        weight_shape = (out_channels,in_channels,1,kernel_size,kernel_size)
        self.weight = Parameter(torch.Tensor(*weight_shape))
        self.num_cls = num_cls
        if bias == True:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None
        self.reset_parameters()

    def __repr__(self):
        return ('{name}({in_channels}->{out_channels}, {io_scales}, {kernel_size})' 
                .format(name=self.__class__.__name__, **self.__dict__))

    def reset_parameters(self):
        """
        # Custom Yu/Koltun-initialization
        stdv = 1e-2
        wsh = self.weights.size()
        self.weights.data.uniform_(-stdv, stdv)

        C = np.gcd(self.in_channels, self.out_channels)
        val = C / (self.out_channels)
        
        ci = self.kernel_size[0] // 2
        cj = self.kernel_size[1] // 2
        for b in range(self.out_channels):
            for a in range(self.in_channels):
                if np.floor(a*C/self.in_channels) == np.floor(b*C/self.out_channels):
                    self.weights.data[b,a,:,ci,cj] = val
                else:
                    pass
        """
        # Just your standard He initialization
        n = self.kernel_size**2 * self.in_channels
        self.weight.data.normal_(0, math.sqrt(2. / n))

        if self.bias is not None:
            self.bias.data.fill_(1)

    def forward(self,input,offset):
        '''
        Input
            Input = input feature map (B,C,S,H,W)
            offset = (B,C,S,H,W)
        Output
            output feature map = (B,C,S,H,W) 
        '''
        dilation = 1
        #print(self.weight.shape,input.shape, offset.shape,"!!!!!!!!!!!!!",flush=True)
        out = [functions.deform_conv_func.DeformConvFunction.apply(input[:,:,i,:,:].contiguous(),offset[:,:,i,:,:].contiguous(), self.weight[:,:,0,:,:].contiguous(),self.bias,self.stride,self.padding,dilation,1,self.num_cls,64) for i in range(self.io_scales)]
        return torch.stack(out).permute(1,2,0,3,4)  


class deform_dss(nn.Module):
    def __init__(self,base,io_scales,n,num_cls,kernel_size=3,stride=1,padding=1):

        super(deform_dss,self).__init__()
        self.conv1 = nn.Conv2d(1,n,kernel_size=kernel_size, stride=stride, padding=padding)
        #self.conv2 = nn.Conv2d(n, 2*n,kernel_size=kernel_size, stride=stride, padding=padding)

        self.dconv1 = deform_dconv(n,2*n,kernel_size,io_scales,stride, padding,num_cls)
        self.offset_conv1 = nn.Conv2d(n, 18*num_cls,kernel_size=kernel_size, stride=stride, padding=padding)

        self.dconv2 = deform_dconv(2*n,4*n,kernel_size,io_scales,stride, padding,num_cls) 
        self.offset_conv2 = nn.Conv2d(2*n, 18*num_cls,kernel_size=kernel_size, stride=stride, padding=padding)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2,stride=2)

        self.bn1 = nn.BatchNorm2d(n)
        self.bn2 = nn.BatchNorm2d(io_scales)
        self.bn3 = nn.BatchNorm2d(io_scales)
        self.bn4 = nn.BatchNorm2d(io_scales)

        self.fc = nn.Linear(4*n*49,10)
    def forward(self,x):
        l = []
        for i in range(x.size(2)):
            l.append(self.bn1(F.relu(self.conv1(x[:,:,i,:,:]))))
        x = torch.stack(l).permute(1,2,0,3,4)
        #l = []
        #for i in range(x.size(2)):
        #    l.append(self.avgpool1(self.bn2(F.relu(self.conv2(x[:,:,i,:,:])))))
        #x = torch.stack(l).permute(1,2,0,3,4)
        
        # offset1
        l = []
        for i in range(x.size(2)):
            l.append(self.offset_conv1(x[:,:,i,:,:]))
        offset1 = torch.stack(l).permute(1,2,0,3,4)
        # dconv1 
        x = self.dconv1(x,offset1)
        # bn
        l = []
        for i in range(x.size(1)):
            l.append(self.maxpool1(self.bn2(F.relu(x[:,i,:,:,:]))))
        x = torch.stack(l).permute(1,0,2,3,4)
        #offset2
        l = []
        for i in range(x.size(2)):
            l.append(self.offset_conv2(x[:,:,i,:,:]))
        offset2 = torch.stack(l).permute(1,2,0,3,4)
        x = self.dconv2(x,offset2)
        #bn
        l = []
        for i in range(x.size(1)):
            l.append(self.maxpool2(self.bn3(x[:,i,:,:,:])))
        x = torch.stack(l).permute(1,0,2,3,4)
        embedding = x.clone()
        x = self.scale_pool(x)
        
        x = x.view([x.size(0),-1])
        x = self.fc(x)
        
        return x, embedding
    def scale_pool(self,x):
        strength, _ = torch.max(x,2)
        return F.relu(strength)
    def parameters(self):
        return filter(lambda p: p.requires_grad, super(deform_dss, self).parameters())

class deform_dss_cifar(nn.Module):
    def __init__(self,base,io_scales,n,num_cls,kernel_size=3,stride=1,padding=1):

        super(deform_dss_cifar,self).__init__()
        self.conv1 = nn.Conv2d(3,n,kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv1 = nn.DataParallel(self.conv1)
        self.conv2 = nn.Conv2d(n, 2*n,kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.DataParallel(self.conv2)

        self.dconv1 = deform_dconv(2*n,2*n,kernel_size,io_scales,stride, padding,num_cls)
        self.dconv1 = nn.DataParallel(self.dconv1)

        self.offset_conv1 = nn.Conv2d(2*n, 18*num_cls,kernel_size=kernel_size, stride=stride, padding=padding)
        self.offset_conv1 = nn.DataParallel(self.offset_conv1)

        self.dconv2 = deform_dconv(2*n,4*n,kernel_size,io_scales,stride, padding,num_cls) 
        self.dconv2 = nn.DataParallel(self.dconv2)

        self.offset_conv2 = nn.Conv2d(2*n, 18*num_cls,kernel_size=kernel_size, stride=stride, padding=padding)
        self.offset_conv2 = nn.DataParallel(self.offset_conv2)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.maxpool1 = nn.DataParallel(self.maxpool1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.maxpool2 = nn.DataParallel(self.maxpool2)

        self.bn1 = nn.BatchNorm2d(n)
        self.bn1 = nn.DataParallel(self.bn1)

        self.bn2 = nn.BatchNorm2d(io_scales)
        self.bn2 = nn.DataParallel(self.bn2)

        self.bn3 = nn.BatchNorm2d(io_scales)
        self.bn3 = nn.DataParallel(self.bn3)

        self.bn4 = nn.BatchNorm2d(io_scales)
        self.bn4 = nn.DataParallel(self.bn4)


        self.fc = nn.Linear(4*n*64,10)
        self.fc = nn.DataParallel(self.fc)

    def forward(self,x):
        l = []
        for i in range(x.size(2)):
            l.append(self.bn1(F.relu(self.conv1(x[:,:,i,:,:]))))
        x = torch.stack(l).permute(1,2,0,3,4)

        l = []
        for i in range(x.size(2)):
            l.append(self.conv2(x[:,:,i,:,:]))
        x = torch.stack(l).permute(1,2,0,3,4)
        #offset 1
        l = []
        for i in range(x.size(2)):
            l.append(self.offset_conv1(x[:,:,i,:,:]))
        offset1 = torch.stack(l).permute(1,2,0,3,4)

        # dconv1 
        x = self.dconv1(x,offset1)
        # bn
        l = []
        for i in range(x.size(1)):
            l.append(self.maxpool1(self.bn2(F.relu(x[:,i,:,:,:]))))
        x = torch.stack(l).permute(1,0,2,3,4)

        #offset2
        l = []
        for i in range(x.size(2)):
            l.append(self.offset_conv2(x[:,:,i,:,:]))
        offset2 = torch.stack(l).permute(1,2,0,3,4)
        # dconv2
        x = self.dconv2(x,offset2)
        #bn
        l = []
        for i in range(x.size(1)):
            l.append(self.maxpool2(self.bn3(x[:,i,:,:,:])))
        
        x = torch.stack(l).permute(1,0,2,3,4)
        embedding = x.clone()
        x = self.scale_pool(x)
        
        x = x.view([x.size(0),-1])
        x = self.fc(x)
        
        return x, embedding
    def scale_pool(self,x):
        strength, _ = torch.max(x,2)
        return F.relu(strength)
    def parameters(self):
        return filter(lambda p: p.requires_grad, super(deform_dss_cifar, self).parameters())

