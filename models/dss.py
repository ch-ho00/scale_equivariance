from dss.Dconv import *
# from 

class DSS_cityscape(nn.Module):
    '''
    default parameter configuration from DSS paper 
    [k,S,N] = [2,4,16] S-resNet, multiscale interaction
    [k,S,N] = [kernel scale dim , num scales, number of channel] 
    '''  
    def __init__(self,base, io_scales=[4,4],n=16):
        super(DSS_cityscape,self).__init__()
        self.res_option = 'A'
        self.use_dropout = False  

        self.res1 = self._make_layer(2,3, n, base,io_scales,0)
        self.avg_pool1 = nn.AvgPool2d((2,2),stride=2)
        self.res2 = self._make_layer(2,n, 2*n, base,io_scales,0)
        self.avg_pool2 = nn.AvgPool2d((2,2),stride=2)
        self.res3 = self._make_layer(1,2*n, 4*n, base,io_scales,0)
        self.res4 = self._make_layer(2,4*n, 4*n, base,io_scales,0)
        self.avg_pool3 = nn.AvgPool2d((2,2),stride=2)
        self.res5 = self._make_layer(1,4*n, 8*n, base,io_scales,0)
        self.res6 = self._make_layer(1,8*n, 8*n, base,io_scales,0)
        self.res7 = self._make_layer(1,8*n, 8*n, base,io_scales,0)
        self.res8 = self._make_layer(2,8*n, 8*n, base,io_scales,0)
        
        self.res_skip = self._make_layer(2,8*n, 8*n, base,io_scales,skip=True)
        
        self.conv2d = nn.Conv2d(8*n,19, kernel_size=(1,1,1))
        self.upsample = nn.Upsample(scale_factor=8,mode='bilinear')

    def _make_layer(self, layer_count, channels_in,channels, base,io_scales,skip):
        return nn.Sequential(
            ResBlock(channels, channels_in, base=1.5, io_scales=io_scales, skip=skip, res_option=self.res_option, use_dropout=self.use_dropout),
            *[ResBlock(channels,None,base,io_scales,skip=skip) for _ in range(layer_count-1)])

    def forward(self, x):
        # sequence 1 
        l = []
        x= self.res1(x)
        with torch.no_grad():
            for i in range(x.size(2)): 
                l.append(self.avg_pool1(x[:,:,i,:,:]))
            x = torch.stack(l).permute(1,2,0,3,4)
        # sequence 2
        l = []

        x= self.res2(x)
        print(x.shape)
        with torch.no_grad():
            for i in range(x.size(2)): 
                l.append(self.avg_pool2(x[:,:,i,:,:]))
            print(x.shape)
            x = torch.stack(l).permute(1,2,0,3,4)
        # sequence 3
        print(x.shape)

        x= self.res3(x)
        x= self.res4(x)
        print(x.shape)

        with torch.no_grad():
            for i in range(x.size(2)):
                l.append(self.avg_pool2(x[:,:,i,:,:]))
            print(self.avg_pool2(x[:,:,0,:,:]))
            x = torch.stack(l).permute(1,2,0,3,4)
        print(x.shape)
        # sequence 4        
        x= self.res5(x)
        x= self.res6(x)
        x= self.res7(x)
        x= self.res8(x)
        print(x.shape)
        x = self.res_skip(x)        
        x = self._scale_pool(x)
        x = x.squeeze(0)
        embedding = x.clone()
        x = self.conv(x)
        x = self.upsample(x)

        return x, embedding
    

    def _scale_pool(self,x):

        strength, _ = torch.max(x,2)
        return F.relu(strength)


    def parameters(self):
        return filter(lambda p: p.requires_grad, super(DSS_cityscape, self).parameters())

# for MNIST 
class DSS_res(nn.Module):
    def __init__(self,base, io_scales,n=8, final = 8):
        super(DSS_res,self).__init__()

        self.res_option = 'A'
        self.use_dropout = False 

        self.res1 = self._make_layer(7,1,n, base,io_scales,1)
        self.avg_pool1 =  nn.AvgPool2d((2,2),stride=2)
        self.res2 = self._make_layer(7,n, 2*n, base,io_scales,1)
        self.avg_pool2= nn.AvgPool2d((2,2),stride=2)
        self.res3 = self._make_layer(7,2*n, 2*n, base,io_scales,1)
        self.res4 = self._make_layer(7,2*n, 4*n, base,io_scales,1)
        self.conv = nn.Conv2d(2*n,final, kernel_size=(1,1))
        self.upsample = nn.Upsample(scale_factor=4)
        self.fc = nn.Linear(final * 49,10)
    def forward(self,x):
        # resblock1 
        l = []
        x= self.res1(x)
        for i in range(x.size(2)): 
            l.append(self.avg_pool1(x[:,:,i,:,:]))
        x = torch.stack(l).permute(1,2,0,3,4)
        # print("1111111",x.shape)
        # resblock2
        l = []
        x= self.res2(x)
        for i in range(x.size(2)): 
            l.append(self.avg_pool2(x[:,:,i,:,:]))
        x = torch.stack(l).permute(1,2,0,3,4)
        x = self._scale_pool(x)
        embedding = x.clone()
        x = self.conv(x)
        
        x = x.view([x.shape[0],-1])
        x = self.fc(x)
        return x, embedding

    def _make_layer(self, layer_count, channels_in,channels, base,io_scales,skip):
        return nn.Sequential(
            ResBlock(channels, channels_in, base=1.5, io_scales=io_scales, skip=skip, res_option=self.res_option, use_dropout=self.use_dropout),
            *[ResBlock(channels,None,base,io_scales,skip=skip) for _ in range(layer_count-1)])

    def _scale_pool(self,x):

        strength, _ = torch.max(x,2)
        return F.relu(strength)

    def parameters(self):
        return filter(lambda p: p.requires_grad, super(DSS_res, self).parameters())

# for MNIST 
class DSS_plain(nn.Module):
    def __init__(self,base, io_scales,n=8, final = 8):
        super(DSS_plain,self).__init__()

        self.res_option = 'A'
        self.use_dropout = False 

        self.dconv1 = Dconv2d(1,n,[1,3,3], base,io_scales,1)
        self.dconv2 = Dconv2d(n, n,[3,3,3], base,io_scales,1)
        self.avg_pool1 =  nn.AvgPool2d((2,2),stride=2)
        
        self.dconv3 = Dconv2d(n, 2*n,[3,3,3], base,io_scales,1)
        self.dconv4 = Dconv2d(2*n, 2*n,[3,3,3], base,io_scales,1)
        self.avg_pool2= nn.AvgPool2d((2,2),stride=2)
        
        self.upsample = nn.Upsample(scale_factor=4)
#        self.conv = nn.Conv2d(2*n,final,kernel_size=1,stride=1)
        self.fc = nn.Linear(2*n*49,10)
    def forward(self,x):
        l = []
        x= self.dconv1(x)
        x= self.dconv2(x)
        for i in range(x.size(2)): 
            l.append(self.avg_pool1(x[:,:,i,:,:]))
        x = torch.stack(l).permute(1,2,0,3,4)

        x= self.dconv3(x)
        x= self.dconv4(x)
        l = []
        for i in range(x.size(2)): 
            l.append(self.avg_pool2(x[:,:,i,:,:]))
        x = torch.stack(l).permute(1,2,0,3,4)
        x = self._scale_pool(x)
        embedding = x.clone()
        #x = self.conv(x)
        x = x.view([x.shape[0],-1])
        x = self.fc(x)
        return x, embedding


    def _scale_pool(self,x):

        strength, _ = torch.max(x,2)
        return F.relu(strength)

    def parameters(self):
        return filter(lambda p: p.requires_grad, super(DSS_plain, self).parameters())

class DSS_plain_2(nn.Module):
    def __init__(self,base, io_scales,n=8, interaction=1):
        super(DSS_plain_2,self).__init__()

        self.scale = io_scales[1]

        self.res_option = 'A'
        self.use_dropout = False 

        self.dconv1 = Dconv2d_2(1,n,[interaction,3,3], base,io_scales,1)
        self.dconv1 = nn.DataParallel(self.dconv1)

        self.dconv2 = Dconv2d_2(n, n,[interaction,3,3], base,io_scales,1)
        self.dconv2 = nn.DataParallel(self.dconv2)
        self.max_pool1 =  nn.MaxPool2d((2,2),stride=2)
        
        #self.dconv3 = Dconv2d_2(n, 2*n,[3,3,3], base,io_scales,1)
        #self.dconv3 = nn.DataParallel(self.dconv3)
        #self.avg_pool2= nn.AvgPool2d((2,2),stride=2)
        
        self.bn1 = nn.BatchNorm2d(n)
        self.bn1 = nn.DataParallel(self.bn1)
        
        fin = 14//int(base**(io_scales[1]-1))
        self.fc = nn.Linear(n*fin*fin,10)
        self.fc = nn.DataParallel(self.fc)
    def forward(self,x):
        l = []
        x= self.dconv1(x)
        x= self.dconv2(x)
        for i in range(self.scale): 
            x[i] = self.bn1(F.relu(self.max_pool1(x[i])))
        #x= self.dconv3(x)
        #l = []
        #for i in range(x.size(2)): 
        #     l.append(self.avg_pool2(x[:,:,i,:,:]))
        # x = torch.stack(l).permute(1,2,0,3,4)
        embedding = [x[i].clone() for i in x]
        x = self._dim_match(x)
        x = self._scale_pool(x)
        #x = self.conv(x)
        x = x.view([x.shape[0],-1])
        
        x = self.fc(x)
        return x, embedding

    def _dim_match(self,x):
        b,c, h,w = x[self.scale-1].shape
        out = torch.zeros(b,c,self.scale, h, w)
        out[:,:,self.scale-1,:,:] = x[self.scale-1]
        for i in range(self.scale-1):
            h1, w1 = x[i].shape[-2:]
            if h1//h == h1/h and w1//w == w1/w:
                padding = 0
            else:
                padding =0 
            out[:,:,i,:,:] = nn.functional.max_pool2d(input=x[i],kernel_size=(h1//h,w1//w),stride=(h1//h,w1//w))
        return out
        

    def _scale_pool(self,x):
        
        strength, _ = torch.max(x,2)
        return F.relu(strength)

    def parameters(self):
        return filter(lambda p: p.requires_grad, super(DSS_plain_2, self).parameters())


class DSS_2_cifar(nn.Module):
    def __init__(self,base, io_scales,n=8, interaction=1):
        super(DSS_2_cifar,self).__init__()

        self.scale = io_scales[1]

        self.res_option = 'A'
        self.use_dropout = False

        self.dconv1 = Dconv2d_2(3,n,[interaction,3,3], base,io_scales,1)
        self.dconv1 = nn.DataParallel(self.dconv1)

        self.dconv2 = Dconv2d_2(n, n,[interaction,3,3], base,io_scales,1)
        self.dconv2 = nn.DataParallel(self.dconv2)
        self.max_pool1 =  nn.MaxPool2d((2,2),stride=2)

        self.dconv3 = Dconv2d_2(n, 2*n,[1,3,3], base,io_scales,1)
        self.dconv3 = nn.DataParallel(self.dconv3)
        #self.avg_pool2= nn.AvgPool2d((2,2),stride=2)

        self.bn1 = nn.BatchNorm2d(n)
        self.bn1 = nn.DataParallel(self.bn1)

        self.bn2 = nn.BatchNorm2d(2*n)
        self.bn2 = nn.DataParallel(self.bn2)

        fin = 16//int(base**(io_scales[1]-1))
        self.fc = nn.Linear(2*n*fin*fin,10)
        self.fc = nn.DataParallel(self.fc)
    def forward(self,x):
        l = []
        x= self.dconv1(x)
        x= self.dconv2(x)
        for i in range(self.scale):
            x[i] = self.bn1(F.relu(self.max_pool1(x[i])))

        x= self.dconv3(x)

        for i in range(self.scale):
            x[i] = self.bn2(x[i])

        x = self._dim_match(x)
        x = self._scale_pool(x)
        embedding = x.clone()
        #x = self.conv(x)
        x = x.view([x.shape[0],-1])

        x = self.fc(x)
        return x, embedding

    def _dim_match(self,x):

        b,c, h,w = x[self.scale-1].shape
        out = torch.zeros(b,c,self.scale, h, w)
        out[:,:,self.scale-1,:,:] = x[self.scale-1]
        for i in range(self.scale-1):
            h1, w1 = x[i].shape[-2:]
            out[:,:,i,:,:] = nn.functional.max_pool2d(input=x[i],kernel_size=(h1//h,w1//w),stride=(h1//h,w1//w))
        return out

    def _scale_pool(self,x):

        strength, _ = torch.max(x,2)
        return F.relu(strength)

    def parameters(self):
        return filter(lambda p: p.requires_grad, super(DSS_2_cifar, self).parameters())


class DSS_cifar(nn.Module):
    def __init__(self,base, io_scales,n=8, interaction=1):
        super(DSS_cifar,self).__init__()

        self.res_option = 'A'
        self.use_dropout = False

        self.dconv1 = Dconv2d(3,n,[interaction,3,3], base,io_scales,1)
        self.dconv1 = nn.DataParallel(self.dconv1)

        #self.dconv2 = Dconv2d(n, n,[interaction,3,3], base,io_scales,1)
        #self.dconv2 = nn.DataParallel(self.dconv2)

        self.max_pool1 =  nn.MaxPool2d((2,2),stride=2)
        self.max_pool1  = nn.DataParallel(self.max_pool1)

        self.bn1 = nn.BatchNorm2d(io_scales[1])
        self.bn1 = nn.DataParallel(self.bn1)


        self.dconv3 = Dconv2d(n, 2*n,[interaction,3,3], base,io_scales,1)
        self.dconv3 = nn.DataParallel(self.dconv3)

        #self.dconv4 = Dconv2d(2*n, 2*n,[interaction,3,3], base,io_scales,1)
        #self.dconv4 = nn.DataParallel(self.dconv4)

        self.dconv5 = Dconv2d(2*n, 4*n,[interaction,3,3], base,io_scales,1)
        self.dconv5 = nn.DataParallel(self.dconv5)

        #self.dconv6 = Dconv2d(4*n, 4*n,[interaction,3,3], base,io_scales,1)
        #self.dconv6 = nn.DataParallel(self.dconv6)


        self.max_pool2 = nn.MaxPool2d((2,2),stride=2)
        self.max_pool2 = nn.DataParallel(self.max_pool2)

        #self.max_pool3 =  nn.MaxPool2d((2,2),stride=2)
        #self.max_pool3  = nn.DataParallel(self.max_pool1)

        #self.max_pool4 =  nn.MaxPool2d((2,2),stride=2)
        #self.max_pool4  = nn.DataParallel(self.max_pool1)

        #self.max_pool5 =  nn.MaxPool2d((2,2),stride=2)
        #self.max_pool5  = nn.DataParallel(self.max_pool1)


        self.bn2 = nn.BatchNorm2d(io_scales[1])
        self.bn2 = nn.DataParallel(self.bn2)


        self.bn3 = nn.BatchNorm2d(io_scales[1])
        self.bn3 = nn.DataParallel(self.bn3)

        self.bn4 = nn.BatchNorm2d(io_scales[1])
        self.bn4 = nn.DataParallel(self.bn4)


        self.bn5 = nn.BatchNorm2d(io_scales[1])
        self.bn5 = nn.DataParallel(self.bn5)

        self.fc = nn.Linear(n*256,10)
        self.fc = nn.DataParallel(self.fc)
    def forward(self,x):
        #layer 1,2
        x= self.dconv1(x)
        x= F.relu(x)

        x = self._bn(x,1)

        #layer 3
        x= F.relu(self.dconv3(x))
        x = self._bn(x,2)

        #x= self.dconv4(x)
        l = []
        for i in range(x.size(1)):
            l.append(F.relu(self.max_pool1(x[:,i,:,:,:])))
        x = torch.stack(l).permute(1,0,2,3,4)
        x = self._bn(x,3)

        #layer 5
        x= self.dconv5(x)
        l = []
        for i in range(x.size(1)):
            l.append(F.relu(self.max_pool2(x[:,i,:,:,:])))
        x = torch.stack(l).permute(1,0,2,3,4)
        x = self._bn(x,4)

        #layer 6
        x = F.relu(x)
        x = self._bn(x,5)

        x = self._scale_pool(x)
        embedding = x.clone()
        #x = self.conv(x)
        x = x.view([x.shape[0],-1])
        x = self.fc(x)
        return x, embedding
    def _bn(self,x,i):
        l = []
        if i == 1:
            for i in range(x.size(1)):
                l.append(self.bn1(x[:,i,:,:,:]))
            x = torch.stack(l).permute(1,0,2,3,4)
        elif i ==2:
            for i in range(x.size(1)):
                l.append(self.bn2(x[:,i,:,:,:]))
            x = torch.stack(l).permute(1,0,2,3,4)
        elif i ==3:
            for i in range(x.size(1)):
                l.append(self.bn3(x[:,i,:,:,:]))
            x = torch.stack(l).permute(1,0,2,3,4)
        elif i ==4:
            for i in range(x.size(1)):
                l.append(self.bn4(x[:,i,:,:,:]))
            x = torch.stack(l).permute(1,0,2,3,4)
        elif i ==5:
            for i in range(x.size(1)):
                l.append(self.bn5(x[:,i,:,:,:]))
            x = torch.stack(l).permute(1,0,2,3,4)
        return x
    def _scale_pool(self,x):

        strength, _ = torch.max(x,2)
        return F.relu(strength)

    def parameters(self):
        return filter(lambda p: p.requires_grad, super(DSS_cifar, self).parameters())

