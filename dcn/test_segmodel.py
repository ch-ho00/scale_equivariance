
from torch.autograd import Variable
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from modules import deform_conv
import sys,os
sys.path.insert(0,os.path.abspath(os.path.join('..')))
print(sys.path)
from utilities import dataset,funcs,equiv_funcs
from models import dcn
import torch
import numpy as np 

def main(model_dir,n_test=10):

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Define variables 
    testing_size = 10000
    batch_size = 2
    ls_list = []
    acc_list = []
    equiv_list = []
    sum_ = 0
    base = 1.5
    io_scale = [4,4]

    # Load Model 
    model = dcn.deform_ResNet101()
    model = torch.load('./models/'+str(model_dir), map_location=device)
    print(model)
    model = model.to(device)
    #params = model.parameters()
    #print("Number of parameters = ",sum([np.prod(p.size()) for p in params]))

    testloader = dataset.get_gen('cityscapes','test',batch_size, 1, n_test,resize=["resize",0.5])
    testloader2 = dataset.get_gen('cityscapes','test',batch_size, 1, n_test,resize=["resize",0.5])

    for i in range(n_test//2):
        ytrue = testloader[i*2:(i+1)*2][1]
        ypred, _  = model(torch.Tensor(testloader[i*2:(i+1)*2][0]))
        for j in range(2):
            print("mIOU of image %d ="%(j),funcs.miou(ypred[j],ytrue[j]))

            _,ypred[j] = funcs.to_seg(ypred[j])
            plt.figure(j)
            plt.subplot(311)
            plt.imshow(testloader2[i*64+j][0].transpose(1,2,0))

            plt.subplot(312)
            plt.imshow(ytrue[j].cpu().detach().numpy())

            plt.subplot(313)
            plt.imshow(ypred[j].squeeze(0).cpu().detach().numpy())
            plt.savefig("./test/test_%d"%(i))
            plt.close()

if __name__ == "__main__":
    model = 'dcn_resnet_cityscape_seg_0220.th'
    main(model) 
