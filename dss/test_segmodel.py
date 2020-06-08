from torch.autograd import Variable
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from Dconv import *
import sys,os
sys.path.insert(0,os.path.abspath(os.path.join('..')))
print(sys.path)
from utilities import dataset,funcs,equiv_funcs
from models import dss
import torch


def main(model_dir,n_test=10):

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Define variables 
    dataset_name = "MNIST"
    testing_size = 10000
    batch_size = 400
    ls_list = []
    acc_list = []
    equiv_list = []
    sum_ = 0
    base = 1.5
    io_scale = [4,4]

    # Load Model 
    model = dss.DSS_cityscape(base,io_scale,16)
    model = torch.load('./models/'+str(model_dir))
    print(model)
    model = model.to(device)
    params = model.parameters()
    print("Number of parameters = ",sum([np.prod(p.size()) for p in params]))

    downscale_kern = BesselConv2d(base,n_scales=4)
    testloader = dataset.get_gen('cityscapes','test',batch_size, 1, n_test, downscale_kern=downscale_kern,resize=["resize",0.5])
    testloader2 = dataset.get_gen('cityscapes','test',batch_size, 1, n_test,resize=["resize",0.5])

    for i in range(n_test):
        ytrue = testloader[i][1]
        ypred, _  = model(testloader[i][0])
        print("mIOU of image %d ="%(i),funcs.miou(ypred,ytrue))

        _,ypred = funcs.to_seg(ypred)
        plt.figure(i)
        plt.subplot(311)
        plt.imshow(testloader2[i][0].transpose(1,2,0))

        plt.subplot(312)
        plt.imshow(ytrue.cpu().detach().numpy())

        plt.subplot(313)
        plt.imshow(ypred.squeeze(0).cpu().detach().numpy())
        plt.savefig("./test/test_%d"%(i))
        plt.close()

if __name__ == "__main__":
    model = 'dss_cityscape_seg_1.th'
    main(model)
