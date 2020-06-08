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

def main(model_dir,n_scale,io_scale, base,n,interaction):

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Define variables 
    dataset_name = "MNIST"
    testing_size = 10000
    batch_size = 16
    ls_list = []
    acc_list = []
    equiv_list = []
    sum_ = 0
    # Load Model 
    if 'dss_v1' in model_dir:
        model = dss.DSS_plain(base,io_scale,n,interaction)
    elif 'dss_v2' in model_dir:
        model = dss.DSS_plain_2(base,io_scale,n,interaction)
    else:
        exit()
    model = nn.DataParallel(model,device_ids=[0])
    print(model,flush=True)
    model = torch.load('./tmp/'+str(model_dir))
    print(model,flush=True)
    model = model.to(device)
    params = model.parameters()
    print("Number of parameters = ",sum([np.prod(p.size()) for p in params]),flush=True)
    downscale_kern = BesselConv2d(base,n_scales=n_scale)
    # test on original dataset
    print("MNIST scaled by 1",flush=True)
    if 'MNIST' in model_dir:
        test_gen = dataset.get_gen("MNIST",'test',batch_size,1,testing_size,[downscale_kern,n_scale], t=0.0,shuffle=False,resize=None)
    elif 'cifar' in model_dir:
        test_gen = dataset.get_gen("cifar",'test',batch_size,1,testing_size,[downscale_kern,n_scale],resize=["resize",2] ,t=0.0,shuffle=False)
        
    acc, reg_em = funcs.test_network(model,test_gen, testing_size,batch_size)
    acc_list.append([1,acc])

    # test accuracy and equivariance error on different scale 
    for s in [1.1,1.5, 2, 2.5, 3]: # 
        # check accuracy
        print("MNIST scaled by %1.3f" %(1/s),flush=True)
        if 'MNIST' in model_dir:
            test_scaled_gen = dataset.get_gen("MNIST",'test',batch_size,1/s,testing_size,[downscale_kern,n_scale] ,t=0.0,shuffle=False)
        elif 'cifar' in model_dir:
            test_scaled_gen = dataset.get_gen("cifar",'test',batch_size,1/s,testing_size,[downscale_kern,n_scale] ,t=0.0,shuffle=False)
            
        acc, _ = funcs.test_network(model,test_scaled_gen, testing_size, batch_size)
        acc_list.append([s,acc])
        sum_ = 0
        # check equivariance error
        for level in range(io_scale[0]):
            sum_level =0
            for idx in range(testing_size//batch_size):
                reg_batch, reg_y = test_gen[idx*batch_size:min((idx+1)*batch_size,testing_size)]
                try:
                    reg_y = torch.from_numpy(reg_y)
                    reg_batch = torch.from_numpy(reg_batch)
                except:
                    pass
                reg_batch, reg_y = reg_batch.float().cuda(), reg_y.long().cuda(); reg_batch, reg_y = Variable(reg_batch), Variable(reg_y)
            
                sc_batch, sc_y = test_scaled_gen[idx*batch_size:min((idx+1)*batch_size,testing_size)]
                try:
                    sc_y = torch.from_numpy(sc_y)
                    sc_batch = torch.from_numpy(sc_batch)
                except:
                    pass
                sc_batch, sc_y = sc_batch.float().cuda(), sc_y.long().cuda() ; sc_batch, sc_y = Variable(sc_batch), Variable(sc_y)
            
                _,reg_em = model(reg_batch)    
                _,sc_em = model(sc_batch)
                if isinstance(sc_em,list):
                    err = equiv_funcs.cal_dss_equiv_err(sc_em, reg_em,s,level,base,io_scale[0])
                else:
                    err  = equiv_funcs.cal_dss_equiv_err(sc_em.detach().permute(3,4,2,1,0),reg_em.detach().permute(3,4,2,1,0),s,level,base,io_scale[0])
                sum_level += err    
            if isinstance(sc_em,list):
                sum_level /= sc_em[0].shape[1] * testing_size
            else:
                sum_level /= sc_em.shape[1]* testing_size
            print("\tEquivariance error for scale level %d = %2.3f"%(level,sum_level),flush=True)
            sum_ += sum_level
        sum_ /= io_scale[0]
        equiv_list.append([s,sum_])
        # Display result
        print("\tEquivariance error when scaled by %2.3f = "%(1/s),str(sum_),flush=True)
    equiv_funcs.plot_graph(equiv_list,acc_list,"DSS_plain")

if __name__ == "__main__":
#def main(model_dir,n_scale,io_scale, base,n,interaction):

    for args in [('dss_v2_MNIST_1_64_4_8_13.th',8,[4,4],1.3,64,1)]:  #('dss_v2_MNIST_final.th',4,[4,4],1.2,16,1)  
        print("=================",args)
        main(args[0],args[1],args[2],args[3],args[4],args[5])

