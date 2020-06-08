from utils import *
from torch.autograd import Variable
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
from ScaleSteerableInvariant_Network import Net_steerinvariant_mnist_scale
import sys,os
sys.path.insert(0,os.path.abspath(os.path.join('..')))
print(sys.path)
from utilities import dataset,funcs,equiv_funcs

def main(model_dir):

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

    # Load Model 
    model = Net_steerinvariant_mnist_scale()
    model.load_state_dict(torch.load(model_dir))
    model = model.to(device)
    params = model.parameters()
    print(model)
    print("Number of parameters = ",sum([np.prod(p.size()) for p in params]))

    test_gen =  dataset.get_gen(dataset_name,"test",batch_size,1,resize=["resize",2])
    print("MNIST scaled by 1")
    acc, reg_em = funcs.test_network(model,test_gen, testing_size, batch_size)
    acc_list.append([1,acc])
    # test accuracy and equivariance error on different scale 
    for s in [1.1,1.5,2, 2.5, 3]: #  
        # check accuracy
        test_scaled_gen = dataset.get_gen(dataset_name,"test",batch_size,1/s,resize=["resize",2])
        print("MNIST scaled by %1.3f" %(1/s))
        acc, _ = funcs.test_network(model,test_scaled_gen,testing_size, batch_size)
        acc_list.append([s,acc])
        # check equivariance error
        for idx in range(testing_size//batch_size):
            reg_batch, reg_y = test_gen[idx*batch_size:min((idx+1)*batch_size,testing_size)]
            reg_batch, reg_y = torch.Tensor(reg_batch).float().cuda(), torch.Tensor(reg_y).long().cuda(); reg_batch, reg_y = Variable(reg_batch), Variable(reg_y)
            
            sc_batch, sc_y = test_scaled_gen[idx*batch_size:min((idx+1)*batch_size,testing_size)]
            sc_batch, sc_y = torch.Tensor(sc_batch).float().cuda(), torch.Tensor(sc_y).long().cuda() ; sc_batch, sc_y = Variable(sc_batch), Variable(sc_y)
            
            _,reg_em = model(reg_batch)    
            _,sc_em = model(sc_batch)
            err  = equiv_funcs.cal_equiv_err(sc_em.detach(),reg_em.detach(),1/s)
            sum_ += err
        sum_ /= testing_size* sc_em.shape[1]
        equiv_list.append([s,sum_])
        # Display result
        print("\tEquivariance error when scaled by %2.3f = "%(1/s),str(sum_))

    plot_graph(equiv_list,acc_list,'SS_CNN')

if __name__ == "__main__":
    model_dir = './models/SSCNN_mnist_new.pt'
    main(model_dir)

