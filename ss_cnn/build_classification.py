import time 
import torchvision.transforms as transforms
import torch.optim as optim
from ScaleSteerableInvariant_Network import *
import sys,os,pickle
import numpy as np
import torch
from torch.utils import data
from PIL import Image
import numpy as np
import scipy.misc
import scipy.misc
import warnings
# from funcs import *
warnings.filterwarnings("ignore")
sys.path.insert(0,os.path.abspath(os.path.join('..')))
print(sys.path)
from utilities import dataset,funcs,equiv_funcs

def main(dataset_name,save_dir,type_, val_splits=1,n_train = 10000, n_test=10000,batch_size=400,init_rate = 0.01,decay_normal = 0.04,step_size = 10,gamma = 0.7,total_epochs = 20):
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    for n in [16,32,40,64]:
        train_gen = dataset.get_gen_rand(dataset_name,"train", batch_size, n=n_train,resize=["resize",2])
        test_gen = dataset.get_gen_rand(dataset_name,"test" ,batch_size, n=n_test,resize=["resize",2])
        if type_ == "cnn":
            if "MNIST" == dataset_name:
                model = Net_steerinvariant_mnist_scale()
            elif "cifar" == dataset_name:
                model = cnn_cifar10(n)
        else:
            if "MNIST" == dataset_name:
                model = Net_steerinvariant_mnist_scale()
            elif "cifar" == dataset_name:
                model = Net_steerinvariant_cifar10_scale()
        print(model)
        t_list = []
        for i in range(total_epochs):
            print("Epoch =",i)
            start= time.time()
            model = funcs.train_network(model,train_gen, init_rate, step_size,gamma,decay_normal,n_train,batch_size)
            end= time.time()
            t_list.append(end-start)
            accuracy, _ = funcs.test_network(model,test_gen,n_test,batch_size)
            print("\tTest Accuracy:",accuracy)

        torch.save(model.state_dict(),save_dir)
    print("Average epoch time =",(sum(t_list)/len(t_list)))



if __name__ == "__main__":
    dataset_name = 'MNIST'
    type_ = "ss_cnn"
    save_dir = './models/SSCNN_mnist_new.pt'
    main(dataset_name,save_dir,type_,n_test=10000,n_train=10000,total_epochs=35)
    