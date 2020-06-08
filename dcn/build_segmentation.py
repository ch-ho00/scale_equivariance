from modules import deform_conv
import warnings,sys,os
warnings.filterwarnings("ignore")
sys.path.insert(0,os.path.abspath(os.path.join('..')))
print(sys.path)
from utilities import dataset,funcs
from models import dcn
import matplotlib.pyplot as plt
import torch
import numpy as np

def main(save_dir,name, val_splits=1,n_train = 2975, n_test=500,batch_size=1,weight_decay=0.001 ,init_rate = 0.001,decay_normal = 0.0001,step_size = 40,gamma = 0.7,total_epochs = 20):
    trainloader = dataset.get_gen('cityscapes','train',batch_size, 1, n_train,resize=["resize",0.5])
    testloader = dataset.get_gen('cityscapes','test',batch_size, 1, n_test,resize=["resize",0.5])
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    model = dcn.deform_ResNet101()
    params = model.parameters()
    print(model,flush=True)
    print("Number of parameters = ",sum([np.prod(p.size()) for p in params]),flush=True)    

    for i in range(total_epochs):
        print("Epoch =",(i+1),flush=True)
        model = funcs.train_seg_network(model, trainloader,init_rate, step_size,gamma, weight_decay, n_train, batch_size)
        ypred = funcs.test_seg_network(model, testloader, n_test, batch_size)
    torch.save(model,save_dir)

if __name__ == "__main__":
    save_dir = "./models/dcn_resnet_cityscape_seg_0227.th"
    dataset_name = "cityscape"
    main(save_dir,dataset_name)
