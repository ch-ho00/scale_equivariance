from Dconv import *
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0,os.path.abspath(os.path.join('..')))
from utilities import dataset,funcs
from models import dss,resnet101
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import torch
import argparse
import tempfile
from tensorflow.summary import scalar
from tensorflow.summary import histogram
from tensorboardX import SummaryWriter
import mlflow 

class Params(object):
    def __init__(self, dataset,batch_size,model_, base,io_scale,n_scale,interaction,n,epoch, init_rate,decay, gamma, step_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.model_ = model_
        self.base = base
        self.io_scale = io_scale
        self.n_scale = n_scale
        self.interaction = interaction
        self.channel = n
        self.epoch = epoch
        self.init_rate = init_rate
        self.decay = decay
        self.gamma = gamma
        self.step_size = step_size

def main(set_,batch_size,model_str,base, io_scale, n_scale,interaction,n,epoch,init_rate,decay,gamma,step_size,save_dir):

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    downscale_kern = BesselConv2d(base,n_scales=n_scale)
    params = Params(set_,batch_size,model_str, base,io_scale,n_scale,interaction,n,epoch, init_rate,decay, gamma, step_size)
 
    n_train = 2975
    n_test = 500

    with mlflow.start_run():
        output_dir = './log'

        for key, value in vars(params).items():
            mlflow.log_param(key, value)
        output_dir = dirpath = tempfile.mkdtemp()
        writer = SummaryWriter(output_dir)
        print("Writing TensorBoard events locally to %s\n" % output_dir)

        tlist = []
        start = time.time()
        if model_str=="dss":
            model = torch.nn.DataParallel(dss.DSS_cityscape(base,io_scale,n)).cuda()
            trainloader,cls_weight = dataset.get_gen(set_,'train',batch_size, 1, n_train, downscale_kern=[downscale_kern,n_scale],resize=["resize",0.5])
            testloader,_ = dataset.get_gen(set_,'test',batch_size, 1, n_test, downscale_kern=[downscale_kern,n_scale],resize=["resize",0.5])

        elif model_str=="resnet":
            model = torch.nn.DataParallel(resnet101.ResNet(resnet101.Bottleneck,[3, 4, 23, 3],20)) # nolabel ?
            trainloader,cls_weight = dataset.get_gen('cityscapes','train',batch_size, 1, n_train,resize=["resize",0.5])
            testloader,_ = dataset.get_gen('cityscapes','test',batch_size, 1, n_test,resize=["resize",0.5])
        print(model,flush=True)
        model_params= model.parameters()
        print("Number of parameters = ",sum([np.prod(p.size()) for p in model_params]), flush=True)

        for i in range(1,epoch+1):
            epstart = time.time()
            if i % 40 ==0:
                init_rate /= 10
            print("Epoch =",i,flush=True)

            model = funcs.train_seg_network(model, trainloader,init_rate, step_size,gamma, 0, n_train, batch_size,i,writer,cls_weight)
            #funcs.log_weights(writer,model,model_str,i)
            epend = time.time()
            tlist.append(epend-epstart)
            torch.cuda.empty_cache()
        print("Average time per epoch",sum(tlist)/len(tlist),flush=True)
            # ypred = funcs.test_seg_network(model, testloader, n_test, batch_size)
        torch.save(model,save_dir)
        end = time.time()
        print("Total time taken =",int(end-start))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",nargs='?',const="cityscapes")
    parser.add_argument("--batch_size",nargs='?',const="1",type=int)
    parser.add_argument("--model",nargs='?',const="dss_v1")
    parser.add_argument("--base",nargs='?',const="1.3",type=float)
    parser.add_argument("--io_scale",nargs='?',const="4",type=int)
    parser.add_argument("--n_scale",nargs='?',const="4",type=int)
    parser.add_argument("--interaction",nargs='?',const='1',type=int)
    parser.add_argument("--channel",nargs='?', const='16',type=int)
    parser.add_argument("--epoch",nargs='?',const="100",type=int)
    parser.add_argument("--init_rate",nargs='?',const="0.005",type=float)
    parser.add_argument("--decay",nargs='?',const="0.005",type=float)
    parser.add_argument("--gamma",nargs='?',const="0.7",type=float)
    parser.add_argument("--step_size",nargs='?',const="10",type=int)
    parser.add_argument("--save_dir",nargs='?',const="./models/dss_cityscape_seg.th")

    args = parser.parse_args()
    main(args.dataset,args.batch_size, args.model, args.base, [args.io_scale,args.io_scale],args.n_scale,args.interaction,args.channel, args.epoch,args.init_rate,args.decay,args.gamma, args.step_size,args.save_dir)
 
