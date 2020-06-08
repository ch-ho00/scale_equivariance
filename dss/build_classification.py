from Dconv import *
import warnings
warnings.filterwarnings("ignore")
import sys,os
sys.path.insert(0,os.path.abspath(os.path.join('..')))
from utilities import dataset,funcs,equiv_funcs
from models import dss,vgg,dss2 #,dcn
from tqdm import tqdm
import mlflow.pytorch
import tensorflow as tf
import tensorflow.summary
from tensorflow.summary import scalar
from tensorflow.summary import histogram
import argparse
from tensorboardX import SummaryWriter
import tempfile
import torch.multiprocessing as mp


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

def main(set_,batch_size,model_str, base,io_scale,n_scale,interaction,n,epoch, init_rate,decay, gamma, step_size,save_dir):
   
    params = Params(set_,batch_size,model_str, base,io_scale,n_scale,interaction,n,epoch, init_rate,decay, gamma, step_size)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    downscale_kern = BesselConv2d(base,n_scales=n_scale).to(device)
    n_train = 10000
    n_test = 10000  
    num_process =8  
    resize= None
    if set_ == "cifar":
        resize= ["resize",2]
    if "dss" in model_str or 'deform' in model_str or 'final' in model_str:
        down = [downscale_kern,n_scale]
    else:
        down = None
    resize= None 
    train_gen = dataset.get_gen_rand(set_,"train", batch_size,downscale_kern=down,n=n_train, resize=resize)
    test_gen = dataset.get_gen_rand(set_,"test" ,batch_size,downscale_kern=down,n=n_test, resize=resize)
    
    with mlflow.start_run():  
        output_dir = './log'

        t_list = []
        # MLflow log parameters
        for key, value in vars(params).items():
            mlflow.log_param(key, value)
        output_dir = dirpath = tempfile.mkdtemp()
        writer = SummaryWriter(output_dir)
        print("Writing TensorBoard events locally to %s\n" % output_dir)

        #load model
        if set_ == "MNIST":
            if model_str == 'dss_v1':
                model = dss.DSS_plain(base,io_scale,n,interaction)
            elif model_str =='dss_v2':
                model = dss.DSS_plain_2(base,io_scale,n,interaction)
            elif model_str == "deform":
                model = dss2.deform_dss(base,io_scale[0],n,10)
            elif model_str == "deform2":
                model = dss2.deform_dss2(base,io_scale[0],n,10)               
        elif set_ == "cifar":
            if model_str == 'dss_v1':
                model = dss.DSS_cifar(base,io_scale,n,interaction)
            elif model_str =='dss_v2':
                model = dss.DSS_2_cifar(base,io_scale,n,interaction)
            elif model_str =="cnn":
                model = vgg.vgg16()
            elif model_str == "deform":
                model = dss2.deform_dss_cifar(base,io_scale[0],n,10)
            elif model_str == "deform_res":
                model = dcn.deform_ResNet101()
        model = model.to(device)
        model = nn.DataParallel(model,device_ids=[0])
        #model.share_memory()
        # For multiprocessing
        #mp.set_start_method('spawn')
        #num_gpus = torch.cuda.device_count()
        #rank = int(os.environ['RANK'])
        #dist.init_process_group(backend='nccl')        


        print(model,flush=True)
        model_param = model.parameters()
        print("Number of parameters = ",sum([np.prod(p.size()) for p in model_param]),flush=True)
        enter = 0 
        # train/val model
        for e in range(1,epoch+1):
            print("Epoch: %d" % (e),flush=True)
            start = time.time()
            model = funcs.train_network(model,train_gen,init_rate, step_size, gamma,decay, n_train,batch_size,e,writer) 
            end = time.time()
            t_list.append(end-start)
            acc, _ = funcs.test_network(model, test_gen, n_test, batch_size,e,writer)
            if e > 3 and acc < 0.2:
                enter = 1
                break
          
        avg_time = sum(t_list)/len(t_list)

        #test model
        if enter == 0:
            for s in [1,1.1,1.5,2.0,2.5,3.0]:
                print("Test for scale by %2.3f"%(1/s), flush=True)
                if  set_ =="cifar":
                    sc_test_gen = dataset.get_gen(set_,"test", batch_size, 1/s,n_test,down, resize=resize)
                else:
                    sc_test_gen = dataset.get_gen(set_,"test", batch_size, 1/s,n_test,down)
                model = model.to(device)
                acc,_ = funcs.test_network(model, sc_test_gen, n_test, batch_size,-1,writer,val=False)
            print("Average time taken for one epoch =",str(avg_time),flush= True)
            torch.save(model, str(save_dir))
        #mlflow.log_artifacts(output_dir, artifact_path="events")
        #print("Launch Tensorboard with",os.path.join(mlflow.get_artifact_uri(),"events"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",nargs='?',const="MNIST")
    parser.add_argument("--batch_size",nargs='?',const="32",type=int)
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
    parser.add_argument("--save_dir",nargs='?', const="models/dss_cityscapes_seg.th")
    args = parser.parse_args()
    print(args)
    saved = os.listdir('./models/')
    for base in [1.3,1.5,1.7]:
        for decay in [0]:
           for gamma in [0.5]:
               for step_size in [20]:
                   for channel in [32]:
                       for io_scale in [4,8]:
                           for n_scale in [4,8]:
                               for interaction in [1,2]:
                                   for init_rate in [0.01,0.001]:
                                       #try:
                                           
                                           save_dir = "./models/%s_%s_%d_%d_%d_%d_%d.th"%(args.model,args.dataset,interaction,channel,io_scale,n_scale,int(base*10))
                                           print(args.dataset,args.batch_size, args.model, base, [io_scale,io_scale],n_scale,interaction,channel, args.epoch,init_rate,decay,gamma, step_size,save_dir,"===========")
                                           
                                           if save_dir[9:] in saved:
                                               continue
                                           main(args.dataset,args.batch_size, args.model, base, [io_scale,io_scale],n_scale,interaction,channel, args.epoch,init_rate,decay,gamma, step_size,save_dir)     
                                       #except Exception as e:
                                           #print(e)
                                           #continue
    #main(args.dataset,args.batch_size, args.model, args.base, [args.io_scale,args.io_scale],args.n_scale,args.interaction,args.channel, args.epoch,args.init_rate,args.decay,args.gamma, args.step_size,args.save_dir)



