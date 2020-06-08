from utils import *
from torch.autograd import Variable
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from modules.deform_conv import DCN,DCNv2,CNN


def main(model_dir):

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Define variables 
    dataset = "MNIST"
    training_size = 60000
    testing_size = 1000
    batch_size = 64
    ls_list = []
    acc_list = []
    equiv_list = []
    sum_ = 0
    # Load Model 
    if 'v2' in model_dir:
        model = DCNv2(1,10,"MNIST")
    elif 'cnn' in model_dir:
        model =CNN(1,10,"MNIST")
    else:
        model = DCN(1,10,"MNIST")
    model = torch.load(model_dir)
    model = model.to(device)
    params = model.parameters()
    print(model)
    print("Number of parameters = ",sum([np.prod(p.size()) for p in params]))

    # test on original dataset
    test_gen = get_gen("MNIST",'test', batch_size=batch_size,
        s=1, t=0.0,
        shuffle=False)
    acc, reg_em = test_network(model,test_gen, testing_size, testing_size//batch_size)
    print("Accuracy = %2.2f when MNIST scaled by 1" %(acc))
    acc_list.append([1,acc])
    # test accuracy and equivariance error on different scale 
    for s in [1.1, 1.5,2, 2.5, 3]:
        # check accuracy
        test_gen = get_gen("MNIST",'test', batch_size=batch_size,
            s=1, t=0.0,
            shuffle=False)
        test_scaled_gen = get_gen("MNIST",'test', batch_size=batch_size,
            s=s, t=0.0,
            shuffle=False)
        acc, _ = test_network(model,test_scaled_gen, testing_size, testing_size//batch_size)
        acc_list.append([s,acc])
        print("MNIST scaled by %1.3f" %(1/s))
        # check equivariance error
        test_scaled_gen = get_gen("MNIST",'test', batch_size=batch_size,
            s=s, t=0.0,
            shuffle=False)
        for idx in range(testing_size//batch_size):
            reg_batch, reg_y = next(test_gen)
            reg_batch, reg_y = torch.from_numpy(reg_batch), torch.from_numpy(reg_y)
            reg_batch = reg_batch.permute(0, 3, 1, 2)
            reg_batch, reg_y = reg_batch.float().cuda(), reg_y.long().cuda(); reg_batch, reg_y = Variable(reg_batch), Variable(reg_y)
            
            sc_batch, sc_y = next(test_scaled_gen)
            sc_batch, sc_y = torch.from_numpy(sc_batch), torch.from_numpy(sc_y)
            sc_batch = sc_batch.permute(0, 3, 1, 2)
            sc_batch, sc_y = sc_batch.float().cuda(), sc_y.long().cuda() ; sc_batch, sc_y = Variable(sc_batch), Variable(sc_y)
            
            _,reg_em = model(reg_batch)    
            _,sc_em = model(sc_batch)
            err  = cal_equiv_err(sc_em.detach(),reg_em.detach(),1/s)
            sum_ += err
        # test size * channel 
        sum_ /= testing_size* sc_em.shape[1]
        equiv_list.append([s,sum_])
        # Display result
        print("\tEquivariance error when scaled by %2.3f = "%(1/s),str(sum_))

    fig1= plt.figure(1)
    plt.title('%s Accuracy across scale transformation'%(model_dir[9:-9]))
    plt.plot([a[0] for a in acc_list], [a[1] for a in acc_list])
    plt.xlabel("scale")
    plt.ylabel("Accuracy")
    fig1.savefig('../results/graph/%s_acc.PNG'%(model_dir[9:-9]))
    fig2= plt.figure(2)
    plt.title('%s Equivariance Error across scale transformation'%(model_dir[9:-9]))
    plt.plot([a[0] for a in equiv_list], [a[1] for a in equiv_list])
    plt.xlabel("Scale")
    plt.ylabel("Equivariance error")
    fig2.savefig('../results/graph/%s_equiv.PNG'%(model_dir[9:-9]))
    plt.show()
if __name__ == "__main__":
    for model_dir in  ['./models/cnn_MNIST_final.th','./models/dcn_MNIST_final.th','./models/dcnv2_MNIST_final.th']:
        print("=================",model_dir)
        main(model_dir)

