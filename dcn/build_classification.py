import time 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from dataset import *
import torch.optim as optim
from modules.deform_conv import DCN, DCNv2, CNN
import warnings
warnings.filterwarnings("ignore")


def train(model, generator, batch_num, epoch,optimizer):
    model.train()
    for batch_idx in range(batch_num):
        data, target = next(generator)
        data, target = torch.from_numpy(data), torch.from_numpy(target)
        # convert BHWC to BCHW
        data = data.permute(0, 3, 1, 2)
        data, target = data.float().cuda(), target.long().cuda()

        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output,_ = model(data)
        # print("Embedding shape"+ str(embedding.shape))

        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

    print('Train Epoch: {}\tLoss: {:.6f}'.format(epoch, loss.data.item()))

def test(model, generator, batch_num, epoch, n_test):
    model.eval()
    test_loss = 0
    correct = 0
    for batch_idx in range(batch_num):
        data, target = next(generator)

        data, target = torch.from_numpy(data), torch.from_numpy(target)
        # convert BHWC to BCHW
        data = data.permute(0, 3, 1, 2)
        data, target = data.float().cuda(), target.long().cuda()

        data, target = Variable(data), Variable(target)
        output, _ = model(data)
        # print("Embedding shape"+ str(embedding.shape))
        # print(type(F.cross_entropy(output, target).data),F.cross_entropy(output, target).data.item())
        test_loss += F.cross_entropy(output, target).data.item()
        pred = output.data.max(1)[1] # get the index of the max log-probability
        # print(pred.shape, target.data.shape)
        correct += pred.eq(target.data).cpu().sum()

    test_loss /=  batch_num# loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, n_test, 100. * correct / n_test))

def main(dataset = "MNIST",batch_size=32, n_train=60000, n_test=10000):
    '''
    Train Deformable Convolution Network on regular MNIST and test on scale-MNIST
    '''
    steps_per_epoch = int(np.ceil(n_train / batch_size))
    validation_steps = int(np.ceil(n_test / batch_size))
    epoch = 7
    train_gen = get_gen_rand(
        dataset,'train', batch_size=batch_size,
        s=1, t=0.0,
        shuffle=True
    )

    test_gen = get_gen_rand(
        dataset,'test', batch_size=batch_size,
        s=1, t=0.0,
        shuffle=False
    )

    # Need to set up channel number!
    model = DCN(1,10,dataset).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    t_list = []
    for epoch in range(epoch):
        test(model, test_gen, validation_steps, epoch,n_test)
        start = time.time() 
        train(model, train_gen, steps_per_epoch, epoch,optimizer)
        end = time.time()
        t_list.append(end-start)
    avg_time = sum(t_list)/len(t_list)
    torch.save(model, 'models/dcn_MNIST_final.th')

    print('\n\nEvaluate  DCN')
    print("\tAverage time taken for one epoch =",str(avg_time))
    test(model, test_gen, validation_steps, epoch,n_test)
    # for s in [1.2, 1.5, 2,3]:
    #     print("Test on images scaled by = %.3f"%(1/s))
    #     test_scaled_gen = get_gen(
    #         dataset,'test', batch_size=batch_size,
    #         s=s, t=0,
    #         shuffle=False
    #     )
    #     test(model, test_scaled_gen, validation_steps, epoch,n_test)

def main_v2(dataset = "MNIST",batch_size=32, n_train=60000, n_test=10000):
    '''
    Train Deformable Convolution Network v2 on  MNIST and test on scale-MNIST
    '''
    steps_per_epoch = int(np.ceil(n_train / batch_size))
    validation_steps = int(np.ceil(n_test / batch_size))
    epoch = 5
    train_gen = get_gen_rand(
        dataset,'train', batch_size=batch_size,
        s=1, t=0.0,
        shuffle=True
    )

    test_gen = get_gen_rand(
        dataset,'test', batch_size=batch_size,
        s=1, t=0.0,
        shuffle=False
    )

    # Need to set up channel number!
    model = DCNv2(1,10,dataset).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    t_list = []
    for epoch in range(epoch):
        test(model, test_gen, validation_steps, epoch,n_test)
        start = time.time() 
        train(model, train_gen, steps_per_epoch, epoch,optimizer)
        end = time.time()
        t_list.append(end-start)
    avg_time = sum(t_list)/len(t_list)
    torch.save(model, 'models/dcnv2_MNIST_final.th')

    print('\n\nEvaluate  DCN')
    print("\tAverage time taken for one epoch =",str(avg_time))
    test(model, test_gen, validation_steps, epoch,n_test)
    for s in [1.2, 1.5, 2,3]:
        print("Test on images scaled by = %.3f"%(1/s))
        test_scaled_gen = get_gen(
            dataset,'test', batch_size=batch_size,
            s=s, t=0,
            shuffle=False
        )
        test(model, test_scaled_gen, validation_steps, epoch,n_test)
def main_CNN(dataset = "MNIST",batch_size=32, n_train=60000, n_test=10000):
    '''
    Train Deformable Convolution Network v2 on  MNIST and test on scale-MNIST
    '''
    steps_per_epoch = int(np.ceil(n_train / batch_size))
    validation_steps = int(np.ceil(n_test / batch_size))
    epoch = 5
    train_gen = get_gen_rand(
        dataset,'train', batch_size=batch_size,
        s=1, t=0.0,
        shuffle=True
    )

    test_gen = get_gen_rand(
        dataset,'test', batch_size=batch_size,
        s=1, t=0.0,
        shuffle=False
    )

    # Need to set up channel number!
    model = CNN(1,10,dataset).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    t_list = []
    for epoch in range(epoch):
        test(model, test_gen, validation_steps, epoch,n_test)
        start = time.time() 
        train(model, train_gen, steps_per_epoch, epoch,optimizer)
        end = time.time()
        t_list.append(end-start)
    avg_time = sum(t_list)/len(t_list)
    torch.save(model, 'models/cnn_MNIST_final.th')

    print('\n\nEvaluate  DCN')
    print("\tAverage time taken for one epoch =",str(avg_time))
    test(model, test_gen, validation_steps, epoch,n_test)
    for s in [1.2, 1.5, 2,3]:
        print("Test on images scaled by = %.3f"%(1/s))
        test_scaled_gen = get_gen(
            dataset,'test', batch_size=batch_size,
            s=s, t=0,
            shuffle=False
        )
        test(model, test_scaled_gen, validation_steps, epoch,n_test)

if __name__ == "__main__":
    main()
    #main_v2()
    #main_CNN()
