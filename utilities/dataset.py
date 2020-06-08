import torch.optim as optim
import os,pickle
import torch
from torch.utils import data
from PIL import Image
import numpy as np
import scipy.misc
import time
import torch.nn as nn
from keras.datasets import mnist,cifar10
import glob
import cv2 as cv
from scipy.ndimage import zoom
from torch.utils import data
import matplotlib.pyplot as plt 
from utilities import equiv_funcs,utils 
import tensorflow as tf 
import urllib.request
import tarfile
import _pickle as pickle
from keras.utils import np_utils
import numpy as np

class Dataset(data.Dataset):
    def __init__(self,name="cityscapes",split="train" ,dir_=None,data=None,dtype="crop", resize=None,scale=None,downscale_kern=None):
        # train_dir, train_dir, test_x_dir, test_y_dir
        self.name = name
        self.split= split
        self.dir_ = dir_
        
        self.resize = resize
        self.dtype = dtype
        self.scale = scale
        self.downscale_kern= downscale_kern
        if dir_ != None:
            tmpl = sorted(os.listdir(dir_[0]))
            tmpl = [dir_[0] + img for img in tmpl]
            self.x_img_dirs = tmpl

            tmpl = sorted(os.listdir(dir_[1]))
            tmpl = [dir_[1] + img for img in tmpl]
            self.y_img_dirs = tmpl
            
        else:
            self.data = data
    def __len__(self):
        return len(self.x_img_dirs)

    def __getitem__(self,idx):
        if self.dir_ != None: # when only directory is given
            if isinstance(self.x_img_dirs[idx], list):
                x_imgs = [np.asarray(Image.open(x_img_dir)) for x_img_dir in self.x_img_dirs[idx]]
                x_imgs = np.array(x_imgs)

                y_imgs = [np.asarray(Image.open(y_img_dir)) for y_img_dir in self.y_img_dirs[idx]]
                y_imgs = np.array(y_imgs)                
            else:
                x_imgs = np.asarray(Image.open(self.x_img_dirs[idx]))
                y_imgs = np.asarray(Image.open(self.y_img_dirs[idx])) 
            if x_imgs.ndim == 4:
                x_imgs = np.transpose(x_imgs,[0,3,1,2])
                #y_imgs = np.transpose(y_imgs,[0,3,1,2])
            elif x_imgs.ndim == 3:
                x_imgs = np.transpose(x_imgs,[2,0,1])
                #y_imgs = np.transpose(y_imgs,[2,0,1])
            if self.scale != None:
                s = self.scale
                # TODO: Fix this part
                if s == 1:
                    pass
                elif s != -1: # when scale is fixed; s must be <1
                    if x_imgs.ndim == 4:
                        x_imgs = np.stack([[equiv_funcs.scale(x_imgs[i][j],s)] for j in range(x_imgs.shape[1])] for i in range(x_imgs.shape[0]))
                        y_imgs = np.stack([equiv_funcs.scale(y_imgs[i],s)] for i in range(y_imgs.shape[0]))
                        x_imgs = x_imgs.squeeze(2)
                        y_imgs = y_imgs.squeeze(2)
                    elif x_imgs.ndim == 3:
                        x_imgs = np.stack([[equiv_funcs.scale(x_imgs[i],s)] for i in range(x_imgs.shape[0])] )
                        y_imgs = np.stack([equiv_funcs.scale(y_imgs,s)] )
                        x_imgs = x_imgs.squeeze(1)
                        y_imgs = y_imgs.squeeze(0)
                else:
                    if x_imgs.ndim == 4:
                        xl = []
                        yl = [] 
                        for i in range(x_imgs.shape[0]):
                            s = np.random.uniform(0.3,1)
                            xl.append([equiv_funcs.scale(x_imgs[i][j],s)] for j in range(x_imgs.shape[1]))
                            yl.append(equiv_funcs.scale(y_imgs[i],s))
                        x_imgs = np.stack(xl)
                        y_imgs = np.stack(yl)
                        x_imgs = x_imgs.squeeze(2)
                        y_imgs = y_imgs.squeeze(2)
                    elif x_imgs.ndim == 3:
                        s = np.random.uniform(0.3,1)
                        x_imgs = np.stack([equiv_funcs.scale(x_imgs[i],s) for i in range(x_imgs.shape[0])] )
                        y_imgs = np.stack([equiv_funcs.scale(y_imgs,s)])
                        x_imgs = x_imgs.squeeze(1)
                        y_imgs = y_imgs.squeeze(0)      
            if self.resize != None:
                s = self.resize[1] # scale < 1 
                if self.resize[0] == "resize" and x_imgs.ndim == 4 :
                    shape = (int(x_imgs.shape[3]*s),int(x_imgs.shape[2]*s))
                    x_imgs = np.array([[cv.resize(x_imgs[i][j],shape) for j in range(x_imgs.shape[1])] for i in range(x_imgs.shape[0])])
                    y_imgs = np.array([[cv.resize(y_imgs[i],shape)] for i in range(y_imgs.shape[0])])
                    y_imgs = y_imgs.squeeze(1)
                elif self.resize[0] == "resize" and x_imgs.ndim == 3:
                    shape = (int(x_imgs.shape[2]*s),int(x_imgs.shape[1]*s))
                    x_imgs = np.stack([cv.resize(x_imgs[i],shape)] for i in range(x_imgs.shape[0])).squeeze(1)
                    y_imgs = np.stack([cv.resize(y_imgs,shape)]).squeeze(0)
            
            if self.downscale_kern != None:
                if x_imgs.ndim == 4:
                    x_imgs = torch.cat([self.downscale_kern[0](torch.tensor(x_imgs[:,i,:,:]).unsqueeze(1)) for i in range(x_imgs.shape[1])] )
                    x_imgs = x_imgs.permute(1,0,2,3,4)
                    # if self.name != "cityscapes":
                    #     y_imgs = torch.cat([self.downscale_kern(torch.Tensor(y_imgs).cuda())])
                    #     y_imgs = y_imgs.reshape(-1,4,x_imgs.shape[-2],x_imgs.shape[-1])
                elif x_imgs.ndim == 3:
                    x_imgs = torch.cat([self.downscale_kern[0](torch.tensor(x_imgs[i,:,:]).unsqueeze(0).unsqueeze(0)) for i in range(x_imgs.shape[0])] )
                    x_imgs = x_imgs.permute(1,0,2,3,4)

                    # if self.name != "cityscapes":
                    #     y_imgs = torch.cat([self.downscale_kern(torch.Tensor(y_imgs).cuda())])
                    #     y_imgs = y_imgs.reshape(-1,4,x_imgs.shape[-2],x_imgs.shape[-1])
            return torch.tensor(x_imgs), torch.tensor(y_imgs)
        else: # when data is explicitly given
             x_imgs = self.data[0][idx]
             if self.name == "cifar":
                 if self.data[1][idx].ndim == 2:
                     _, ylabels = torch.max(torch.tensor(self.data[1][idx]),1)
                 else:
                     _,ylabels = torch.max(torch.tensor(self.data[1][idx]),0)
             else:
                 ylabels = self.data[1][idx]
             if self.scale != None:
                s = self.scale
                if s != -1: # when scale is fixed; s must be <1
                    if x_imgs.ndim == 4:
                        x_imgs = np.stack([[equiv_funcs.scale(x_imgs[i][j],s)] for j in range(x_imgs.shape[1])] for i in range(x_imgs.shape[0]))
                        x_imgs = x_imgs.squeeze(2)
                    elif x_imgs.ndim == 3:
                        x_imgs = np.stack([[equiv_funcs.scale(x_imgs[i],s)] for i in range(x_imgs.shape[0])] )
                        x_imgs = x_imgs.squeeze(1)
                else:
                    if x_imgs.ndim == 4:
                        scaled = []
                        x_imgs = np.transpose(x_imgs,(2,3,0,1))
                       
                        for j in range(x_imgs.shape[2]):
                            rand = np.random.uniform(0.3,1)
                            scaled.append(equiv_funcs.scale(np.expand_dims(x_imgs[:,:,j,:],axis=2),rand)) 
                        x_imgs = np.transpose(np.stack(scaled).squeeze(3),(0,3,1,2))
                    elif x_imgs.ndim == 3:
                        x_imgs = equiv_funcs.scale(np.transpose(x_imgs,(1,2,0)),np.random.uniform(0.3,1))
                        x_imgs = np.transpose(x_imgs,(2,0,1))
             if self.resize != None:
                s = self.resize[1] # scale < 1 
                if self.resize[0] == "resize" and x_imgs.ndim == 4 :
                    shape = (int(x_imgs.shape[3]*s),int(x_imgs.shape[2]*s))
                    x_imgs = np.array([[cv.resize(x_imgs[i][j],shape) for j in range(x_imgs.shape[1])] for i in range(x_imgs.shape[0])])
                elif self.resize[0] == "resize" and x_imgs.ndim == 3:
                    shape = (int(x_imgs.shape[2]*s),int(x_imgs.shape[1]*s))
                    x_imgs = np.stack([cv.resize(x_imgs[i],shape)] for i in range(x_imgs.shape[0])).squeeze(1)

             if self.downscale_kern != None:
                x_imgs = torch.stack([self.downscale_kern[0](torch.tensor(x_imgs[:,i,:,:]).unsqueeze(1).cuda()) for i in range(x_imgs.shape[1])]).squeeze(2) 
                x_imgs = x_imgs.permute(1,0,2,3,4)
             return torch.tensor(x_imgs), torch.tensor(ylabels)
    def __repr__(self):
        return self.dataset +"-"+ self.split
    
def get_mnist_dataset(n,resize=None,scale=None,downscale_kern=None):
    (X_train, y_train), (X_test, y_test) = mnist.load_data('mnist.npz')
    X_train = X_train[:n].astype('float32') / 255
    X_test = X_test[:n].astype('float32') / 255
    X_train = X_train[..., None]
    X_test = X_test[..., None]
    Y_train = y_train[:n]
    Y_test = y_test[:n]

    X_train = np.transpose(X_train,(0,3,1,2))
    X_test = np.transpose(X_test,(0,3,1,2))

    trainloader = Dataset("mnist","train", data=[X_train,Y_train],resize=resize,scale=scale,downscale_kern=downscale_kern )
    testloader = Dataset("mnist","test", data=[X_test,Y_test],resize=resize,scale=scale,downscale_kern=downscale_kern )
    return trainloader, testloader

def get_cityscapes_dataset(resize=None,scale=None,downscale_kern=None):
    # /home/SENSETIME/parkchanho/Desktop/etc-repo/cityscapes/gtFine/train
    train_x_dir = '/mnt/lustre/parkchanho/etc-repo/data/leftImg8bit/train/'
    train_y_dir = '/mnt/lustre/parkchanho/etc-repo/data/gtFine/train/'

    test_x_dir = '/mnt/lustre/parkchanho/etc-repo/data/leftImg8bit//val/'
    test_y_dir = '/mnt/lustre/parkchanho/etc-repo/data/gtFine/val/'

    if 'class_weights.pkl' not in os.listdir('./utilities'):
        utils.cityscapes_cls_weight(train_y_dir)

    with open("./class_weights.pkl", "rb") as file: # (needed for python3)
        class_weights = np.array(pickle.load(file))

    trainloader = Dataset("cityscapes","train",[train_x_dir,train_y_dir],resize=resize,scale=scale,downscale_kern=downscale_kern)
    testloader = Dataset("cityscapes","test",[test_x_dir,test_y_dir],resize=resize,scale=scale,downscale_kern=downscale_kern)
    return trainloader, testloader, class_weight

# CIFAR-10 constants
home = os.path.expanduser('~')
data_path = os.path.join(home, "data/CIFAR-10/")

img_size = 32
img_channels = 3
nb_classes = 10
# length of the image after we flatten the image into a 1-D array
img_size_flat = img_size * img_size * img_channels
nb_files_train = 5
images_per_file = 10000
# number of all the images in the training dataset
nb_images_train = nb_files_train * images_per_file


def load_data(file_name):
    file_path = os.path.join(data_path, "cifar-10-batches-py/", file_name)
    
    #print('Loading ' + file_name)
    with open(file_path, mode='rb') as file:    
        data = pickle.load(file, encoding='bytes')
    raw_images = data[b'data']
    cls = np.array(data[b'labels'])
    
    images = raw_images.reshape([-1, img_channels, img_size, img_size])    
    # move the channel dimension to the last
    images = np.rollaxis(images, 1, 4)
    
    return images, cls

def load_training_data():    
    # pre-allocate the arrays for the images and class-numbers for efficiency.
    images = np.zeros(shape=[nb_images_train, img_size, img_size, img_channels], 
                      dtype=int)
    cls = np.zeros(shape=[nb_images_train], dtype=int)
    
    begin = 0
    for i in range(nb_files_train):
        images_batch, cls_batch = load_data(file_name="data_batch_" + str(i + 1))
        num_images = len(images_batch)
        end = begin + num_images
        images[begin:end, :] = images_batch
        cls[begin:end] = cls_batch
        begin = end
        
    return images, np_utils.to_categorical(cls, nb_classes)

def load_test_data():
    images, cls = load_data(file_name="test_batch")
    
    return images, np_utils.to_categorical(cls, nb_classes)


def download_and_extract_cifar(file_path, data_path):
    tarfile.open(name=file_path, mode="r:gz").extractall(data_path)
    print('\nExtracting... ', end='',flush=True)
    print('done',flush=True)    

def load_cifar():
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        download_and_extract_cifar("~/data/cifar-10-python.tar.gz",data_path)

    X_train, Y_train = load_training_data()
    X_test, Y_test = load_test_data()
    
    return (X_train, Y_train), (X_test, Y_test)

def get_cifar_dataset(n,resize=None,scale=None,downscale_kern=None):
    (X_train, Y_train), (X_test, Y_test) = load_cifar()
    
    X_train = X_train[:n].astype('float32') / 255
    X_test = X_test[:n].astype('float32') / 255
   
    X_train = np.transpose(X_train,(0,3,1,2))
    X_test = np.transpose(X_test,(0,3,1,2))
    trainloader = Dataset("cifar","train", data=[X_train,Y_train],resize=resize,scale=scale,downscale_kern=downscale_kern )
    testloader = Dataset("cifar","test", data=[X_test,Y_test],resize=resize,scale=scale,downscale_kern=downscale_kern )
    return trainloader, testloader


def get_gen(dataset,set_name, batch_size,  s,n=10000,downscale_kern=None,t=0,
            shuffle=False, resize=None):
    '''
    Create image generator with images scaled by 1/s and translated by t
    Input:
        dataset = name of dataset
        set_name = train/test
        s = single float number representing scale operation on dataset 
        n = number of dataset size
        downscale_kern = downscale kernel for DSS
        t = translation
        shuffle
        preprocess = ['upscale','downscale']
    Output:
        torch.data.dataset object
    '''
    if dataset == "MNIST":
        if set_name == 'train':
            trainloader, _ = get_mnist_dataset(n,resize,s,downscale_kern)
            return trainloader
        elif set_name == 'test':
            _, testloader = get_mnist_dataset(n,resize,s,downscale_kern)
            return testloader
    elif dataset == "cifar":
        if set_name == 'train':
            trainloader, _ = get_cifar_dataset(n,resize,s,downscale_kern)
            return trainloader
        elif set_name == 'test':
            _, testloader = get_cifar_dataset(n,resize,s,downscale_kern)
            return testloader
    elif dataset == "cityscapes":
        if set_name == 'train':
            loader, _ = get_cityscapes_dataset(resize=resize, scale=s,downscale_kern=downscale_kern)
        elif set_name == 'test':
            _, loader = get_cityscapes_dataset(resize=resize, scale=s,downscale_kern=downscale_kern)
        return loader

    return None


def get_gen_rand(dataset,set_name, batch_size=32,n=10000, downscale_kern=None,t=0, shuffle=False,resize=None):
    '''
    Create image generator with images randomly scaled between 0.3 to 1 
    Input:
        dataset = name of dataset
        set_name = train/test

    Output:
        torch.data.dataset object
    '''
    if dataset == "MNIST":
        if set_name == 'train':
            trainloader, _ = get_mnist_dataset(n, resize,-1 ,downscale_kern)
            return trainloader
        elif set_name == 'test':
            _, testloader = get_mnist_dataset(n, resize,-1,downscale_kern)
            return testloader
    if dataset == "cifar":
        if set_name == 'train':
            trainloader, _ = get_cifar_dataset(n, resize,-1 ,downscale_kern)
            return trainloader
        elif set_name == 'test':
            _, testloader = get_cifar_dataset(n, resize,-1,downscale_kern)
            return testloader
    elif dataset == "cityscapes":
        if set_name == 'train':
            loader, _ = get_cityscapes_dataset(resize=resize, scale=-1,downscale_kern=downscale_kern)
        elif set_name == 'test':
            _, loader = get_cityscapes_dataset(resize=resize, scale=-1,downscale_kern=downscale_kern)
        return loader
    return None


