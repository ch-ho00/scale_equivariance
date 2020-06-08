import os
import time

import numpy as np
import torch

from Dconv import  Dconv2d ,BesselConv2d
from matplotlib import pyplot as plt
from skimage.io import imread


def np2torch(x):
    return torch.from_numpy(x).type(torch.FloatTensor)


def main(im):
    # Perform 3 layers of scale space convolutions
    # The rate of downsampling
    base = 1.5
    downscale_kern = BesselConv2d(base,n_scales=4)
    im = downscale_kern(im)
    print("Downscaled images dimension = ",im.shape)
    # TODO: figure out conventions for these
    io_channels = [32,3]
    io_scales =  [4,4]

    # Define the convolutions
    conv = Dconv2d(3,32,[3,3,3], base, io_scales).cuda()
    conv2 = Dconv2d(32,32, [3,3,3],base, io_scales).cuda()
    conv3 = Dconv2d(32,32, [3,3,3],base, io_scales).cuda()

    # Run the convolutions
    start = time.time()
    # im = torch.unsqueeze(im, 0)
    out1 = conv(im)
    out2 = conv2(out1)
    out3 = conv3(out2).detach().cpu().numpy()
    out = np.transpose(out3, [0,1,2,4,3])

    print(out.shape)
    plt.figure(1)
    for i in range(10):
        plt.subplot(4,3,i+1)
        plt.imshow(np.transpose(out[0,i,0,...]))
    plt.show()


if __name__ == '__main__':

    image_folder = '../../utils/img/up'
    stack = []
    for i in range(1,4):
        image_address = os.path.join(image_folder, 'test{}.png'.format(i))
        stack.append(imread(image_address))
    im = np.stack(stack, 0)

    im = np.transpose(im, [0,3,1,2])
    """
    plt.figure(1)
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.imshow(im[i,...])
    plt.show()
    """
    # im = np.transpose(im, [0,3,1,2])

    im = np2torch(im).cuda()
    main(im)
