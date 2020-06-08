from __future__ import print_function, absolute_import, division
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import sys,os
sys.path.insert(0,os.path.abspath(os.path.join('..')))
print(sys.path)
from utilities import dataset,funcs
import matplotlib.pyplot as plt
import torch



import os
from PIL import Image
import cv2
import numpy as np

data = dataset.get_cityscapes_dataset(resize=["resize",0.5])

print(type(data[0]))
print(data[0][1][1].shape, torch.max(data[0][1][1]), torch.mean(data[0][1][1]))
print(data[0][1][0].shape)

img = plt.imread("/home/SENSETIME/parkchanho/Desktop/etc-repo/gtFine/train/aachen_000003_000019_gtFine_labelTrainIds.png")
img2 = Image.open("/home/SENSETIME/parkchanho/Desktop/etc-repo/gtFine/train/aachen_000003_000019_gtFine_labelTrainIds.png")
img2 = np.array(img2)
mask = np.zeros_like(img2)
mask = ((img2 == 255) | (img2 == -1)) 
mask = np.array(mask, dtype=int)
print(np.unique(img2))
print(mask)


# plt.figure(1)
# plt.imshow(np.transpose(data[0][1][0],(1,2,0)))
# plt.figure(2)
# plt.imshow(data[0][1][1])
# plt.show()
# plt.close()
# print(img, np.max(img), np.min(img), np.mean(img))
# print(np.array(img2), np.max(img2),np.min(img2), np.mean(img2))
