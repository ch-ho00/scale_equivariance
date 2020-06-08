import pickle
import os
import numpy as np
import cv2
def cityscapes_cls_weight(train_y_dir):

    train_y_dirs = os.listdir(train_y_dir)
    train_label_img_paths = [train_y_dir+y for y in train_y_dirs]

    num_classes = 20
    trainId_to_count = {}
    for trainId in range(num_classes):
        trainId_to_count[trainId] = 0

    # get the total number of pixels in all train label_imgs that are of each object class:
    for step, label_img_path in enumerate(train_label_img_paths):
        if step % 100 == 0:
            print (step)

        label_img = cv2.imread(label_img_path, -1)
        label_img[label_img>18] = 19
        for trainId in range(num_classes):
            # count how many pixels in label_img which are of object class trainId:
            trainId_mask = np.equal(label_img, trainId)
            trainId_count = np.sum(trainId_mask)

            # add to the total count:
            trainId_to_count[trainId] += trainId_count

    # compute the class weights according to the ENet paper:
    class_weights = []
    total_count = sum(trainId_to_count.values())
    for trainId, count in trainId_to_count.items():
        trainId_prob = float(count)/float(total_count)
        trainId_weight = 1/np.log(1.02 + trainId_prob)
        class_weights.append(trainId_weight)

    print (class_weights)

    with open("./class_weights.pkl", "wb") as file:
        pickle.dump(class_weights, file, protocol=2) # (protocol=2 is needed to be able to open this file with python2)

