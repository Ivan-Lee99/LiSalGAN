import numpy as np
from constants import *
import os
import cv2
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image


class DataLoader(object):

    def __init__(self, img_dir, batch_size=5):
        # Reading training img filename list
        self.img_list = os.listdir(img_dir)
        self.batch_size = batch_size
        self.size = len(self.img_list)
        self.cursor = 0
        self.num_batch = int(self.size / self.batch_size)

    def get_batch(self):
        if self.cursor + self.batch_size > self.size:
            self.cursor = 0
            np.random.shuffle(self.img_list)

        img = torch.zeros(self.batch_size, 3, 192, 256)
        salmap = torch.zeros(self.batch_size, 1, 192, 256)

        to_tensor = transforms.ToTensor()

        for index in range(self.batch_size):
            file_name = self.img_list[self.cursor]
            file_name = file_name.split('.')[0]
            img_full_path = os.path.join(pathToResizedImagesTrain, file_name + '.jpg')
            salmap_full_path = os.path.join(pathToResizedMapsTrain, file_name + '.png')
            self.cursor += 1

            input_img = cv2.imread(img_full_path)
            img[index] = to_tensor(input_img)

            salmap_img = cv2.imread(salmap_full_path, 0)
            salmap_img = np.expand_dims(salmap_img, axis=2)  # Add a dimension to conform to img
            salmap[index] = to_tensor(salmap_img)

        return img, salmap

# Check the loader

# trainset = DataLoader(img_dir=pathToResizedImagesTrain)

# print('Data Loader implemented.')
# img, salmap = d.get_batch()
# I1 = img[4].numpy()
# S1 = salmap[4].numpy()

# img_after = np.transpose(I1, (1, 2, 0))
# salmap_after = np.transpose(S1, (1, 2, 0))

# cv2.imshow('img', img_after)
# cv2.waitKey(0)
# cv2.imshow('img', salmap_after)
# cv2.waitKey(0)

