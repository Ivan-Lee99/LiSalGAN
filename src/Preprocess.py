import glob
import os
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from constants import *

INPUT_SIZE = (256, 192)

# Path to SALICON raw data
pathToTrain = '../imgs/train'
pathToVal = '../imgs/val'

# Path to SALICON corresponding saliency map
pathToTrainMap = '../maps/train'
pathToValMap = '../maps/val'

# Path to resized above data
pathToResizedImagesTrain = '../imgs/train256x192'
pathToResizedMapsTrain = '../maps/train256x192'

pathToResizedImagesVal = '../imgs/val256x192'
pathToResizedMapsVal = '../maps/val256x192'

# Make Train dir(img&map)
if not os.path.exists(pathToResizedImagesTrain):
    os.makedirs(pathToResizedImagesTrain)
if not os.path.exists(pathToResizedMapsTrain):
    os.makedirs(pathToResizedMapsTrain)

# Make Val dir(img&map)
if not os.path.exists(pathToResizedImagesVal):
    os.makedirs(pathToResizedImagesVal)
if not os.path.exists(pathToResizedMapsVal):
    os.makedirs(pathToResizedMapsVal)

for file in tqdm(os.listdir(pathToTrain)):
    file = file.split('.')[0]
    img_path = os.path.join(pathToTrain, file + '.jpg')
    map_path = os.path.join(pathToTrainMap, file + '.png')
    img_save_path = os.path.join(pathToResizedImagesTrain, file + '.jpg')
    map_save_path = os.path.join(pathToResizedMapsTrain, file + '.png')
    imageResized = cv2.resize(cv2.imread(img_path), INPUT_SIZE, interpolation=cv2.INTER_AREA)
    mapResized = cv2.resize(cv2.imread(map_path), INPUT_SIZE, interpolation=cv2.INTER_AREA)
    # cv2.imshow('none', mapResized)
    # cv2.waitKey(0)
    cv2.imwrite(img_save_path, imageResized)
    cv2.imwrite(map_save_path, mapResized)
    # print(img_path, map_path)

for file in tqdm(os.listdir(pathToVal)):
    file = file.split('.')[0]
    img_path = os.path.join(pathToVal, file + '.jpg')
    map_path = os.path.join(pathToValMap, file + '.png')
    img_save_path = os.path.join(pathToResizedImagesVal, file + '.jpg')
    map_save_path = os.path.join(pathToResizedMapsVal, file + '.png')
    imageResized = cv2.resize(cv2.imread(img_path), INPUT_SIZE, interpolation=cv2.INTER_AREA)
    mapResized = cv2.resize(cv2.imread(map_path), INPUT_SIZE, interpolation=cv2.INTER_AREA)
    # cv2.imshow('none', mapResized)
    # cv2.waitKey(0)
    cv2.imwrite(img_save_path, imageResized)
    cv2.imwrite(map_save_path, mapResized)
    # print(img_path, map_path)
