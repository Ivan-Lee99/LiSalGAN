from generator import *
from discriminator import *
from data_loader import *
from constants import *
from torchvision.transforms import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch
import numpy as np


def show(img):
    i = img.numpy()
    i = np.transpose(i, (1, 2, 0))
    # print(img.shape)
    # s = np.array(pilImg)
    cv2.imshow('none', i)
    cv2.waitKey(0)


def cc(x, y):
    m = np.array([x, y])
    return np.corrcoef(m)


Val_data = DataLoader(pathToResizedImagesVal, batch_size=1)
model = Generator()
pretrained_dict = torch.load('./generator.pkl')
model.load_state_dict(pretrained_dict)
if torch.cuda.is_available():
    model.cuda()

for i in Val_data.num_batch:
    print(i)




