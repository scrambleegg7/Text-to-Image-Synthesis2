import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from datetime import datetime
import model
from dataset import TextDataset

import matplotlib.pyplot as plt   
import numpy as np   



def proc():

    imageSize=64

    image_transform = transforms.Compose([
        transforms.RandomCrop(imageSize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0, 0, 0), (1, 1, 1))
    ])

    dataroot = "/home/donchan/Documents/DATA/CULTECH_BIRDS/CUB_200_2011/"

    train_dataset = TextDataset(dataroot, transform=image_transform)

    data = next( iter( train_dataset ) )

    real_cpu, text_embedding, captions_text = data
    batch_size = real_cpu.size(0)
    #text_embedding = Variable(text_embedding)

    print(real_cpu.shape)
    print(text_embedding.shape)

    print(captions_text)
    image = real_cpu.data.numpy()
    image = np.transpose( image, (1,2,0) )
    plt.imshow(image)
    plt.show()

def main():
    proc()

if __name__ == "__main__":

    main()
    pass