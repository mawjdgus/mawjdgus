
from __future__ import print_function
import os
import random

import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from torch.nn.modules.activation import LeakyReLU
from torch.nn.modules.batchnorm import BatchNorm1d
from torch import optim
import torchvision
from torchvision import transforms
from torchvision.datasets import utils
from torchvision.transforms.transforms import CenterCrop

PATH = './myenv/bin/GANs/VanillaGANs/data/'



manualSeed = 1211
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

workers = 2
BATCH_SIZE = 32
image_size = 28*28
nc = 1
nz = 100
ngf = 128
ndf = 128
out_dim = 1
num_epochs = 5
lr = 0.0002
beta1 = 0.5
ngpu = 1

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0 )else "cpu")
# dataset = torchvision.datasets.ImageFolder(root=PATH,
# transform=transforms.Compose([
#     transforms.Resize(image_size),
#     transforms.CenterCrop(image_size),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,),(0.5,)),
# ]))




mnist_train = torchvision.datasets.MNIST(PATH, train=True, transform=transforms.ToTensor(),download=True)
mnist_test = torchvision.datasets.MNIST(PATH, train=False,transform=transforms.ToTensor(),download=True)

train_iter = torch.utils.data.DataLoader(mnist_train,
batch_size=BATCH_SIZE,
shuffle=True,
num_workers=workers)

test_iter = torch.utils.data.DataLoader(mnist_test,
batch_size=BATCH_SIZE,
shuffle=True,
num_workers=workers)




class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()

        self.fc0 = nn.Sequential(
            nn.Linear(z_dim, ngf), # 128
            nn.LeakyReLU(0.2)           
        )

        self.fc1 = nn.Sequential(
            nn.Linear(ngf, ngf*2), # 256
            nn.LeakyReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(ngf*2, ngf*4), # 512
            nn.LeakyReLU(0.2)
        )

        self.fc3 = nn.Sequential(
            nn.Linear(ngf*4, ngf*8), # 1024
            nn.LeakyReLU(0.2)
        )

        self.fc4 = nn.Sequential(
            nn.Linear(ngf*8, image_size), # 784
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc0(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = x.view(-1,1,28,28)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.fc0 = nn.Sequential(
            nn.Linear(image_size, ndf*8), # 1024
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(ndf*8, ndf*4), # 512
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(ndf*4, ndf*2), # 512
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.fc3 = nn.Sequential(
            nn.Linear(ndf*2, ndf*1), # 512
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.fc4 = nn.Sequential(
            nn.Linear(ndf*1, out_dim), # 512
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1,1,28*28).to(device)
        x = self.fc0(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

