import torch
from torch import nn, optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset, random_split, Subset
import torch.nn.functional as F
from torchvision.models import resnet50
import os
import random
import torchvision.transforms.functional as TF
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, accuracy_score, classification_report



class Encoder(nn.Module):
  def __init__(self, emb_dim):
    super(Encoder, self).__init__()
    ndf = 4  # number of feature maps
    nc = 3  # number of channels
    self.conv1 = nn.Conv2d(nc, ndf * 4, kernel_size=3, padding=1)
    self.relu1 = nn.ReLU(True)
    self.pool1 = nn.MaxPool2d(2, return_indices=True)

    self.conv2 = nn.Conv2d(ndf *4, ndf * 8, kernel_size=3, padding=1)
    self.relu2 = nn.ReLU(True)
    self.pool2 = nn.MaxPool2d(2, return_indices=True)
    self.conv3 = nn.Conv2d(ndf *8, ndf * 16, kernel_size=3, padding=1)
    self.relu3 = nn.ReLU(True)
    self.pool3 = nn.MaxPool2d(2, return_indices=True)
    self.linear1 = nn.Linear(12544, emb_dim*3)
    self.linear2 = nn.Linear(emb_dim * 3, emb_dim)

  def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x, indices1 = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x, indices2 = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x, indices3 = self.pool3(x)
        x = x.reshape(x.shape[0], -1)
        x = F.leaky_relu(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))

        return x, [indices1, indices2, indices3]


class Decoder(nn.Module):
  def __init__(self, emb_dim):
    super(Decoder, self).__init__()
    ndf =4  # number of feature maps
    nc = 3  # number of channels
    self.unpool1 = nn.MaxUnpool2d(2)
    self.deconv1 = nn.ConvTranspose2d(ndf * 16, ndf * 8, kernel_size=3, padding=1)
    self.relu2 = nn.ReLU(True)

    self.unpool2 = nn.MaxUnpool2d(2)
    self.deconv2 = nn.ConvTranspose2d(ndf * 8, ndf*4, kernel_size=3, padding=1)
    self.relu3 = nn.ReLU(True)

    self.unpool3 = nn.MaxUnpool2d(2)
    self.deconv3 = nn.ConvTranspose2d(ndf * 4, nc, kernel_size=3, padding=1)
    self.sigmoid = nn.Sigmoid()

    self.linear2 = nn.Linear(emb_dim*3, 12544)
    self.linear1 = nn.Linear(emb_dim, emb_dim * 3)
    self.relu1 = nn.ReLU(True)

  def forward(self,x, indices1, indices2, indices3):
    x = F.leaky_relu(self.linear1(x))
    x = self.linear2(x)
    x = x.reshape(x.shape[0],4 * 16, 14, 14)
    x = self.unpool1(x, indices3)
    x = self.deconv1(x)

    x = self.relu2(x)

    x = self.unpool2(x, indices2)
    x = self.deconv2(x)
    x = self.relu3(x)
    
    x = self.unpool3(x, indices1)
    x = self.deconv3(x)
    x = self.relu1(x)
    return x

class AutoEncoder(nn.Module):
    def __init__(self, emb_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(emb_dim)
        self.decoder = Decoder(emb_dim)

    def forward(self, x):
        emb, pool_indices = self.encoder(x)
        x_hat = self.decoder(emb, pool_indices[0], pool_indices[1], pool_indices[2])
        return x_hat