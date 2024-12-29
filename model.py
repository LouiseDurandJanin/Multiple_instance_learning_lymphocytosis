
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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
'''
Attention based MIL
'''
class Aggregation_attention(nn.Module):
    def __init__(self):
        super(Aggregation_attention, self).__init__()
        self.M = 60
        self.L = 128
        self.ATTENTION_BRANCHES = 1
        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.M),
            nn.ReLU(),
        )

        self.attention_V = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix V
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix U
            nn.Sigmoid()
        )

        self.attention_w = nn.Linear(self.L, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)

        self.classifier = nn.Sequential(
            nn.Linear(self.M*self.ATTENTION_BRANCHES, 1),
            nn.Sigmoid()
        )

    def forward(self, H):
        '''
        H must be size batch_size * emb_dim
        '''

        A_V = self.attention_V(H)  # KxL
        A_U = self.attention_U(H)  # KxL
        A = self.attention_w(A_V * A_U) # element wise multiplication # KxATTENTION_BRANCHES
        A = torch.transpose(A, 1, 0)  # ATTENTION_BRANCHESxK
        A = F.softmax(A, dim=1)  # softmax over K

        Z = torch.mm(A, H)  # ATTENTION_BRANCHESxM


        return Z

class BinaryClassifier(nn.Module):
    def __init__(self, input_size):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  
        self.fc2 = nn.Linear(128, 64)  
        self.fc3 = nn.Linear(64, 1)   

    def forward(self, x):
        x = F.relu(self.fc1(x))  
        x = F.relu(self.fc2(x))  
        x = torch.sigmoid(self.fc3(x))  
        return x
    
class Final_Model(nn.Module):
    def __init__(self, encoder, bag_aggregator, classifier):
        super(Final_Model, self).__init__()
        self.encoder = encoder
        self.bag_aggregator = bag_aggregator

        self.classifier = classifier

    def forward(self, list_images, feats):
        batch_features = []
        for patient_images in list_images:
            patient_features = []
            for image in patient_images:
                image = image.to(device)
                output = self.encoder(image.unsqueeze(0))[0]
                patient_features.append(output)
            # Aggregate features for each patient, e.g., by averaging
            patient_features = self.bag_aggregator(torch.stack(patient_features, dim=0).squeeze(1))
            batch_features.append(patient_features)
        batch_features_tensor = torch.stack(batch_features)

        # Size : (batch size, embedding dimension)

        if feats is not None:
            # Concatenate additional features
            p = torch.cat((batch_features_tensor.squeeze(1), feats), 1)

        out = self.classifier(p)
        return out