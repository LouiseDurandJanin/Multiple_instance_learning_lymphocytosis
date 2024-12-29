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

def get_year(x):
  if '/' in x:
    return int(x.split('/')[-1])
  elif '-' in x:
    return int(x.split('-')[-1])

def preprocess_dataframe(clinical_annotation_dir):
    # Clinical csv

    clinical_annotation = pd.read_csv(clinical_annotation_dir, index_col=1)
    clinical_annotation.drop(clinical_annotation.columns[0], axis=1, inplace=True)

    # Preprocessing on structured data
    dict_to_encode = {'M': 0, "F":1, "f":1}
    clinical_annotation.head()
    clinical_annotation['GENDER'] = clinical_annotation['GENDER'].replace(dict_to_encode).astype(float)
    clinical_annotation['YEAR'] = clinical_annotation['DOB'].apply(get_year)
    clinical_annotation['AGE'] = 2023 - clinical_annotation['YEAR']
    clinical_annotation.drop(['DOB', 'YEAR'], axis=1, inplace=True)
    df_test = clinical_annotation[clinical_annotation['LABEL'] == -1]
    df_train = clinical_annotation[clinical_annotation['LABEL'] != -1]
    target = df_train['LABEL']
    target_test = df_test['LABEL']
    df_train.drop(['LABEL'], axis=1, inplace=True)
    df_test.drop(['LABEL'], axis=1, inplace=True)
    return df_train, df_test, target, target_test


# Functiton that downsample to size 112 * 112
def downsample_2d(X):
    if X.dim() == 3:
        X = X.unsqueeze(0)
    channels = X.shape[1]
    kernel = torch.ones((channels, 1, 3, 3)) / 9.0  
    X = F.conv2d(X, kernel, groups=channels, stride=2, padding=1)
    X = X.squeeze(0)
    return X

# Image Dataset for the pre training of the feature extractor
class ImagesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        patients = [os.path.join(root_dir, name) for name in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, name))]

        for patient_folder in patients:
            for image_file in os.listdir(patient_folder):
                full_path = os.path.join(patient_folder, image_file)
                if os.path.isfile(full_path):
                    self.image_paths.append(full_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        #print(image.size)
        if self.transform:
            image = self.transform(image)
            #print(image.shape)
            image = downsample_2d(image)

        return image
    
class PatientImagesDataset(Dataset):
    def __init__(self, root_dir, clinical_data, labels, transform=None):
        """
        Args:
            root_dir (string): Directory with all the patient folders.
            patient_id (list of str) : To distinguish between train and valid
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir

        self.clinical_data = clinical_data
        self.transform = transform
        self.patients = [os.path.join(root_dir, name) for name in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, name))]

        self.labels = labels

    def __len__(self):
        return len(self.patients)


    def __getitem__(self, idx):
            patient_folder = self.patients[idx]
            patient_id = os.path.basename(patient_folder)
            image_files = [os.path.join(patient_folder, name) for name in os.listdir(patient_folder) ]
            images = []
            for image_file in image_files:
                # Load the image
                image = Image.open(image_file).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                    image = downsample_2d(image)

                images.append(image)

            # Get clinical data for this patient
            clinical_info = self.clinical_data.loc[patient_id].values
            clinical_info = torch.tensor(clinical_info, dtype=torch.float)
            label = self.labels.loc[patient_id]
            label = torch.tensor(label, dtype=torch.float)

            return images, clinical_info,label


# Function to deal with folders of different size per patient
def custom_collate_fn(batch):
    images = [item[0] for item in batch]# This will be a list of tensors
    clinical_infos = torch.stack([item[1] for item in batch])
    labels = torch.stack([item[2] for item in batch])
    return images, clinical_infos, labels

# To augment the images dataset with specific rotation 
def augment_patient_images(patient_folder, save_folder, angles):
    # Create file if needed
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for image_name in os.listdir(patient_folder):
        if image_name.endswith((".png", ".jpg", ".jpeg")): 
            image_path = os.path.join(patient_folder, image_name)
            image = Image.open(image_path)
            
            # Perform rotation
            for angle in angles:
                rotated_image = transforms.functional.rotate(image, angle)
                
  
                save_path = os.path.join(save_folder, f"{os.path.splitext(image_name)[0]}_rotated_{angle}.png")
                

                rotated_image.save(save_path)