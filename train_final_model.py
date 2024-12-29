from data import augment_patient_images, PatientImagesDataset, preprocess_dataframe
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
from model import Aggregation_attention, BinaryClassifier, Final_Model
from autoencoder import Encoder


clinical_annotation_dir = 'dlmi-lymphocytosis-classification/clinical_annotation.csv'
augmented_trainset_dir = 'dlmi-lymphocytosis-classification/trainset'
# Import preprocessed dataframes
df_train, df_test, target, target_test = preprocess_dataframe(clinical_annotation_dir=clinical_annotation_dir)

# Batch size for the training of the final model 
### TO ADJUST ###
batch_size = 8 

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
dataset = PatientImagesDataset(
    root_dir=augmented_trainset_dir,
    clinical_data=df_train, labels=target,
    transform=transform
)

# Split the dataset into train and validation using stratify
valid_size = 0.2
train_indices, valid_indices = train_test_split(
    range(len(dataset)),
    test_size=valid_size,
    stratify=target,
    random_state=42
)
# Function to deal with folders of different size per patient
def custom_collate_fn(batch):
    images = [item[0] for item in batch]# This will be a list of tensors
    clinical_infos = torch.stack([item[1] for item in batch])
    labels = torch.stack([item[2] for item in batch])
    return images, clinical_infos, labels

dataset_train = Subset(dataset, train_indices)

dataset_valid = Subset(dataset, valid_indices)
# Créez les DataLoaders pour les ensembles d'entraînement et de validation
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
dataloader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

autoencoder_dir = "autoencoder_model.pt"
state_dict = torch.load(autoencoder_dir)
encoder = Encoder(emb_dim=60)
encoder_state_dict = {k.partition('encoder.')[2]: v for k, v in state_dict.items() if k.startswith('encoder.')}

# Load the filtered parameters to the encoder 
encoder.load_state_dict(encoder_state_dict)

bag_aggregator = Aggregation_attention()
classifier = BinaryClassifier(input_size = 63)
model = Final_Model(encoder, bag_aggregator, classifier)

# Instanciate Binary Cross Entropy Loss
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=0.0005)

# Move model to GPU if available
print(device)
model.to(device)

# Training Loop
num_epochs = 100
iteration = 0
best_val_loss = float('inf')
best_accu = 0
i=0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch in dataloader_train:
        images, clinical_data, labels = batch

        # Zero the parameter gradients
        optimizer.zero_grad()


        # Forward pass
        outputs = model(images, clinical_data.to(device))


        # Calculate loss
        loss = criterion(outputs, labels.view(-1, 1).type_as(outputs))

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader_train.dataset)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
    if (epoch) % 10 == 0:  # Every 10 epochs
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in dataloader_valid:
                images, clinical_data, labels = batch

                outputs = model(images, clinical_data.to(device))
                preds = (outputs).round()  

                all_preds.extend(preds.view(-1).cpu().numpy())
                all_labels.extend(labels.view(-1).cpu().numpy())

        # Calculate balanced accuracy
        balanced_acc = balanced_accuracy_score(all_labels, all_preds)
        if balanced_acc > best_accu:
            best_accu = balanced_acc
            nom_modele = "modele_{}.pt".format(i)
            i+=1
            torch.save(model.state_dict(), nom_modele)

        accuracy = accuracy_score(all_labels, all_preds)
        print(classification_report(all_labels, all_preds))
        print(f'Balanced Accuracy on Validation Set: {balanced_acc:.4f}')
        print(f'Accuracy on Validation Set: {accuracy:.4f}')