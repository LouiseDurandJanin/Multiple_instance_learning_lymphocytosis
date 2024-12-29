from data import augment_patient_images
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
import os
from tqdm import tqdm
from torchvision import transforms
from autoencoder import AutoEncoder
from data import ImagesDataset

# Perform data augmentation
root_dir = 'dlmi-lymphocytosis-classification/trainset'
save_dir = 'augmented_trainset'
'''
patients = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

for patient in patients:
    patient_folder = os.path.join(root_dir, patient)
    patient_save_folder = os.path.join(save_dir, patient)
    augment_patient_images(patient_folder, patient_save_folder, angles=[90, 180, 270])
'''
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Assuming this is the input size for your autoencoder
    transforms.ToTensor(),
])

# Instantiate the dataset
dataset = ImagesDataset(root_dir=save_dir, transform=transform)
valid_size = 0.2 


num_samples = len(dataset)
num_valid = int(num_samples * valid_size)
num_train = num_samples - num_valid


dataset_train, dataset_valid = random_split(dataset, [num_train, num_valid])


train_dataloader = DataLoader(dataset_train, batch_size=128, shuffle=True)
valid_dataloader = DataLoader(dataset_valid, batch_size=128, shuffle=False)

# Training of the Autoencoder

# Instanciate the autoencoder
autoencoder = AutoEncoder(emb_dim=60)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
autoencoder = autoencoder.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=1e-4)
num_epochs = 100


best_val_loss=10000

# Training loop 
for epoch in range(num_epochs):
    for iter, images in tqdm(enumerate(train_dataloader)):
        images = images.to(device)  # Ensure images are on the correct device
        optimizer.zero_grad()
        output = autoencoder(images)
        loss = criterion(output, images)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    if epoch %10==0:
      val_loss = 0.0
      with torch.no_grad():  # Pas besoin de calculer les gradients pour la validation
            for images in valid_dataloader:
                images = images.to(device)
                output = autoencoder(images)
                loss = criterion(output, images)
                val_loss += loss.item()
      val_loss /= len(valid_dataloader)

      if val_loss < best_val_loss:
            print(f'Validation loss decreased ({best_val_loss:.4f} --> {val_loss:.4f}). Saving model ...')
            best_val_loss = val_loss
            torch.save(autoencoder.state_dict(), "./autoencoder_model.pt")
