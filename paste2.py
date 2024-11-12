import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
import pickle

# Paths
data_dir = 'D:/work/A2M/MediGAN/images_001/images'  # Change this to your actual images directory
csv_file = 'D:/work/A2M/MediGAN/Data_Entry_2017/Data_Entry_2017.csv'  # Change this to your actual CSV file

# Load the CSV file
labels_df = pd.read_csv(csv_file)

# Filter the DataFrame to only include existing images
existing_images = set(os.listdir(data_dir))
labels_df = labels_df[labels_df['Image Index'].isin(existing_images)]

# Custom Dataset Class
class XRayDataset(Dataset):
    def __init__(self, data_dir, labels_df, transform=None):
        self.data_dir = data_dir
        self.labels_df = labels_df
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.labels_df.iloc[idx, 0])
        image = Image.open(img_name).convert('L')  # Convert to grayscale
        if self.transform:
            image = self.transform(image)
        return image

# Image transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Prepare Dataset and DataLoader
dataset = XRayDataset(data_dir=data_dir, labels_df=labels_df, transform=transform)

# GAN Models
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(100, 8*8*256),
            nn.ReLU(True),
            nn.Unflatten(1, (256, 8, 8)),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.main(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(8*8*256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.main(x)
        return x

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize models
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Loss and Optimizer
criterion = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=0.0002)  # Increased learning rate for G
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002)  # Decreased learning rate for D

# Save paths
save_path_g = 'generator.pth'
save_path_d = 'discriminator.pth'
losses_path = 'losses.pkl'

# Early stopping parameters
min_epochs = 70
patience = 10
best_loss = float('inf')
epochs_no_improve = 0

# Lists to store loss values
losses_g = []
losses_d = []

if __name__ == '__main__':
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

    num_epochs = 150
    try:
        for epoch in range(num_epochs):
            generator.train()
            discriminator.train()
            
            epoch_loss_g = 0.0
            epoch_loss_d = 0.0
            
            for i, data in enumerate(dataloader):
                real_data = data.to(device)
                batch_size = real_data.size(0)
                labels_real = torch.ones(batch_size, 1).to(device)
                labels_fake = torch.zeros(batch_size, 1).to(device)

                # Train Discriminator
                optimizer_d.zero_grad()
                output_real = discriminator(real_data).view(-1, 1)
                loss_real = criterion(output_real, labels_real)
                
                noise = torch.randn(batch_size, 100).to(device)
                fake_data = generator(noise)
                output_fake = discriminator(fake_data.detach()).view(-1, 1)
                loss_fake = criterion(output_fake, labels_fake)
                
                loss_d = loss_real + loss_fake
                loss_d.backward()
                optimizer_d.step()

                # Train Generator
                optimizer_g.zero_grad()
                output_fake = discriminator(fake_data).view(-1, 1)
                loss_g = criterion(output_fake, labels_real)
                loss_g.backward()
                optimizer_g.step()

                epoch_loss_g += loss_g.item()
                epoch_loss_d += loss_d.item()

                if i % 50 == 0:
                    print(f"Epoch [{epoch}/{num_epochs}] Batch {i}/{len(dataloader)} \
                          Loss D: {loss_d.item()}, Loss G: {loss_g.item()}")
            
            # Average loss for the epoch
            epoch_loss_g /= len(dataloader)
            epoch_loss_d /= len(dataloader)
            
            losses_g.append(epoch_loss_g)
            losses_d.append(epoch_loss_d)

            # Early stopping check
            if epoch >= min_epochs:
                if epoch_loss_g < best_loss:
                    best_loss = epoch_loss_g
                    epochs_no_improve = 0
                    torch.save(generator.state_dict(), save_path_g)
                    torch.save(discriminator.state_dict(), save_path_d)
                    print(f"Models saved at epoch {epoch+1} with generator loss: {best_loss}")
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                torch.save(generator.state_dict(), save_path_g)
                torch.save(discriminator.state_dict(), save_path_d)
                print(f"Models saved at epoch {epoch+1}")
    except KeyboardInterrupt:
        print("Training interrupted. Saving models and loss data...")
    finally:
        # Save models and loss data
        torch.save(generator.state_dict(), save_path_g)
        torch.save(discriminator.state_dict(), save_path_d)
        with open(losses_path, 'wb') as f:
            pickle.dump({'losses_g': losses_g, 'losses_d': losses_d}, f)
        print("Models and loss data saved.")

