import torch
import matplotlib.pyplot as plt
from train_gan import Generator  # Import Generator class from generator.py

# Load the trained generator
generator = Generator()
generator.load_state_dict(torch.load('generator.pth'))
generator.eval()  # Set the model to evaluation mode

# Generate synthetic images
num_samples = 5  # Number of synthetic images to generate
noise = torch.randn(num_samples, 100)  # Generate random noise
with torch.no_grad():
    generated_images = generator(noise).cpu()  # Generate images from noise

# Visualize the generated images
plt.figure(figsize=(15, 5))
for i in range(num_samples):
    plt.subplot(1, num_samples, i + 1)
    generated_image = generated_images[i].squeeze().numpy()  # Convert tensor to numpy array
    plt.imshow(generated_image, cmap='gray')
    plt.title(f"Generated Image {i+1}")
    plt.axis('off')
plt.show()
