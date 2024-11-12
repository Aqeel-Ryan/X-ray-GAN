# X-ray-GAN
This code implements a Generative Adversarial Network (GAN) for generating synthetic X-ray images. Here's a comprehensive breakdown:

## Core Components

**Dataset Handling**
- Custom `XRayDataset` class for loading X-ray images and labels
- Image preprocessing using PIL and torchvision transforms
- Grayscale image conversion and normalization to (-0.5, 0.5)

**Architecture**
- Generator: Transforms 100-dimensional noise into 64x64 grayscale images
- Discriminator: Classifies images as real or fake with a binary output
- Both networks use convolutional layers with batch normalization

**Training Framework**
- Adam optimizer with different learning rates (0.001 for generator, 0.0003 for discriminator)
- Binary Cross Entropy (BCE) loss function
- Early stopping mechanism with patience of 10 epochs
- Model checkpointing for best performing states

## Technical Techniques Used

**Deep Learning Components**
- Convolutional layers (Conv2d, ConvTranspose2d)
- Batch Normalization
- ReLU and LeakyReLU activations
- Sigmoid activation for binary classification
- Linear layers for dimensionality transformation

**Training Optimizations**
- GPU acceleration when available
- Multi-worker data loading
- Early stopping to prevent overfitting
- Model state saving and loading
- Loss tracking and visualization

**Image Processing**
- Resizing to 64x64 pixels
- Grayscale conversion
- Normalization
- Tensor transformations

**Generation Process**
- Random noise sampling
- Batch processing
- Visualization using matplotlib
- Model evaluation mode for inference
