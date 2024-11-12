import pickle
import matplotlib.pyplot as plt

# Path to the saved losses
losses_path = 'losses.pkl'

# Load the loss values
with open(losses_path, 'rb') as f:
    losses = pickle.load(f)
    losses_g = losses['losses_g']
    losses_d = losses['losses_d']

# Plotting the loss values
plt.figure(figsize=(10, 5))
plt.plot(losses_g, label='Generator Loss')
plt.plot(losses_d, label='Discriminator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Generator and Discriminator Loss During Training')
plt.show()
