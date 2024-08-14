import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from HVAE import HVAE
import ssl
# Hyperparameters and settings
BATCH_SIZE = 128
EPOCHS = 2000
LEARNING_RATE = 1e-3
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Data loading
ssl._create_default_https_context = ssl._create_unverified_context
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


# Model setup
class Args:
    def __init__(self):
        self.z_dim = 20
        self.n_IS = 1  # For NLL calculation
        self.n_batch_test = BATCH_SIZE
        self.n_gen_samples = 16  # For generating samples
        self.K = 5  # Number of leapfrog steps
        self.init_lf = 0.01  # Initial leapfrog step size
        self.max_lf = 0.1  # Maximum leapfrog step size
        self.vary_eps = 'false'  # Whether to vary epsilon across layers
        self.temp_method = 'free'  # Tempering method: 'free', 'fixed', or 'none'
        self.init_alpha = 0.9  # Initial alpha for free tempering
        self.init_T_0 = 2.0  # Initial temperature for fixed tempering


args = Args()
avg_logit = 0.0  # You may want to calculate this based on your data
model = HVAE(args, avg_logit).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# Training loop
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(DEVICE)
        optimizer.zero_grad()
        try:
            elbo = model.get_elbo(data, args)
            loss = -elbo  # Negative ELBO is the loss we want to minimize

            # Check for NaNs in the loss
            if torch.isnan(loss).any():
                print(f"NaN detected in loss at epoch {epoch}, batch {batch_idx}. Stopping training.")
                break

            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        except Exception as e:
            print(f"Error in batch {batch_idx}: {str(e)}")
            print(f"Input shape: {data.shape}")
            print(f"Input min: {data.min()}, max: {data.max()}")
            raise

    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader):.4f}')


# Visualization function
def visualize_reconstructions(epoch):
    model.eval()
    with torch.no_grad():
        sample = next(iter(train_loader))[0][:8].to(DEVICE)
        recon = model._gen_network(model._inf_network(sample)[0])
        recon = torch.sigmoid(recon)

        fig, axes = plt.subplots(2, 8, figsize=(16, 4))
        for i in range(8):
            axes[0, i].imshow(sample[i].cpu().squeeze(), cmap='gray')
            axes[0, i].axis('off')
            axes[1, i].imshow(recon[i].cpu().squeeze(), cmap='gray')
            axes[1, i].axis('off')
        plt.tight_layout()
        plt.savefig(f'reconstruction_epoch_{epoch}.png')
        plt.close()


# Main training loop
for epoch in range(1, EPOCHS + 1):
    train(epoch)
    if epoch % 10 == 0:
        visualize_reconstructions(epoch)

# Generate samples after training
model.eval()
with torch.no_grad():
    sample = model.get_samples(args)
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i in range(16):
        ax = axes[i // 4, i % 4]
        ax.imshow(sample[i].cpu().squeeze(), cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('generated_samples.png')
    plt.close()

print("Training complete. Reconstructions and generated samples have been saved.")