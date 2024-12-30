import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import numpy as np
import os
from model import linear_beta_schedule, get_alphas, Encoder, Decoder, DiffusionUNet
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ================================
# Step 1: Set up Configurations
# ================================
# Parameters
latent_dim = 64
timesteps = 1000
batch_size = 128
num_epochs = 20
learning_rate = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Save directories
save_dir = './models_diffusion'
os.makedirs(save_dir, exist_ok=True)

# TensorBoard
writer = SummaryWriter(log_dir='./logs/diffusion')


def linear_beta_schedule(timesteps):
    beta_start = 1e-4
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps).to(device)

def get_alphas(betas):
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return alphas, alphas_cumprod

betas = linear_beta_schedule(timesteps)
alphas, alphas_cumprod = get_alphas(betas)

encoder = Encoder(latent_dim=latent_dim).to(device)
decoder = Decoder(latent_dim=latent_dim).to(device)

encoder.load_state_dict(torch.load('./models/encoder_epoch_30.pth'))
decoder.load_state_dict(torch.load('./models/decoder_epoch_30.pth'))

encoder.eval()  # Freeze encoder
decoder.eval()  # Freeze decoder

print("Encoder and decoder models loaded.")


transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

latent_data = []
for images, _ in data_loader:
    with torch.no_grad():
        z = encoder(images.to(device))
        latent_data.append(z.cpu())
latent_data = torch.cat(latent_data)
latent_loader = DataLoader(latent_data, batch_size=batch_size, shuffle=True)

def forward_diffusion_sample(z0, t, alphas_cumprod):
    noise = torch.randn_like(z0).to(device)
    alpha_t = alphas_cumprod[t].view(-1, 1, 1, 1)
    sqrt_alpha_t = torch.sqrt(alpha_t)
    sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
    z_t = sqrt_alpha_t * z0 + sqrt_one_minus_alpha_t * noise
    return z_t, noise

def diffusion_loss(model, z0, t, alphas_cumprod):
    z_t, noise = forward_diffusion_sample(z0, t, alphas_cumprod)
    noise_pred = model(z_t, t)
    return F.mse_loss(noise_pred, noise)


diffusion_model = DiffusionUNet(latent_dim=latent_dim).to(device)
optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=learning_rate)

print("Diffusion model initialized.")

for epoch in range(num_epochs):
    epoch_loss = 0
    for z0 in latent_loader:  # Latent space representations
        z0 = z0.to(device)
        t = torch.randint(0, timesteps, (z0.shape[0],), device=device).long()
        
        optimizer.zero_grad()
        loss = diffusion_loss(diffusion_model, z0, t, alphas_cumprod)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(latent_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}")
    writer.add_scalar('Loss/train', avg_loss, epoch)

    if (epoch + 1) % 5 == 0:
        torch.save(diffusion_model.state_dict(), os.path.join(save_dir, f"diffusion_model_epoch_{epoch+1}.pth"))
        print(f"Saved diffusion model checkpoint at epoch {epoch+1}.")


torch.save(diffusion_model.state_dict(), os.path.join(save_dir, "diffusion_model_final.pth"))
print("Training complete. Final diffusion model saved.")

writer.close()




