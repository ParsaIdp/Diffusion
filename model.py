import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Encoder(nn.Module):
    def __init__(self, latent_dim = 64, input_dim = 3):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, latent_dim)
        )
    def forward(self, x):
        return self.encoder(x)
    
class Decoder(nn.Module):
    def __init__(self, latent_dim = 64, output_dim = 3):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (256, 8, 8)),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, output_dim, 3, stride=1, padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.decoder(x)

class VAE(nn.Module):
    def __init__(self, latent_dim = 64, input_dim = 3):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim, input_dim)
        self.decoder = Decoder(latent_dim, input_dim)
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def linear_beta_schedule(timesteps):
    beta_start = 1e-4
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def get_alphas(betas):
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return alphas, alphas_cumprod   

def forward_diffusion_sample(z0, t, alphas_cumprod, device):
    noise = torch.randn_like(z0).to(device)
    alpha_t = alphas_cumprod[t].view(-1, 1, 1, 1)
    sqrt_alpha_t = torch.sqrt(alpha_t)
    sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
    z_t = sqrt_alpha_t * z0 + sqrt_one_minus_alpha_t * noise
    return z_t, noise

class DiffusionUNet(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        self.net = nn.Sequential(
            nn.Conv2d(latent_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, latent_dim, kernel_size=3, padding=1),
        )

    def forward(self, z_t, t):
        t_emb = self.time_embed(t.float().view(-1, 1)).unsqueeze(-1).unsqueeze(-1)
        return self.net(z_t + t_emb)










