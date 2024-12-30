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

device = torch.device("cuda")
def sample(diffusion_model, decoder, num_samples, latent_dim, timesteps, 
           betas, alphas_cumprod, device):
    """Use the standard DDPM sampling formula."""
    # Precompute alpha, alpha_bar, etc.
    # betas: shape [timesteps]
    # alphas: shape [timesteps] = 1 - betas
    # alpha_bars: shape [timesteps] = product of alphas up to t

    alphas = 1.0 - betas
    alpha_bars = alphas_cumprod  # same as your code

    # Start from Gaussian noise in latent shape
    z_t = torch.randn(num_samples, latent_dim, 8, 8, device=device)
    
    for i in reversed(range(timesteps)):
        t_tensor = torch.full((num_samples,), i, device=device, dtype=torch.long)

        # 1. Predict noise
        epsilon_theta = diffusion_model(z_t, t_tensor)  # noise_pred

        # 2. Gather the actual alpha_t, alpha_bar_t, etc.
        alpha_t     = alphas[i]       # a scalar
        alpha_bar_t = alpha_bars[i]   # a scalar
        if i > 0:
            alpha_bar_t_prev = alpha_bars[i-1]
        else:
            alpha_bar_t_prev = torch.tensor(1.0, device=device)  # or 1 for alpha_bar(-1)

        # Because alpha_t and alpha_bar_t are scalars, expand to the shape of z_t
        alpha_t     = alpha_t.view(1,1,1,1)
        alpha_bar_t = alpha_bar_t.view(1,1,1,1)
        alpha_bar_t_prev = alpha_bar_t_prev.view(1,1,1,1)
        
        # 3. x_{t-1} formula
        #    The standard DDPM version:
        #    x_{t-1} = 1/sqrt(alpha_t) * ( x_t - (1 - alpha_t)/sqrt(1 - alpha_bar_t) * eps_theta ) + sigma_t * z
        #    where sigma_t^2 = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        
        one_over_sqrt_alpha_t = 1.0 / torch.sqrt(alpha_t)
        one_minus_alpha_t = (1.0 - alpha_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)
        
        # The "deterministic" part
        z_t = one_over_sqrt_alpha_t * (
            z_t - one_minus_alpha_t / sqrt_one_minus_alpha_bar_t * epsilon_theta
        )
        
        # The "stochastic" part: only add noise if i > 0
        if i > 0:
            beta_t = betas[i].view(1,1,1,1)
            # sigma_t^2 = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
            sigma_t = torch.sqrt(
                beta_t * (1.0 - alpha_bar_t_prev) / (1.0 - alpha_bar_t)
            )
            z_t = z_t + sigma_t * torch.randn_like(z_t)
    
    # After the loop, z_t is your final sample in latent space
    # If you want a global pooling, do so (though many VAE-based latents
    # would use the entire [C,H,W] shape). 
    # But your code does a mean pool:
    z_t = z_t.mean(dim=[2, 3])  # shape (batch_size, latent_dim)

    # Then decode
    generated_images = decoder(z_t)  # shape [B, 3, H, W]
    return generated_images



# Generate 8 images
# Initialize models
latent_dim = 64
diffusion_model = DiffusionUNet(latent_dim=latent_dim).to(device)
decoder = Decoder(latent_dim=latent_dim).to(device)

# Load pretrained weights
diffusion_model.load_state_dict(torch.load('./models_diffusion/diffusion_model_final.pth'))
decoder.load_state_dict(torch.load('./models/decoder_epoch_30.pth'))

diffusion_model.eval()
decoder.eval()

for name, param in diffusion_model.named_parameters():
    if torch.isnan(param).any():
        print(f"NaN in parameter {name}")

for name, param in decoder.named_parameters():
    if torch.isnan(param).any():
        print(f"NaN in parameter {name}")
        


betas = linear_beta_schedule(1000).to(device)
alphas, alphas_cumprod = get_alphas(betas)
generated_images = sample(
    diffusion_model, 
    decoder, 
    num_samples=8, 
    latent_dim=64, 
    timesteps=1000, 
    betas=betas,                 
    alphas_cumprod=alphas_cumprod.to(device),  
    device=device
)



import matplotlib.pyplot as plt
from torchvision.utils import make_grid

# Save the grid of generated images to a file
grid = make_grid(generated_images, nrow=4, normalize=True).permute(1, 2, 0).cpu()
plt.figure(figsize=(10, 5))
plt.imshow(grid)
plt.axis('off')
plt.title('Generated Images')

# Save the image to a file
output_path = './generated_images.png'
plt.savefig(output_path)
print(f"Generated images saved to {output_path}")