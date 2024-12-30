from torch.utils.tensorboard import SummaryWriter
import torch
from torch.nn.functional import mse_loss
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import VAE
from torch import optim
import os


# Create a writer to log metrics
writer = SummaryWriter(log_dir='./logs/autoencoder')

# Data preparation
transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
data_loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

# Model, optimizer, and loss
latent_dim = 64
autoencoder = VAE(latent_dim=latent_dim).cuda()
optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)

# Save directory for models
save_dir = './models'
os.makedirs(save_dir, exist_ok=True)

# Training loop
num_epochs = 30
for epoch in range(num_epochs):
    epoch_loss = 0
    for images, _ in data_loader:
        images = images.cuda()
        
        # Forward pass
        recon_images, z = autoencoder(images)
        
        # Compute reconstruction loss
        loss = mse_loss(recon_images, images)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(data_loader)
    
    # Log the loss
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}")
    writer.add_scalar('Loss/train', avg_loss, epoch)  # TensorBoard
    
    # Save models periodically
    if (epoch + 1) % 5 == 0:  # Save every 5 epochs
        torch.save(autoencoder.encoder.state_dict(), os.path.join(save_dir, f"encoder_epoch_{epoch+1}.pth"))
        torch.save(autoencoder.decoder.state_dict(), os.path.join(save_dir, f"decoder_epoch_{epoch+1}.pth"))
        print(f"Saved checkpoint at epoch {epoch+1}")
    
    # Log reconstructed images to TensorBoard
    if (epoch + 1) % 5 == 0:  # Log every 5 epochs
        with torch.no_grad():
            sample_images = images[:8]  # Take a batch of 8 images
            recon_images, _ = autoencoder(sample_images)
            writer.add_images('Original Images', sample_images, epoch)
            writer.add_images('Reconstructed Images', recon_images, epoch)

# Save final encoder and decoder
torch.save(autoencoder.encoder.state_dict(), os.path.join(save_dir, "encoder.pth"))
torch.save(autoencoder.decoder.state_dict(), os.path.join(save_dir, "decoder.pth"))
print("Encoder and decoder models saved successfully.")

# Close the writer
writer.close()
