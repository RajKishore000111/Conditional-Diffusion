import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.utils import save_image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TIMESTEPS = 150
EPOCHS = 5
BATCH_SIZE = 128
IMG_SIZE = 32
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]


# 1. Scheduling the noise
betas = torch.linspace(1e-4, 0.02, TIMESTEPS).to(DEVICE)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)

def add_noise(x0, t, noise):
    sqrt_ab = alpha_bars[t].sqrt().view(-1, 1, 1, 1)
    sqrt_1mab = (1 - alpha_bars[t]).sqrt().view(-1, 1, 1, 1)
    return sqrt_ab * x0 + sqrt_1mab * noise


# 2. Defining the model

# Residual block with time and class conditioning


class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, emb_dim):

        super().__init__()

        # First convolution
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, padding=1
        )

        # Normalize feature maps
        self.norm1 = nn.GroupNorm(8, out_channels)

        # Second convolution
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, padding=1
        )

        self.norm2 = nn.GroupNorm(8, out_channels)

        # Convert time + class embedding to feature channels
        self.emb_proj = nn.Linear(emb_dim, out_channels)

        # Match input channels to output channels if necessary
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels, out_channels, 1
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, emb):

        # Save input for residual connection
        residual = self.shortcut(x)

        # First convolution
        h = self.conv1(x)
        h = self.norm1(h)

        # Add time and class conditioning
        emb = self.emb_proj(emb)
        emb = emb[:, :, None, None]

        h = h + emb
        h = F.relu(h)

        # Second convolution
        h = self.conv2(h)
        h = self.norm2(h)

        # Residual connection
        h = h + residual

        return F.relu(h)



# U-Net Denoiser

class SimpleDenoiser(nn.Module):

    def __init__(self, channels=64, num_classes=10):

        super().__init__()

        
        # TIME AND CLASS EMBEDDINGS
        

        self.time_embed = nn.Embedding(
            TIMESTEPS, channels
        )

        self.class_embed = nn.Embedding(
            num_classes, channels
        )

        
        # ENCODER
        

        # Input image
        self.enc1 = ResBlock(
            3, channels, channels
        )

        # Downsample: 
        self.down1 = nn.Conv2d(
            channels, channels * 2,
            kernel_size=3, stride=2, padding=1
        )

        self.enc2 = ResBlock(
            channels * 2, channels * 2, channels
        )

        # Downsample
        self.down2 = nn.Conv2d(
            channels * 2, channels * 4,
            kernel_size=3, stride=2, padding=1
        )

       
        # BOTTLENECK
        

        self.mid1 = ResBlock(
            channels * 4, channels * 4, channels
        )

        self.mid2 = ResBlock(
            channels * 4, channels * 4, channels
        )

        
        # DECODER
        

        # Upsample
        self.up2 = nn.Conv2d(
            channels * 4, channels * 2, 3, padding=1
        )

        
        self.dec2 = ResBlock(
            channels * 4, channels * 2, channels
        )

        # Upsample: 
        self.up1 = nn.Conv2d(
            channels * 2, channels, 3, padding=1
        )

        
        self.dec1 = ResBlock(
            channels * 2, channels, channels
        )

        
        # OUTPUT
       

        self.conv_out = nn.Conv2d(
            channels, 3, 3, padding=1
        )

    def forward(self, x, t, y):

        
        # EMBEDDINGS
        

        # Combine timestep and class information
        temb = self.time_embed(t)
        cemb = self.class_embed(y)

        emb = temb + cemb

        
        # ENCODER
        

        # First encoder level
        e1 = self.enc1(x, emb)

        # Downsample
        d1 = self.down1(e1)

        # Second encoder level
        e2 = self.enc2(d1, emb)

        # Downsample again
        d2 = self.down2(e2)

        
        # BOTTLENECK
        

        mid = self.mid1(d2, emb)
        mid = self.mid2(mid, emb)

        
        # DECODER
        

        # Upsample to the second encoder's resolution
        u2 = F.interpolate(
            mid,
            size=e2.shape[2:],
            mode="nearest"
        )

        u2 = self.up2(u2)

        # Skip connection from encoder level 2
        u2 = torch.cat([u2, e2], dim=1)

        u2 = self.dec2(u2, emb)

        # Upsample to the first encoder's resolution
        u1 = F.interpolate(
            u2,
            size=e1.shape[2:],
            mode="nearest"
        )

        u1 = self.up1(u1)

        # Skip connection from encoder level 1
        u1 = torch.cat([u1, e1], dim=1)

        u1 = self.dec1(u1, emb)

        
        # PREDICT NOISE
        

        return self.conv_out(u1)


# 3. Training on CIFAR 10

def train():

    transform = T.Compose([T.ToTensor(), T.Normalize((0.5,) * 3, (0.5,) * 3)])
    dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = SimpleDenoiser().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            t = torch.randint(0, TIMESTEPS, (images.size(0),), device=DEVICE)
            noise = torch.randn_like(images)
            noisy_images = add_noise(images, t, noise)

            predicted_noise = model(noisy_images, t, labels)
            loss = F.mse_loss(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{EPOCHS} - loss: {total_loss / len(loader):.4f}")

    return model



# 4. Image generation


@torch.no_grad()
def generate(model, class_name, n_images=5):
    """Generates n_images of the given class name by reversing the noise process step by step."""
    class_idx = CLASS_NAMES.index(class_name)
    labels = torch.full((n_images,), class_idx, dtype=torch.long, device=DEVICE)

    model.eval()
    x = torch.randn(n_images, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)

    for t_step in reversed(range(TIMESTEPS)):
        t = torch.full((n_images,), t_step, dtype=torch.long, device=DEVICE)

        predicted_noise = model(x, t, labels)

        alpha = alphas[t_step]
        alpha_bar = alpha_bars[t_step]
        beta = betas[t_step]

        mean = (x - beta / (1 - alpha_bar).sqrt() * predicted_noise) / alpha.sqrt()

        if t_step > 0:
            x = mean + beta.sqrt() * torch.randn_like(x)
        else:
            x = mean

    return (x.clamp(-1, 1) + 1) / 2



# 5. RUN IT


if __name__ == "__main__":
    trained_model = train()

    os.makedirs("generated", exist_ok=True)

    class_to_generate = "airplane"
    how_many = 5

    images = generate(trained_model, class_to_generate, n_images=how_many)
    for i, img in enumerate(images):
        save_image(img, f"generated/{class_to_generate}_{i}.png")
    save_image(images, f"generated/{class_to_generate}_grid.png", nrow=how_many)

    print(f"Saved {how_many} images of '{class_to_generate}' to the 'generated/' folder")
