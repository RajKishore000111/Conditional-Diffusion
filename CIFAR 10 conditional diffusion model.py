"""
Simple Conditional Diffusion Model for CIFAR-10

"""

# imports
import os                                  # for creating the output folder
import torch                               # core PyTorch
import torch.nn as nn                      # layers
import torch.nn.functional as F            # activation functions etc.
import torchvision                         # gives us the CIFAR-10 dataset
import torchvision.transforms as T         # image preprocessing
from torch.utils.data import DataLoader    # batches the dataset
from torchvision.utils import save_image   # saves tensors as PNG files

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use GPU if available
TIMESTEPS = 300          # number of noise steps (smaller = faster, less sharp)
EPOCHS = 5            # how many passes over the training data
BATCH_SIZE = 128         # images per training batch
IMG_SIZE = 32            # CIFAR-10 images are 32x32
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]


# 1. Scheduling the noise

# betas: how much noise gets added at each of the TIMESTEPS steps
betas = torch.linspace(1e-4, 0.02, TIMESTEPS).to(DEVICE)
alphas = 1.0 - betas                                               # fraction of "signal" kept each step
alpha_bars = torch.cumprod(alphas, dim=0)                          # cumulative product of alphas up to step t, shall the original portion be reduced to 1-2 %??


def add_noise(x0, t, noise):

    sqrt_ab = alpha_bars[t].sqrt().view(-1, 1, 1, 1)               # signal-scale factor, reshaped for broadcasting
    sqrt_1mab = (1 - alpha_bars[t]).sqrt().view(-1, 1, 1, 1)       # noise-scale factor, reshaped for broadcasting
    return sqrt_ab * x0 + sqrt_1mab * noise                        # weighted mix of clean image and noise


# 2. Defining the model


class SimpleDenoiser(nn.Module):

    def __init__(self, channels=64, num_classes=10):
        super().__init__()  # set up nn.Module internals
        self.time_embed = nn.Embedding(TIMESTEPS, channels)        # one learned vector per timestep
        self.class_embed = nn.Embedding(num_classes, channels)     # one learned vector per class

        self.conv_in = nn.Conv2d(3, channels, 3, padding=1)        # bring 3 RGB channels up to `channels`
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)   # hidden conv layer 1
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)   # hidden conv layer 2
        self.conv3 = nn.Conv2d(channels, channels, 3, padding=1)   # hidden conv layer 3
        self.conv_out = nn.Conv2d(channels, 3, 3, padding=1)       # project back down to 3 channels (predicted noise)

    def forward(self, x, t, y):
        temb = self.time_embed(t)[:, :, None, None]                # (batch, channels, 1, 1) time embedding
        cemb = self.class_embed(y)[:, :, None, None]                # (batch, channels, 1, 1) class embedding

        h = self.conv_in(x)                                         # lift image to feature space
        h = F.relu(self.conv1(h) + temb + cemb)                     # conv + add conditioning + activate
        h = F.relu(self.conv2(h) + temb + cemb)                     # conv + add conditioning + activate
        h = F.relu(self.conv3(h) + temb + cemb)                     # conv + add conditioning + activate
        return self.conv_out(h)                                     # output predicted noise, same shape as input



# 3. Training on CIFAR 10


def train():

    transform = T.Compose([T.ToTensor(), T.Normalize((0.5,) * 3, (0.5,) * 3)])  # scale pixels to [-1, 1]
    dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)  # dataset
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)  # batches + shuffles the data

    model = SimpleDenoiser().to(DEVICE)                             # create the model on GPU/CPU
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)        # Adam optimizer

    model.train()  # training mode
    for epoch in range(EPOCHS):  # loop over the dataset multiple times
        total_loss = 0.0  # track loss for this epoch
        for images, labels in loader:  # go through each batch
            images = images.to(DEVICE)                              # move images to device
            labels = labels.to(DEVICE)                               # move class labels to device

            t = torch.randint(0, TIMESTEPS, (images.size(0),), device=DEVICE)  # random timestep per image
            noise = torch.randn_like(images)                          # random noise, same shape as images
            noisy_images = add_noise(images, t, noise)                # apply forward diffusion

            predicted_noise = model(noisy_images, t, labels)          # model guesses what noise was added
            loss = F.mse_loss(predicted_noise, noise)                 # compare guess to the real noise

            optimizer.zero_grad()                                     # clear old gradients
            loss.backward()                                           # backpropagate
            optimizer.step()                                          # update weights

            total_loss += loss.item()                                 # accumulate loss

        print(f"Epoch {epoch + 1}/{EPOCHS} - loss: {total_loss / len(loader):.4f}")  # print progress

    return model  # give back the trained model



# 4. Image generation


@torch.no_grad()  # no gradients needed for generation
def generate(model, class_name, n_images=5):
    """Generates n_images of the given class name by reversing the noise process step by step."""
    class_idx = CLASS_NAMES.index(class_name)                        # turn "airplane" into 0, etc.
    labels = torch.full((n_images,), class_idx, dtype=torch.long, device=DEVICE)  # same label repeated n times

    model.eval()  # evaluation mode
    x = torch.randn(n_images, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)  # start from pure random noise

    for t_step in reversed(range(TIMESTEPS)):  # walk backwards from the last timestep to the first
        t = torch.full((n_images,), t_step, dtype=torch.long, device=DEVICE)  # current timestep, repeated

        predicted_noise = model(x, t, labels)                         # model predicts the noise in x right now

        alpha = alphas[t_step]                                        # alpha for this step
        alpha_bar = alpha_bars[t_step]                                # cumulative alpha for this step
        beta = betas[t_step]                                          # beta for this step

        mean = (x - beta / (1 - alpha_bar).sqrt() * predicted_noise) / alpha.sqrt()  # estimated x at previous step

        if t_step > 0:                                                # add a little noise back in, except at the end
            x = mean + beta.sqrt() * torch.randn_like(x)
        else:
            x = mean                                                  # final step: no extra noise added

    return (x.clamp(-1, 1) + 1) / 2  # rescale from [-1, 1] back to [0, 1] for saving as an image



# 5. RUN IT


if __name__ == "__main__":
    trained_model = train()  # train the model

    os.makedirs("generated", exist_ok=True)  # make a folder to save results in

    class_to_generate = "airplane"
    how_many = 5

    images = generate(trained_model, class_to_generate, n_images=how_many)  # run reverse diffusion
    for i, img in enumerate(images):  # save each image separately
        save_image(img, f"generated/{class_to_generate}_{i}.png")
    save_image(images, f"generated/{class_to_generate}_grid.png", nrow=how_many)  # also save a grid version

    print(f"Saved {how_many} images of '{class_to_generate}' to the 'generated/' folder")
