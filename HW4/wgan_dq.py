import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.utils as utils
import numpy as np
from scipy.linalg import sqrtm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# CIFAR10 dataset
dataset = datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Hyperparameters
latent_dim = 100
batch_size = 64
channels = 3
lr = 2e-4
n_discriminator = 5
clip_value = 0.01  # clip parameter for WGAN
epochs = 50
img_size = 64

# Generator Model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # Input is Z, going into a convolution
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # State size. 512 x 4 x 4
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # State size. 256 x 8 x 8
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # State size. 128 x 16 x 16
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # State size. 64 x 32 x 32
            nn.ConvTranspose2d(64, channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output size. channels x 64 x 64
        )

    def forward(self, input):
        return self.model(input)

# Critic (Discriminator) Model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # Input is channels x 64 x 64
            nn.Conv2d(channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State size. 64 x 32 x 32
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # State size. 128 x 16 x 16
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # State size. 256 x 8 x 8
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # State size. 512 x 4 x 4
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
        )

    def forward(self, input):
        return self.model(input).view(-1, 1)

# Initialize generator and critic
generator = Generator().to(device)
discriminator = Discriminator().to(device)
print(generator)
print(discriminator)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

generator.apply(weights_init)
discriminator.apply(weights_init)

# Optimizers
criterion = nn.BCELoss()
optimizerG = optim.RMSprop(generator.parameters(), lr=lr)
optimizerD = optim.RMSprop(discriminator.parameters(), lr=lr)

generator.train()
discriminator.train()


epochs = 50
iters = 0
fixed_noise = torch.randn(32, 100, 1, 1).to(device)
img_list = []
g_losses = []
d_losses = []


for epoch in range(epochs):
    for i, (real_images, _) in enumerate(dataloader):
        # Train Critic with real images
        discriminator.zero_grad()
        real_images = real_images.to(device)
        real_loss = discriminator(real_images).mean()
        real_loss.backward(torch.tensor(-1.0))  # mone is torch.tensor(-1.0), which is used to flip the sign

        # Train Critic with fake images
        noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        fake_images = generator(noise).detach()  # detach to avoid training the generator on these labels
        fake_loss = discriminator(fake_images).mean()
        fake_loss.backward(torch.tensor(1.0))  # one is torch.tensor(1.0)

        discriminator_loss = fake_loss - real_loss
        optimizerD.step()

        # Clip weights of critic to satisfy Lipschitz constraint
        for p in discriminator.parameters():
            p.data.clamp_(-0.01, 0.01)

        # Update Generator
        if i % n_discriminator == 0:
            generator.zero_grad()
            gen_fake = generator(noise)
            gen_loss = -discriminator(gen_fake).mean()
            gen_loss.backward()
            optimizerG.step()

            g_losses.append(gen_loss.item())
            d_losses.append(discriminator_loss.item())

           
        if i % 100 == 0:
 
            with torch.no_grad():
                fake = generator(fixed_noise).detach().cpu()
            img_list.append(utils.make_grid(fake, padding=2, normalize=True))

        iters += 1

    

    with torch.no_grad():
        fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)
        fake_images = generator(fixed_noise).detach().cpu()
    save_image(fake_images, f'./figure/wgan/epoch_{epoch:03d}.png', nrow=8, normalize=True)



