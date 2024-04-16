import torch
import torchvision
from torchvision import utils
from torch.utils.data import DataLoader
from torch import nn
from torch.autograd import Variable
from pytorch_gan_metrics import get_inception_score
from tqdm import tqdm
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

def generate_images(generator):
    z = torch.randn(batch_size, 100, 1, 1).to(device)
    samples = generator(z)
    samples = samples.mul(0.5).add(0.5)
    samples = samples.data.cpu()
    grid = utils.make_grid(samples)
    print("Grid of 8x8 images saved to 'dcgan_generated_images.png'.")
    utils.save_image(grid, 'dcgan_generated_images.png')

def load_model(model, model_filename): 
    model.load_state_dict(torch.load(model_filename))

def create_CIFAR10_dataloaders(batch_size): 
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(), 
        torchvision.transforms.Resize(32), 
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_CIFAR10_set = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transform)
    test_CIFAR10_set = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transform)

    train_CIFAR10_dataloader = DataLoader(train_CIFAR10_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_CIFAR10_dataloader = DataLoader(test_CIFAR10_set, batch_size=batch_size, shuffle=True, drop_last=True)
    return train_CIFAR10_dataloader, test_CIFAR10_dataloader

print("Downloading CIFAR10 dataset...")
batch_size = 32
train_dataloader, test_dataloader = create_CIFAR10_dataloaders(batch_size)




class Generator_DCGAN(nn.Module):
    def __init__(self): 
        super(Generator_DCGAN, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_channels=100, out_channels=1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=1024),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(in_channels=256, out_channels=3, kernel_size=4, stride=2, padding=1),
            nn.Tanh())

    def forward(self, x):
        return self.net(x)

class Discriminator_DCGAN(nn.Module):
    def __init__(self):
        super(Discriminator_DCGAN, self).__init__()
        self.net = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=256, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(512),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(1024),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0),
        nn.Sigmoid())

    def forward(self, x):
        return self.net(x)


dcgan_generator = Generator_DCGAN()
dcgan_discriminator = Discriminator_DCGAN()
dcgan_generator.to(device)
dcgan_discriminator.to(device)



learning_rate = 0.0002
epochs = 50


def train(generator, discriminator, train_dataloader):
    loss = nn.BCELoss()
    optim_generator = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optim_discriminator = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    if not os.path.exists('train_generated_images_dcgan/'): 
        os.makedirs('train_generated_images_dcgan')
        
    inception_score_file = open("inception_score_dcgan.csv", "w")
    inception_score_file.write('epoch, inception_score \n')

    for epoch in tqdm(range(epochs)): 
        for real_images, _ in train_dataloader:
            real_images = real_images.to(device)
            z = Variable(torch.randn(batch_size, 100, 1, 1)).to(device)
            real_labels = torch.ones(batch_size).to(device)
            fake_labels = torch.zeros(batch_size).to(device)

            ### train discriminator
            # compute loss using real images
            preds = discriminator(real_images)
            disc_loss_real = loss(preds.flatten(), real_labels)

            # compute loss using fake images
            fake_images = generator(z)
            preds = discriminator(fake_images)
            disc_loss_fake = loss(preds.flatten(), fake_labels)

            # optimize discriminator
            disc_loss = disc_loss_real + disc_loss_fake
            discriminator.zero_grad()
            disc_loss.backward()
            optim_discriminator.step()

            ### train generator
            # compute loss with fake images
            z = Variable(torch.randn(batch_size, 100, 1, 1)).to(device)
            fake_images = generator(z)
            preds = discriminator(fake_images)
            gen_loss = loss(preds.flatten(), real_labels)

            # optimize generator 
            generator.zero_grad()
            gen_loss.backward()
            optim_generator.step()

        # compute inception score and samples every epoch
        z = Variable(torch.randn(800, 100, 1, 1)).to(device)
        samples = generator(z)

        # normalize to [0, 1]
        samples = samples.mul(0.5).add(0.5)
        
        assert 0 <= samples.min() and samples.max() <= 1
        inception_score, inception_score_std = get_inception_score(samples)
        print("epoch: " + str(epoch) + ', inception score: ' + str(round(inception_score, 2)))

        samples = samples[:64].data.cpu()
        grid = utils.make_grid(samples)
        utils.save_image(grid, 'train_generated_images_dcgan/epoch_{}.png'.format(str(epoch)))
        
        inception_score_file.write(str(epoch) + ', ' + str(round(inception_score, 2)) + '\n')

    inception_score_file.close()



train(dcgan_generator, dcgan_discriminator, train_dataloader)

