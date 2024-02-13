from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from IPython.display import Image, display
import matplotlib.pyplot as plt
import os

if not os.path.exists('results'):
    os.mkdir('results')

batch_size = 100
latent_size = 20

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        #TODO
        self.Layer1 = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(),
        )

        self.Mean_Layer = nn.Linear(400, latent_size)
        self.Variance_Layer = nn.Linear(400, latent_size)

        self.Layer2 = nn.Sequential(
            nn.Linear(latent_size, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid()
        )

    def encode(self, x):
        #The encoder will take an input of size 784, and will produce two vectors of size latent_size (corresponding to the coordinatewise means and log_variances)
        #It should have a single hidden linear layer with 400 nodes using ReLU activations, and have two linear output layers (no activations)
        #TODO
        output = self.Layer1(x)
        mean = self.Mean_Layer(output)
        log_variance = self.Variance_Layer(output)
        return mean, log_variance

    def reparameterize(self, means, log_variances):
        #The reparameterization module lies between the encoder and the decoder
        #It takes in the coordinatewise means and log-variances from the encoder (each of dimension latent_size), and returns a sample from a Gaussian with the corresponding parameters
        #TODO
        # variances = torch.exp(log_variances)
        data = torch.distributions.Normal(0, 1).sample(means.shape)   
        z = means + log_variances*data
        return z

    def decode(self, z):
        #The decoder will take an input of size latent_size, and will produce an output of size 784
        #It should have a single hidden linear layer with 400 nodes using ReLU activations, and use Sigmoid activation for its outputs
        #TODO
        output = self.Layer2(z)
        return output

    def forward(self, x):
        #Apply the VAE encoder, reparameterization, and decoder to an input of size 784
        #Returns an output image of size 784, as well as the means and log_variances, each of size latent_size (they will be needed when computing the loss)
        #TODO
        mean, log_variance = self.encode(x)
        z = self.reparameterize(mean, log_variance)
        output = self.decode(z)
        return output, mean, log_variance

def vae_loss_function(reconstructed_x, x, means, log_variances):
    #Compute the VAE loss
    #The loss is a sum of two terms: reconstruction error and KL divergence
    #Use cross entropy loss between x and reconstructed_x for the reconstruction error (as opposed to L2 loss as discussed in lecture -- this is sometimes done for data in [0,1] for easier optimization)
    #The KL divergence is -1/2 * sum(1 + log_variances - means^2 - exp(log_variances)) as described in lecture
    #Returns loss (reconstruction + KL divergence) and reconstruction loss only (both scalars)
    #TODO
    criterion = nn.BCELoss(reduction = 'sum')
    reconstruction_loss = criterion(reconstructed_x, x)
    KL_Divergence = -1/2 * torch.sum(1 + log_variances - torch.square(means) - torch.exp(log_variances))
    loss = reconstruction_loss + KL_Divergence
    return loss, reconstruction_loss


def train(model, optimizer):
    #Trains the VAE for one epoch on the training dataset
    #Returns the average (over the dataset) loss (reconstruction + KL divergence) and reconstruction loss only (both scalars)
    #TODO
    avg_train_loss = 0
    avg_train_reconstruction_loss = 0
    index = -1
    for (i, (images, _)) in enumerate(train_loader):
        index = i + 1
        images = images.view(batch_size, 784)
        optimizer.zero_grad()
        reconstructed_images, mean, log_variance = model(images)
        batch_train_loss, batch_reconstruction_loss = vae_loss_function(reconstructed_images, images, mean, log_variance)
        avg_train_loss += batch_train_loss.item()
        avg_train_reconstruction_loss += batch_reconstruction_loss.item()
        batch_train_loss.backward()
        optimizer.step()
    print(f"the train loss is {avg_train_loss / index}")
    return avg_train_loss / index, avg_train_reconstruction_loss / index

def test(model):
    #Runs the VAE on the test dataset
    #Returns the average (over the dataset) loss (reconstruction + KL divergence) and reconstruction loss only (both scalars)
    #TODO
    with torch.no_grad():
        avg_test_loss = 0
        avg_test_reconstruction_loss = 0
        index = -1
        for (i, (images, _)) in enumerate(test_loader):
            index = i + 1
            images = images.view(batch_size, 784)
            reconstructed_images, mean, variance = model(images)
            batch_test_loss, batch_reconstruction_loss = vae_loss_function(reconstructed_images, images, mean, variance)
            avg_test_loss += batch_test_loss.item()
            avg_test_reconstruction_loss += batch_reconstruction_loss.item()

    return avg_test_loss / index, avg_test_reconstruction_loss / index

epochs = 50
avg_train_losses = []
avg_train_reconstruction_losses = []
avg_test_losses = []
avg_test_reconstruction_losses = []

vae_model = VAE().to(device)
vae_optimizer = optim.Adam(vae_model.parameters(), lr=1e-3)

for epoch in range(1, epochs + 1):
    avg_train_loss, avg_train_reconstruction_loss = train(vae_model, vae_optimizer)
    avg_test_loss, avg_test_reconstruction_loss = test(vae_model)
    avg_train_losses.append(avg_train_loss)
    avg_train_reconstruction_losses.append(avg_train_reconstruction_loss)
    avg_test_losses.append(avg_test_loss)
    avg_test_reconstruction_losses.append(avg_test_reconstruction_loss)

    with torch.no_grad():
        sample = torch.randn(64, latent_size).to(device)
        sample = vae_model.decode(sample).cpu()
        save_image(sample.view(64, 1, 28, 28),
                   'results/sample_' + str(epoch) + '.png')
        print('Epoch #' + str(epoch))
        display(Image('results/sample_' + str(epoch) + '.png'))
        print('\n')

plt.plot(avg_train_reconstruction_losses)
plt.title('Training Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch #')
plt.show()

plt.plot(avg_test_reconstruction_losses)
plt.title('Test Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch #')
plt.show()
