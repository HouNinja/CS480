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


class Generator(nn.Module):
    #The generator takes an input of size latent_size, and will produce an output of size 784.
    #It should have a single hidden linear layer with 400 nodes using ReLU activations, and use Sigmoid activation for its outputs
    def __init__(self):
        super(Generator, self).__init__()
        #TODO
        self.Layer1 = nn.Sequential(
            nn.Linear(latent_size, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid()
        )

    def forward(self, z):
        #TODO
        output = self.Layer1(z)
        return output

class Discriminator(nn.Module):
    #The discriminator takes an input of size 784, and will produce an output of size 1.
    #It should have a single hidden linear layer with 400 nodes using ReLU activations, and use Sigmoid activation for its output
    def __init__(self):
        super(Discriminator, self).__init__()
        #TODO
        self.Layer1 = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(),
            nn.Linear(400, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        #TODO
        output = self.Layer1(x)
        return output

def train(generator, generator_optimizer, discriminator, discriminator_optimizer):
    #Trains both the generator and discriminator for one epoch on the training dataset.
    #Returns the average generator and discriminator loss (scalar values, use the binary cross-entropy appropriately)
    #TODO
    avg_generator_loss = 0
    avg_discriminator_loss = 0
    criterion = nn.BCELoss()
    index = -1
    for (i, (images, _)) in enumerate(train_loader):
        ##Train with Real batch
        index = i + 1
        images = images.view(batch_size, 784)
        discriminator_optimizer.zero_grad()
        predicted_real_data_lable = discriminator(images)
        data_label = torch.full((batch_size, 1), 1, dtype=torch.float, device=device)
        discriminator_real_data_train_loss = criterion(predicted_real_data_lable, data_label)
        discriminator_real_data_train_loss.backward()

        ##Train with generated fake batch that has the same batch size
        noise = torch.randn(batch_size, latent_size, device=device)
        fake_images = generator(noise)
        predicted_fake_data_lable = discriminator(fake_images.detach())
        data_label.fill_(0)
        discriminator_fake_data_train_loss = criterion(predicted_fake_data_lable, data_label)
        discriminator_fake_data_train_loss.backward()
        avg_discriminator_loss += discriminator_real_data_train_loss.item()
        avg_discriminator_loss += discriminator_fake_data_train_loss.item()
        discriminator_optimizer.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        generator_optimizer.zero_grad()
        ##As we update the discriminator, we need to update the output using new discriminator
        updated_predicted_fake_labels = discriminator(fake_images)
        data_label.fill_(1)
        generator_train_error = criterion(updated_predicted_fake_labels, data_label)
        avg_generator_loss += generator_train_error.item()
        generator_train_error.backward()
        # Update G
        generator_optimizer.step()

    print(f"the generator loss{avg_generator_loss / index}, dis_loss is {avg_discriminator_loss / index / 2}")
    return avg_generator_loss / index, avg_discriminator_loss / index / 2

def test(generator, discriminator):
    #Runs both the generator and discriminator over the test dataset.
    #Returns the average generator and discriminator loss (scalar values, use the binary cross-entropy appropriately)
    #TODO
    with torch.no_grad():
        avg_generator_loss = 0
        avg_discriminator_loss = 0
        criterion = nn.BCELoss()
        real_data_label = torch.full((batch_size,1), 1, dtype=torch.float, device=device)
        fake_data_label = torch.full((batch_size,1), 0, dtype=torch.float, device=device)
        index = -1
        for (i, (images, _)) in enumerate(test_loader):
            index = i + 1
            images = images.view(batch_size, 784)
            predicted_real_data_lable = discriminator(images)
            discriminator_real_data_test_loss = criterion(predicted_real_data_lable, real_data_label)
            noise = torch.randn(batch_size, latent_size, device=device)
            fake_images = generator(noise)
            predicted_fake_data_lable = discriminator(fake_images)
            discriminator_fake_data_test_loss = criterion(predicted_fake_data_lable, fake_data_label)
            avg_discriminator_loss += discriminator_real_data_test_loss.item()
            avg_discriminator_loss += discriminator_fake_data_test_loss.item()

            ##Generator has nothing to do with real data, just sum up the fake data loss
            generator_test_loss = criterion(predicted_fake_data_lable, real_data_label)
            avg_generator_loss += discriminator_fake_data_test_loss.item()


    return avg_generator_loss / index, avg_discriminator_loss / index / 2

epochs = 50

discriminator_avg_train_losses = []
discriminator_avg_test_losses = []
generator_avg_train_losses = []
generator_avg_test_losses = []

generator = Generator().to(device)
discriminator = Discriminator().to(device)

generator_optimizer = optim.Adam(generator.parameters(), lr=1e-3)
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=1e-3)

##The generated sample for every 10th, 20th ...... epoch
selected_noise = torch.randn(1, latent_size, device=device)

for epoch in range(1, epochs + 1):
    generator_avg_train_loss, discriminator_avg_train_loss = train(generator, generator_optimizer, discriminator, discriminator_optimizer)
    generator_avg_test_loss, discriminator_avg_test_loss = test(generator, discriminator)

    discriminator_avg_train_losses.append(discriminator_avg_train_loss)
    generator_avg_train_losses.append(generator_avg_train_loss)
    discriminator_avg_test_losses.append(discriminator_avg_test_loss)
    generator_avg_test_losses.append(generator_avg_test_loss)


    with torch.no_grad():
        sample = torch.randn(64, latent_size).to(device)
        sample = generator(sample).cpu()
        save_image(sample.view(64, 1, 28, 28),
                   'results/sample_' + str(epoch) + '.png')
        print('Epoch #' + str(epoch))
        display(Image('results/sample_' + str(epoch) + '.png'))
        print('\n')


plt.plot(discriminator_avg_train_losses)
plt.plot(generator_avg_train_losses)
plt.title('Training Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Disc','Gen'], loc='upper right')
plt.show()

plt.plot(discriminator_avg_test_losses)
plt.plot(generator_avg_test_losses)
plt.title('Test Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Disc','Gen'], loc='upper right')
plt.show()
