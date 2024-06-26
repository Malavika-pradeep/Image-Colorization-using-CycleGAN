import torch 
import torch.nn as nn
from model import Generator
from model import Discriminator
import torch.optim as optim
from dataset import dataloader

# Set the hyperparameters
num_epochs = 1
lr = 0.0002
alpha = 10
beta = 1

# Initialize the generators and discriminators
G_color = Generator(in_channels=1,out_channels=3)
G_gray = Generator(in_channels=3,out_channels=1)
D_color = Discriminator(in_channels=3)
D_gray = Discriminator(in_channels=1)

# Initialize the optimizers
optimizer_G = optim.Adam(list(G_color.parameters()) + list(G_gray.parameters()), lr=lr)
optimizer_D_color = optim.Adam(D_color.parameters(), lr=lr)
optimizer_D_gray = optim.Adam(D_gray.parameters(), lr=lr)

# Initialize the loss functions
criterion_GAN = nn.BCELoss()
criterion_cycle = nn.L1Loss()
criterion_detail = nn.MSELoss()

# Train the model
for epoch in range(num_epochs):
    for i, (x, y) in enumerate(dataloader):
        # Set the inputs
        x_gray = y
        y_color = x

        # Forward pass
        x_fake_color = G_color(x_gray)
        y_fake_gray = G_gray(y_color)
        x_reconstructed = G_gray(x_fake_color)
        y_reconstructed = G_color(y_fake_gray)

        # Calculate the losses
        loss_GAN_color = criterion_GAN(D_color(x_fake_color.detach()), torch.ones_like(D_color(x_fake_color)))
        loss_GAN_gray = criterion_GAN(D_gray(y_fake_gray.detach()), torch.ones_like(D_gray(y_fake_gray)))
        loss_cycle = criterion_cycle(x_reconstructed, x_gray) + criterion_cycle(y_reconstructed, y_color)
        loss_detail = criterion_detail(x_fake_color, y_color) + criterion_detail(y_fake_gray, x_gray)

        # Calculate the total loss
        loss_G = loss_GAN_color + loss_GAN_gray + alpha * loss_cycle + beta * loss_detail

        # Backward pass
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        # Train the discriminators
        loss_D_color = criterion_GAN(D_color(x_fake_color.detach()), torch.zeros_like(D_color(x_fake_color))) + criterion_GAN(D_color(y_color), torch.ones_like(D_color(y_color)))
        loss_D_gray = criterion_GAN(D_gray(y_fake_gray.detach()), torch.zeros_like(D_gray(y_fake_gray))) + criterion_GAN(D_gray(x_gray), torch.ones_like(D_gray(x_gray)))

        optimizer_D_color.zero_grad()
        loss_D_color.backward()
        optimizer_D_color.step()

        optimizer_D_gray.zero_grad()
        loss_D_gray.backward()
        optimizer_D_gray.step()
        print(f'Epoch {epoch+1}, Loss G: {loss_G.item():.4f}, Loss D_color: {loss_D_color.item():.4f}, Loss D_gray: {loss_D_gray.item():.4f}')