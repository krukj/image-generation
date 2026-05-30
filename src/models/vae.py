import torch
from torch import nn


class VAEEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, dim_latent: int = 100):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)  # (32, 32)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)  # (16, 16)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)  # (8, 8)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc_mean = nn.Linear(8 * 8 * 128, dim_latent)
        self.fc_var = nn.Linear(8 * 8 * 128, dim_latent)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)

        mu = self.fc_mean(x)
        log_sigma = self.fc_var(x)
        return mu, log_sigma


class VAEDecoder(nn.Module):
    def __init__(self, in_channels: int = 3, dim_latent: int = 100):
        super().__init__()
        self.fc = nn.Linear(dim_latent, 8 * 8 * 128)
        self.convt1 = nn.ConvTranspose2d(128, 64, 4, 2, 1)  # (16, 16)
        self.convt2 = nn.ConvTranspose2d(64, 32, 4, 2, 1)  # (32, 32)
        self.convt3 = nn.ConvTranspose2d(32, in_channels, 4, 2, 1)  # (64, 64)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 128, 8, 8)

        x = self.relu(self.convt1(x))
        x = self.relu(self.convt2(x))
        x_hat = self.sigmoid(self.convt3(x))
        return x_hat


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = VAEEncoder()
        self.decoder = VAEDecoder()

    def reparametrize(self, mu, log_sigma):
        sigma = torch.exp(0.5 * log_sigma)
        eps = torch.randn_like(sigma)
        z = mu + sigma * eps
        return z

    def forward(self, x):
        mu, log_sigma = self.encoder(x)
        z = self.reparametrize(mu, log_sigma)
        x_hat = self.decoder(z)

        return x_hat, mu, log_sigma
