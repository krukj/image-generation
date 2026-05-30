import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.datasets import ImageFolder
from pathlib import Path
import time

from models.vae import VAE


def vae_loss_function(x_hat, x, mu, log_sigma, beta=1.0):
    bce = F.binary_cross_entropy(x_hat, x, reduction="sum")

    kld = -0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp())

    batch_size = x.size(0)
    return (bce + beta * kld) / batch_size, bce / batch_size, kld / batch_size


def train_vae(
    dataroot: Path,
    out_dir: Path,
    dim_latent: int = 100,
    batch_size: int = 64,
    epochs: int = 50,
    lr: float = 1e-3,
):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    model = VAE(dim_latent=dim_latent).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    dataset = ImageFolder(
        root=dataroot,
        transform=transforms.Compose(
            [
                transforms.Resize(64),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
            ]
        ),
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    fixed_noise = torch.randn(64, dim_latent, device=device)

    print(f"Rozpoczęcie treningu na: {device}")

    for epoch in range(epochs):
        model.train()
        total_loss, total_bce, total_kld = 0, 0, 0
        t0 = time.time()

        for i, (images, _) in enumerate(dataloader):
            images = images.to(device)

            optimizer.zero_grad()

            # Forward
            x_hat, mu, log_sigma = model(images)

            # ELBO
            loss, bce, kld = vae_loss_function(x_hat, images, mu, log_sigma)

            # Backward
            loss.backward()
            optimizer.step()

            # Stats
            total_loss += loss.item()
            total_bce += bce.item()
            total_kld += kld.item()

            if i % 50 == 0:
                print(
                    f"[{epoch}/{epochs}][{i}/{len(dataloader)}] "
                    f"Loss: {loss.item():.2f} (BCE: {bce.item():.2f}, KLD: {kld.item():.2f})"
                )

        # Logging
        avg_loss = total_loss / len(dataloader)
        elapsed = time.time() - t0
        print(f"====> Epoka: {epoch} Średni Loss: {avg_loss:.2f} Czas: {elapsed:.1f}s")

        # Image generation
        model.eval()
        with torch.no_grad():
            generated_images = model.decoder(fixed_noise).cpu()
            vutils.save_image(generated_images, out_dir / f"epoch_{epoch:03d}.png", nrow=8)

            # Weights
            torch.save(model.state_dict(), out_dir / "vae_cats.pt")


if __name__ == "__main__":
    train_vae(dataroot=Path("data/cats"), out_dir=Path("results/vae"))
