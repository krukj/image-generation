from pathlib import Path

import numpy as np
import torch
import torchvision.utils as vutils
from diffusers import DDIMPipeline, DDIMScheduler, UNet2DModel
from diffusers.utils import make_image_grid
from PIL import Image

from src.config import DCGAN_DEFAULT, DDIM_DEFAULT, NGPU, TARGET_SIZE
from src.models.dcgan import Generator




def load_ddim_and_interpolate(
    weights_path: Path,
    out_dir: Path,
    num_steps: int = 10,
    num_inference_steps: int = 50,
):  
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = UNet2DModel(
        sample_size=TARGET_SIZE,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(64, 128, 128, 256),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )

    model.load_state_dict(
        torch.load(weights_path, map_location=device, weights_only=True)
    )
    model.to(device)

    scheduler = DDIMScheduler(
        num_train_timesteps=DDIM_DEFAULT.get("num_train_timesteps"),
        beta_schedule=DDIM_DEFAULT.get("beta_schedule"),
    )

    pipeline = DDIMPipeline(unet=model, scheduler=scheduler)
    pipeline.to(device)

    out_dir.mkdir(parents=True, exist_ok=True)
    device = pipeline.device
    model = pipeline.unet
    scheduler = pipeline.scheduler

    shape = (1, 3, TARGET_SIZE, TARGET_SIZE)

    noise_A = torch.randn(shape, device=device)
    noise_B = torch.randn(shape, device=device)

    interpolated_latents = []

    for i in range(num_steps):
        alpha = i / (num_steps - 1)
        noise_interp = (1 - alpha) * noise_A + alpha * noise_B
        interpolated_latents.append(noise_interp)

    images = []
    scheduler.set_timesteps(num_inference_steps, device=device)

    for i, latent in enumerate(interpolated_latents):
        image = latent.clone()

        with torch.no_grad():
            for t in scheduler.timesteps:
                model_output = model(image, t).sample

                image = scheduler.step(model_output, t, image).prev_sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = Image.fromarray((image * 255).round().astype(np.uint8))

        images.append(image)
        image.save(out_dir / f"interp_{i:02d}.png")

    grid = make_image_grid(images, rows=1, cols=num_steps)
    grid.save(out_dir / "interpolation_grid.png")


def load_dcgan_and_interpolate(weights_path: Path, out_dir: Path):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    nz = DCGAN_DEFAULT.get("nz")
    netG = Generator(NGPU, nz=nz).to(device)

    netG.load_state_dict(
        torch.load(weights_path, map_location=device, weights_only=True)
    )

    netG.eval()

    out_dir.mkdir(parents=True, exist_ok=True)
    num_steps = 10

    z_A = torch.randn(1, nz, 1, 1, device=device)
    z_B = torch.randn(1, nz, 1, 1, device=device)

    images_list = []

    for i in range(num_steps):
        alpha = i / (num_steps - 1)

        z_interp = (1 - alpha) * z_A + alpha * z_B

        with torch.no_grad():
            fake_img = netG(z_interp)

        images_list.append(fake_img.cpu())

        vutils.save_image(fake_img, out_dir / f"interp_{i:02d}.png", normalize=True)

    all_images_tensor = torch.cat(images_list, dim=0)

    vutils.save_image(
        all_images_tensor,
        out_dir / "interpolation_grid.png",
        normalize=True,
        nrow=num_steps,
    )

if __name__ == "__main__":
    load_dcgan_and_interpolate(weights_path=Path("results/dcgan_default/seed1/netG.pt"), out_dir=Path("dcgan_interpolations"))
    load_ddim_and_interpolate(weights_path=Path("results/ddim_default/seed1/DDIM.pt"), out_dir=Path("ddim_interpolations"))