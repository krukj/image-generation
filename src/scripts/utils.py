import datetime
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch_fidelity
import torchvision.transforms as transforms
import torchvision.utils as vutils


def setup_logger() -> logging.Logger:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                log_dir / f"experiment_{datetime.datetime.now():%Y%m%d_%H%M%S}.log"
            ),
        ],
    )
    return logging.getLogger(__name__)

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)



def save_images(netG: nn.Module, noise: torch.Tensor, out_dir: Path, tag: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        fake = netG(noise).detach().cpu()

    grid = vutils.make_grid(fake, padding=2, normalize=True)
    vutils.save_image(grid, out_dir / f"{tag}_grid.png")

    singles_dir = out_dir / "singles"
    singles_dir.mkdir(exist_ok=True)
    for idx, img in enumerate(fake):
        vutils.save_image(img, singles_dir / f"{tag}_{idx:04d}.png", normalize=True)





def compute_metrics(real_path: str | Path, fake_path: str | Path) -> dict:
    """Calculate FID, IS, precision and recall"""

    metrics = torch_fidelity.calculate_metrics(
        input1=str(fake_path),
        input2=str(real_path),
        cuda=torch.cuda.is_available(),
        isc=True,
        fid=True,
        prc=True,
        prc_neighborhood_k=5,
        batch_size=32
    )
    return {
        "fid": round(metrics["frechet_inception_distance"], 4),
        "is_mean": round(metrics["inception_score_mean"], 4),
        "is_std": round(metrics["inception_score_std"], 4),
        "precision": round(metrics["precision"], 4),
        "recall": round(metrics["recall"], 4),
    }