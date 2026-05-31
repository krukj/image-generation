import gc
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.datasets import ImageFolder

from src.models.vae import VAE
from src.config import (
    DATA_PROCESSED_TORCH_DIR,
    NGPU,
    RESULTS_PATH,
    TARGET_SIZE,
    VAE_DEFAULT,
    VAE_GRID,
    WORKERS,
)
from src.scripts.utils import compute_metrics, set_seed, setup_logger

device = torch.device(
    "mps"
    if (torch.backends.mps.is_available() and NGPU > 0)
    else "cuda" if torch.cuda.is_available() else "cpu"
)
logger = setup_logger()


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def vae_loss_function(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bce = F.binary_cross_entropy(x_hat, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp())
    batch_size = x.size(0)
    return (bce + beta * kld) / batch_size, bce / batch_size, kld / batch_size


def _aggregate_seeds(exp_name: str, results: list[dict]) -> dict:
    keys = ["fid", "is_mean", "precision", "recall"]
    agg = {
        "exp_name": exp_name,
        "n_seeds": len(results),
        "params": results[0]["params"],
    }
    for k in keys:
        vals = [r["final_metrics"].get(k, float("nan")) for r in results]
        t = torch.tensor(vals, dtype=torch.float32)
        agg[f"{k}_mean"] = round(t.mean().item(), 4)
        agg[f"{k}_std"] = round(t.std().item(), 4)
    return agg


def _generate_fid_images(
    model: VAE,
    dim_latent: int,
    out_dir: Path,
    n: int = 2000,
    batch_size: int = 32,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(out_dir.glob("*.png")))
    if existing >= n:
        return
    needed = n - existing
    idx = existing
    generated = 0
    model.eval()
    with torch.no_grad():
        while generated < needed:
            cnt = min(batch_size, needed - generated)
            z = torch.randn(cnt, dim_latent, device=device)
            images = model.decoder(z).cpu()
            for img_tensor in images:
                vutils.save_image(img_tensor, out_dir / f"gen_{idx:05d}.png")
                idx += 1
            generated += cnt
    clear_memory()


def _save_sample_grid(
    model: VAE,
    fixed_noise: torch.Tensor,
    out_dir: Path,
    tag: str,
):
    model.eval()
    with torch.no_grad():
        images = model.decoder(fixed_noise).cpu()

    n = len(images)
    cols = int(math.sqrt(n))
    rows = n // cols
    grid_images = images[: rows * cols]

    out_dir.mkdir(parents=True, exist_ok=True)
    vutils.save_image(grid_images, out_dir / f"{tag}_grid.png", nrow=cols)

    singles_dir = out_dir / "singles"
    singles_dir.mkdir(exist_ok=True)
    for idx, img in enumerate(images):
        vutils.save_image(img, singles_dir / f"{tag}_{idx:04d}.png")


def _update_summary(summary_path: Path, entry: dict) -> None:
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
    else:
        summary = []

    summary = [s for s in summary if s["exp_name"] != entry["exp_name"]]
    summary.append(entry)

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to: {summary_path}")


def run_single_experiment(
    exp_name: str,
    seed: int,
    dim_latent: int,
    beta: float,
    lr: float,
    num_epochs: int,
    batch_size: int,
    workers: int,
    dataroot: Path,
    real_path: Path,
    results_root: Path,
) -> dict:
    set_seed(seed)
    run_dir = results_root / exp_name / f"seed{seed}"
    metrics_path = run_dir / "metrics.json"

    if metrics_path.exists():
        logger.info(f"Skipping {exp_name}/seed{seed} — already done")
        with open(metrics_path) as f:
            return json.load(f)

    images_dir = run_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    dataset = ImageFolder(
        root=dataroot,
        transform=transforms.Compose(
            [
                transforms.Resize(TARGET_SIZE),
                transforms.CenterCrop(TARGET_SIZE),
                transforms.ToTensor(),
            ]
        ),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=workers
    )

    model = VAE(dim_latent=dim_latent).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    fixed_noise = torch.randn(batch_size, dim_latent, device=device)

    logger.info(f"Experiment: {exp_name} | seed={seed}")
    logger.info(f"Params: dim_latent={dim_latent}, beta={beta}, lr={lr}")

    losses: list[float] = []
    metrics_history: list[dict] = []
    epoch = 0
    t0 = time.time()

    for epoch in range(num_epochs):
        model.train()
        total_loss, total_bce, total_kld = 0.0, 0.0, 0.0

        for i, (images, _) in enumerate(dataloader):
            images = images.to(device)
            optimizer.zero_grad()

            x_hat, mu, log_sigma = model(images)
            loss, bce, kld = vae_loss_function(x_hat, images, mu, log_sigma, beta=beta)
            loss.backward()
            optimizer.step()

            loss_val = loss.detach().item()
            total_loss += loss_val
            total_bce += bce.detach().item()
            total_kld += kld.detach().item()
            losses.append(loss_val)

            if i % 50 == 0:
                elapsed = time.time() - t0
                logger.info(
                    f"[{epoch}/{num_epochs}][{i}/{len(dataloader)}]"
                    f" Loss: {loss_val:.4f}"
                    f" (BCE: {bce.item():.4f}, KLD: {kld.item():.4f})"
                    f" ({elapsed:.0f}s)"
                )

        avg_loss = total_loss / len(dataloader)
        elapsed = time.time() - t0
        logger.info(
            f"Epoch {epoch}/{num_epochs} | Avg Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s"
        )


        _save_sample_grid(
            model=model,
            fixed_noise=fixed_noise,
            out_dir=images_dir,
            tag=f"epoch{epoch:03d}",
        )
        clear_memory()

    logger.info("Calculating metrics")
    singles_dir = images_dir / f"epoch{epoch:03d}" / "singles"
    _generate_fid_images(model, dim_latent, singles_dir, n=2000)
    metrics = compute_metrics(real_path, singles_dir)
    metrics["epoch"] = epoch
    metrics_history.append(metrics)
    logger.info(
        f"FID={metrics['fid']} | IS={metrics['is_mean']}±{metrics['is_std']} | "
        f"P={metrics['precision']} | R={metrics['recall']}"
    )
    clear_memory()

    result = {
        "exp_name": exp_name,
        "seed": seed,
        "params": {
            "dim_latent": dim_latent,
            "beta": beta,
            "lr": lr,
            "num_epochs": num_epochs,
        },
        "training_time": round(time.time() - t0, 1),
        "final_metrics": metrics_history[-1] if metrics_history else {},
        "metrics_history": metrics_history,
        "loss": losses,
    }

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    torch.save(model.state_dict(), run_dir / "VAE.pt")

    return result


def run_vae_experiment(
    seeds: list[int],
    dim_latent: int = VAE_DEFAULT.get("dim_latent"),
    beta: float = VAE_DEFAULT.get("beta"),
    lr: float = VAE_DEFAULT.get("lr"),
    num_epochs: int = VAE_DEFAULT.get("num_epochs"),
    batch_size: int = VAE_DEFAULT.get("batch_size"),
    workers: int = WORKERS,
    dataroot: str | Path = Path(DATA_PROCESSED_TORCH_DIR),
    real_path: str | Path = Path(DATA_PROCESSED_TORCH_DIR),
    results_root: str | Path = Path(RESULTS_PATH),
    exp_name: str | None = None,
) -> list[dict]:

    if exp_name is None:
        exp_name = f"vae_latent{dim_latent}_beta{beta}"

    results_root = Path(results_root)
    results_root.mkdir(exist_ok=True)

    all_results = []
    for seed in seeds:
        r = run_single_experiment(
            exp_name=exp_name,
            seed=seed,
            dim_latent=dim_latent,
            beta=beta,
            lr=lr,
            num_epochs=num_epochs,
            batch_size=batch_size,
            workers=workers,
            dataroot=Path(dataroot),
            real_path=Path(real_path),
            results_root=results_root,
        )
        all_results.append(r)

    summary_entry = _aggregate_seeds(exp_name, all_results)
    _update_summary(results_root / f"summary_{exp_name}.json", summary_entry)

    return all_results


def main():
    REAL_PATH = Path(DATA_PROCESSED_TORCH_DIR) / "cats"
    SEEDS = [1, 2, 3]

    # run_vae_experiment(
    #     exp_name="vae_default",
    #     seeds=SEEDS,
    #     real_path=REAL_PATH,
    # )

    # for beta in VAE_GRID.get("beta"):
    #     run_vae_experiment(
    #         exp_name=f"vae_beta{beta}",
    #         beta=beta,
    #         seeds=SEEDS,
    #         real_path=REAL_PATH,
    #     )

    for dim_latent in VAE_GRID.get("dim_latent"):
        run_vae_experiment(
            exp_name=f"vae_latent{dim_latent}",
            dim_latent=dim_latent,
            seeds=SEEDS,
            real_path=REAL_PATH,
        )


if __name__ == "__main__":
    main()
