import gc
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from diffusers import DDIMPipeline, DDIMScheduler, UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.utils import make_image_grid
from torchvision.datasets import ImageFolder

from src.config import (
    DATA_PROCESSED_TORCH_DIR,
    DDIM_DEFAULT,
    DDIM_GRID,
    NGPU,
    RESULTS_PATH,
    TARGET_SIZE,
    WORKERS,
)
from src.scripts.utils import compute_metrics, set_seed, setup_logger

device = torch.device("mps" if (torch.mps.is_available() and NGPU > 0) else "cpu")
logger = setup_logger()


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _aggregate_seeds(exp_name: str, results: list[dict]) -> dict:
    keys = ["fid", "is_mean", "precision", "recall"]
    agg = {
        "exp_name": exp_name,
        "n_seeds": len(results),
        "params": results[0]["params"],
    }
    for k in keys:
        vals = [r["final_metrics"].get(k, float("nan")) for r in results]
        t = torch.tensor(vals)
        agg[f"{k}_mean"] = round(t.mean().item(), 4)
        agg[f"{k}_std"] = round(t.std().item(), 4)
    return agg


def _generate_fid_images(
    pipeline: DDIMPipeline, out_dir: Path, n: int = 2000, batch_size: int = 32
):
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(out_dir.glob("*.png")))
    if existing >= n:
        return
    needed = n - existing
    idx = existing
    generated = 0
    with torch.no_grad():
        while generated < needed:
            cnt = min(batch_size, needed - generated)
            images = pipeline(
                batch_size=cnt, generator=torch.Generator(device="cpu"), num_inference_steps=50
            ).images
            for img in images:
                img.save(out_dir / f"gen_{idx:05d}.png")
                idx += 1
            generated += len(images)
    clear_memory()


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


def save_images_ddim(batch_size, pipeline: DDIMPipeline, out_dir: Path, tag: str):
    images = pipeline(
        batch_size=batch_size, generator=torch.Generator(device="cpu")
    ).images
        
    n = len(images)
    cols = int(math.sqrt(n))
    rows = n // cols
    images_grid = images[:rows * cols]

    image_grid = make_image_grid(images=images_grid, rows=rows, cols=cols)

    out_dir.mkdir(parents=True, exist_ok=True)

    image_grid.save(out_dir / f"{tag}_grid.png")

    singles_dir = out_dir / "singles"
    singles_dir.mkdir(exist_ok=True)
    for idx, img in enumerate(images):
        img.save(singles_dir / f"{tag}_{idx:04d}.png")


def run_single_experiment(
    exp_name: str,
    seed: int,
    lr: float,
    lr_warmup_steps: int,
    beta_schedule: str,
    num_train_timesteps: int,
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
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=workers
    )

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
    ).to(device)
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=num_train_timesteps, beta_schedule=beta_schedule
    )
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps,
        num_training_steps=len(dataloader) * num_epochs,
    )

    logger.info(f"Experiment: {exp_name} | seed = {seed}")
    logger.info(f"Params: num_train_timesteps={num_train_timesteps}, lr={lr}")

    losses = []
    metrics_history = []
    iters = 0
    t0 = time.time()
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            clean_images = data[0].to(device)
            noise = torch.randn(clean_images.shape, device=clean_images.device)
            bs = clean_images.shape[0]
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bs,),
                device=clean_images.device,
                dtype=torch.int64,
            )

            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
            loss = F.mse_loss(noise_pred, noise)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            loss_detached = loss.detach().item()
            if i % 50 == 0:
                elapsed = time.time() - t0
                logger.info(
                    f"[{epoch}/{num_epochs}][{i}/{len(dataloader)}]"
                    f" Loss: {loss_detached:.4f}"
                    f" ({elapsed:.0f}s)"
                )
            losses.append(loss_detached)

            iters += 1

        pipeline = DDIMPipeline(unet=model, scheduler=noise_scheduler)
        save_images_ddim(
            batch_size=batch_size,
            pipeline=pipeline,
            out_dir=images_dir,
            tag=f"epoch{epoch:03d}",
        )
        del pipeline
        clear_memory()


    logger.info("Calculating metrics")
    pipeline = DDIMPipeline(unet=model, scheduler=noise_scheduler)
    singles_dir = images_dir / f"epoch{epoch:03d}" / "singles"
    _generate_fid_images(pipeline, singles_dir, n=2000)
    metrics = compute_metrics(real_path, singles_dir)
    metrics["epoch"] = epoch
    metrics_history.append(metrics)
    print(
        f"FID={metrics['fid']} | IS={metrics['is_mean']}±{metrics['is_std']} | "
        f"P={metrics['precision']} | R={metrics['recall']}"
    )
    del pipeline
    clear_memory()

    result = {
        "exp_name": exp_name,
        "seed": seed,
        "params": {
            "lr": lr,
            "beta_schedule": beta_schedule,
            "num_train_timesteps": num_train_timesteps,
            "num_epochs": num_epochs,
        },
        "training_time": round(time.time() - t0, 1),
        "final_metrics": metrics_history[-1] if metrics_history else {},
        "metrics_history": metrics_history,
        "loss": losses,
    }

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    torch.save(model.state_dict(), run_dir / "DDIM.pt")

    return result


def run_ddim_experiment(
    seeds: list[int],
    lr: float = DDIM_DEFAULT.get("lr"),
    beta_schedule: float = DDIM_DEFAULT.get("beta_schedule"),
    num_train_timesteps: float = DDIM_DEFAULT.get("num_train_timesteps"),
    lr_warmup_steps: int = DDIM_DEFAULT.get("lr_warmup_steps"),
    num_epochs: int = DDIM_DEFAULT.get("num_epochs"),
    batch_size: int = DDIM_DEFAULT.get("batch_size"),
    workers: int = WORKERS,
    dataroot: str | Path = Path(DATA_PROCESSED_TORCH_DIR),
    real_path: str | Path = Path(DATA_PROCESSED_TORCH_DIR),
    results_root: str | Path = Path(RESULTS_PATH),
    exp_name: str | None = None,
) -> list[dir]:

    if exp_name is None:
        exp_name = f"ddim_ntt{num_train_timesteps}betasched{beta_schedule}"

    results_root = Path(results_root)
    results_root.mkdir(exist_ok=True)

    all_results = []
    for seed in seeds:
        r = run_single_experiment(
            exp_name=exp_name,
            seed=seed,
            lr=lr,
            lr_warmup_steps=lr_warmup_steps,
            beta_schedule=beta_schedule,
            num_train_timesteps=num_train_timesteps,
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

    # run_ddim_experiment(
    #     exp_name="ddim_default",
    #     seeds=SEEDS,
    #     real_path=REAL_PATH,
    # )

    for beta_schedule in DDIM_GRID.get("beta_schedule"):
        run_ddim_experiment(
            exp_name=f"ddim_betas{beta_schedule}",
            beta_schedule=beta_schedule,
            seeds=SEEDS,
            real_path=REAL_PATH,
        )


if __name__ == "__main__":
    main()
