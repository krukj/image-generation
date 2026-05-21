import gc
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.datasets import ImageFolder

from src.config import (DATA_PROCESSED_TORCH_DIR, DCGAN_DEFAULT, DCGAN_GRID,
                        NGPU, RESULTS_PATH, TARGET_SIZE, WORKERS)
from src.models.dcgan import Discriminator, Generator
from src.scripts.utils import (compute_metrics, save_images, set_seed,
                               setup_logger, weights_init)

device = torch.device("mps" if (torch.mps.is_available() and NGPU > 0) else "cpu")
logger = setup_logger()


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

def _create_g_and_d(nz: int):
    netG = Generator(NGPU, nz=nz).to(device)
    netD = Discriminator(NGPU).to(device)

    if (device.type == "mps") and (NGPU > 1):
        netG = nn.DataParallel(netG, list(range(NGPU)))
        netD = nn.DataParallel(netD, list(range(NGPU)))

    netG.apply(weights_init)
    netD.apply(weights_init)

    return netG, netD


def _generate_fid_images(netG, nz, out_dir: Path, n: int = 2000):
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(out_dir.glob("*.png")))
    if existing >= n:
        return
    needed = n - existing
    bs = 64
    with torch.no_grad():
        for batch_start in range(0, needed, bs):
            cnt = min(bs, needed - batch_start)
            noise = torch.randn(cnt, nz, 1, 1, device=device)
            imgs = netG(noise).detach().cpu()
            for j, img in enumerate(imgs):
                vutils.save_image(
                    img,
                    out_dir / f"gen_{existing + batch_start + j:05d}.png",
                    normalize=True,
                )
    clear_memory()


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
    lr: float,
    beta: float,
    nz: int,
    d_to_g_ratio: int,
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

    netG, netD = _create_g_and_d(nz=nz)

    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    real_label = 0.9
    fake_label = 0

    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta, 0.999))

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

    G_losses, D_losses = [], []
    metrics_history = []
    iters = 0
    t0 = time.time()

    logger.info(f"Experiment: {exp_name} | seed = {seed}")
    logger.info(f"Params: nz={nz}, lr={lr}, beta={beta}, d_to_g_ratio={d_to_g_ratio}")

    for epoch in range(num_epochs):

        for i, data in enumerate(dataloader, 0):

            for _ in range(d_to_g_ratio):
                netD.zero_grad()
                real_cpu = data[0].to(device)
                b_size = real_cpu.size(0)
                label = torch.full(
                    size=(b_size,),
                    fill_value=real_label,
                    dtype=torch.float,
                    device=device,
                )
                output = netD(real_cpu).view(-1)

                errD_real = criterion(output, label)
                errD_real.backward()
                D_x = output.mean().item()

                noise = torch.randn(b_size, nz, 1, 1, device=device)
                fake = netG(noise)
                label.fill_(fake_label)
                output = netD(fake.detach()).view(-1)
                errD_fake = criterion(output, label)
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                errD = errD_real + errD_fake
                optimizerD.step()

            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            if i % 50 == 0:
                elapsed = time.time() - t0
                print(
                    f"[{epoch}/{num_epochs}][{i}/{len(dataloader)}]"
                    f" Loss_D: {errD.item():.4f}  Loss_G: {errG.item():.4f}"
                    f" D(x): {D_x:.4f}  D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}"
                    f" ({elapsed:.0f}s)"
                )

            G_losses.append(errG.item())
            D_losses.append(errD.item())

            iters += 1

        save_images(netG, fixed_noise, images_dir, tag=f"epoch{epoch:03d}")

    logger.info("Calculating metrics")
    singles_dir = images_dir / f"epoch{epoch:03d}" / "singles"
    _generate_fid_images(netG, nz, singles_dir, n=2000)
    metrics = compute_metrics(real_path, singles_dir)
    metrics["epoch"] = epoch
    metrics_history.append(metrics)
    print(
        f"FID={metrics['fid']} | IS={metrics['is_mean']}±{metrics['is_std']} | "
        f"P={metrics['precision']} | R={metrics['recall']}"
    )
    clear_memory()

    result = {
        "exp_name": exp_name,
        "seed": seed,
        "params": {
            "lr": lr,
            "beta": beta,
            "nz": nz,
            "d_to_g_ratio": d_to_g_ratio,
            "num_epochs": num_epochs,
        },
        "training_time": round(time.time() - t0, 1),
        "final_metrics": metrics_history[-1] if metrics_history else {},
        "metrics_history": metrics_history,
        "loss_G": G_losses,
        "loss_D": D_losses,
    }

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    torch.save(netG.state_dict(), run_dir / "netG.pt")
    torch.save(netD.state_dict(), run_dir / "netD.pt")

    return result


def run_dcgan_experiment(
    seeds: list[int],
    nz: int = DCGAN_DEFAULT.get("nz"),
    d_to_g_ratio: int = DCGAN_DEFAULT.get("d_to_g_ratio"),
    lr: float = DCGAN_DEFAULT.get("lr"),
    beta: float = DCGAN_DEFAULT.get("beta"),
    num_epochs: int = DCGAN_DEFAULT.get("num_epochs"),
    batch_size: int = DCGAN_DEFAULT.get("batch_size"),
    workers: int = WORKERS,
    dataroot: str | Path = Path(DATA_PROCESSED_TORCH_DIR),
    real_path: str | Path = Path(DATA_PROCESSED_TORCH_DIR),
    results_root: str | Path = Path(RESULTS_PATH),
    exp_name: str | None = None,
) -> list[dict]:

    if exp_name is None:
        exp_name = f"dcgan_nz{nz}ratio{d_to_g_ratio}"

    results_root = Path(results_root)
    results_root.mkdir(exist_ok=True)

    all_results = []
    for seed in seeds:
        r = run_single_experiment(
            exp_name=exp_name,
            seed=seed,
            lr=lr,
            beta=beta,
            nz=nz,
            d_to_g_ratio=d_to_g_ratio,
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
    REAL_PATH = Path(DATA_PROCESSED_TORCH_DIR)/ "cats"
    SEEDS = [1, 2, 3]

    # run_dcgan_experiment(
    #     exp_name="dcgan_default",
    #     nz=DCGAN_DEFAULT.get("nz"),
    #     d_to_g_ratio=DCGAN_DEFAULT.get("d_to_g_ratio"),
    #     seeds=SEEDS,
    #     real_path=REAL_PATH,
    # )

    # for nz in DCGAN_GRID.get("nz"):
    #     run_dcgan_experiment(
    #         exp_name=f"dcgan_nz{nz}",
    #         nz=nz,
    #         d_to_g_ratio=DCGAN_DEFAULT.get("d_to_g_ratio"),
    #         seeds=SEEDS,
    #         real_path=REAL_PATH,
    #     )

    for ratio in DCGAN_GRID.get("d_to_g_ratio"):
        run_dcgan_experiment(
            exp_name=f"dcgan_ratio{ratio}",
            nz=DCGAN_DEFAULT.get("nz"),
            d_to_g_ratio=ratio,
            seeds=SEEDS,
            real_path=REAL_PATH,
        )


if __name__ == "__main__":
    main()
