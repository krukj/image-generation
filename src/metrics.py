import os

import torch
import torch_fidelity


def calculate_metrics(real_path: str, fake_path: str, min_images=2000):
    n_real = len([f for f in os.listdir(real_path) if f.endswith((".jpg", ".png"))])
    n_fake = len([f for f in os.listdir(fake_path) if f.endswith((".jpg", ".png"))])

    if min(n_real, n_fake) < min_images:
        print("Too little images, FID will contain noise")

    return torch_fidelity.calculate_metrics(
        input1=fake_path,
        input2=real_path,
        cuda=torch.cuda.is_available(),
        isc=True,
        fid=True,
        prc=True,
        prc_neighborhood_k=5,
    )
