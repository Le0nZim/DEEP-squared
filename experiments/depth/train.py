"""Train the repository's scSE-UNet with paired settings and epoch-level resume."""
import csv
import hashlib
import os
import random
import platform
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from Modules.model import UNet
from .common import VARIANTS, checkpoint_dir, dataset_path, write_json
from .data import MeasurementDataset, check_file, training_scales


def loss_value(prediction, target, name):
    if name == "generalized_kl":
        # I-divergence for nonnegative intensities, including the linear terms.
        p = prediction + 1e-8
        # Split the logarithms: xlogy(0, 0/p) has a NaN derivative in PyTorch.
        return (torch.xlogy(target, target) - target * torch.log(p) - target + p).mean()
    if name == "mse":
        return F.mse_loss(prediction, target)
    if name == "legacy_kl":
        # Deliberately reproduces run.py's incorrect use of KLDivLoss.
        return F.kl_div(prediction, target, reduction="mean")
    raise ValueError(name)


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def state_hash(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # CUDA bilinear interpolation can be nondeterministic; do not conceal that.
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_rng(generator):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "loader": generator.get_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def set_rng(state, generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    generator.set_state(state["loader"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def train_one(c, variant, depth, seed, device):
    settings = c["training"]
    folder = checkpoint_dir(c, variant, depth, seed)
    folder.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        check_file(dataset_path(c, variant, depth, split), c)
    scale = training_scales(c, depth)
    seed_all(seed)
    generator = torch.Generator().manual_seed(seed)
    model = UNet(n_classes=1, n_patterns=32).to(device)
    initial_hash = state_hash(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"], betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.3)
    history, best, start = [], float("inf"), 0
    last = folder / "last.pt"
    if last.exists():
        # Only load checkpoints produced by this runner, never untrusted downloads.
        saved = torch.load(last, map_location=device, weights_only=False)
        if saved["experiment_id"] != c["experiment_id"] or saved["input_scale"] != scale["input"]:
            raise ValueError(f"Checkpoint provenance mismatch: {last}")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        history, best, start = saved["history"], saved["best_val"], saved["epoch"]
        set_rng(saved["rng"], generator)
        if start >= settings["epochs"]:
            if not (folder / "best.pt").exists():
                raise FileNotFoundError(f"Missing best checkpoint in completed run: {folder}")
            write_json(folder / "complete.json", {"experiment_id": c["experiment_id"], "epochs": start})
            print(f"COMPLETE {depth:g} SLS / {variant} / seed {seed}", flush=True)
            return
    train = MeasurementDataset(dataset_path(c, variant, depth, "train"), scale["input"])
    val = MeasurementDataset(dataset_path(c, variant, depth, "val"), scale["input"])
    train_loader = DataLoader(train, batch_size=settings["batch_size"], shuffle=True,
                              num_workers=settings["workers"], generator=generator, drop_last=False)
    val_loader = DataLoader(val, batch_size=settings["batch_size"], shuffle=False,
                            num_workers=settings["workers"], drop_last=False)
    meta = {"experiment_id": c["experiment_id"], "variant": variant, "depth_sls": depth,
            "seed": seed, "architecture": "Modules.model.UNet (scSE)", "initial_state_sha256": initial_hash,
            "input_scale": scale["input"], "target_scale": 1.0, "loss": settings["loss"],
            "torch": str(torch.__version__), "device": str(device), "dtype": "float32",
            "python": platform.python_version(), "numpy": np.__version__,
            "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor()}
    write_json(folder / "metadata.json", meta)
    for epoch in range(start + 1, settings["epochs"] + 1):
        values = {}
        for phase, loader in (("train", train_loader), ("val", val_loader)):
            model.train(phase == "train")
            total, count = 0.0, 0
            with torch.set_grad_enabled(phase == "train"):
                for x, y, _ in loader:
                    x, y = x.to(device), y.to(device)
                    if phase == "train":
                        optimizer.zero_grad(set_to_none=True)
                    prediction = model(x)
                    loss = loss_value(prediction, y, settings["loss"])
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite {phase} loss: {folder}, epoch {epoch}")
                    if phase == "train":
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
                        optimizer.step()
                    total += loss.item() * len(x)
                    count += len(x)
            values[phase] = total / count
        scheduler.step()
        history.append({"epoch": epoch, **values})
        improved = values["val"] < best
        best = min(best, values["val"])
        checkpoint = {**meta, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "scheduler": scheduler.state_dict(), "epoch": epoch, "history": history,
                      "best_val": best, "rng": get_rng(generator)}
        if improved:
            atomic_save(checkpoint, folder / "best.pt")
        atomic_save(checkpoint, last)
        with open(folder / "history.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=("epoch", "train", "val"))
            writer.writeheader()
            writer.writerows(history)
        print(f"{depth:g} SLS {variant} seed={seed} epoch={epoch}/{settings['epochs']} "
              f"train={values['train']:.6g} val={values['val']:.6g}", flush=True)
    train.close()
    val.close()
    write_json(folder / "complete.json", {"experiment_id": c["experiment_id"], "epochs": settings["epochs"]})


def train_all(c, device):
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Fix the PyTorch/driver installation or explicitly use --device cpu")
    for depth in c["depths_sls"]:
        for seed in c["seeds"]:
            for variant in VARIANTS:
                train_one(c, variant, depth, seed, device)
