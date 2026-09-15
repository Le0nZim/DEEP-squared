"""Evaluate both training arms on each identical test stack; no test-time fitting."""
import csv
from pathlib import Path

import h5py
import numpy as np
import torch
from skimage.metrics import structural_similarity

from Modules.model import UNet
from .common import VARIANTS, checkpoint_dir, dataset_path, depth_tag, read_json, write_json
from .data import MeasurementDataset, check_file


def metrics(prediction, target):
    prediction, target = np.asarray(prediction, np.float64), np.asarray(target, np.float64)
    mse = float(np.mean((prediction - target) ** 2))
    # 120 dB is an explicit numerical ceiling for zero error, not an infinite JSON value.
    return {"mse": mse, "psnr": float(-10 * np.log10(max(mse, 1e-12))),
            "ssim": float(structural_similarity(target, prediction, data_range=1.0))}


def evaluate_all(c, device):
    device = torch.device(device)
    out = Path(c["run_dir"]) / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    rows, missing_real = [], []
    for depth in c["depths_sls"]:
        domains = [(v, dataset_path(c, v, depth, "test")) for v in VARIANTS]
        real = Path(c["run_dir"]) / "experimental" / f"{depth_tag(depth)}.h5"
        if real.exists():
            domains.append(("experimental", real))
        else:
            missing_real.append(depth)
        for seed in c["seeds"]:
            for variant in VARIANTS:
                checkpoint = checkpoint_dir(c, variant, depth, seed) / "best.pt"
                if not checkpoint.exists():
                    raise FileNotFoundError(f"Missing model: {checkpoint}; finish train first")
                completion = checkpoint.parent / "complete.json"
                if not completion.exists():
                    raise RuntimeError(f"Training incomplete: {checkpoint.parent}; resume train before comparison")
                done = read_json(completion)
                if done.get("experiment_id") != c["experiment_id"] or done.get("epochs") != c["training"]["epochs"]:
                    raise ValueError(f"Completed epoch count/provenance mismatch: {completion}")
                saved = torch.load(checkpoint, map_location=device, weights_only=False)
                if saved["experiment_id"] != c["experiment_id"]:
                    raise ValueError(f"Wrong checkpoint: {checkpoint}")
                model = UNet(n_classes=1, n_patterns=32).to(device)
                model.load_state_dict(saved["model"], strict=True)
                model.eval()
                for domain, path in domains:
                    ids = check_file(path, c, require_gt=domain != "experimental")
                    data = MeasurementDataset(path, saved["input_scale"])
                    predictions = []
                    with torch.no_grad():
                        for i in range(len(data)):
                            x, y, _ = data[i]
                            pred = model(x[None].to(device))[0, 0].cpu().numpy()
                            if not np.isfinite(pred).all():
                                raise FloatingPointError(f"Nonfinite prediction: {checkpoint}")
                            row = {"depth_sls": depth, "depth_um": depth * 50, "train_psf": variant,
                                   "test_domain": domain, "seed": seed, "sample_id": int(ids[i]),
                                   "has_ground_truth": int(data.has_gt)}
                            if data.has_gt:
                                row.update(metrics(pred, y[0].numpy()))
                            rows.append(row)
                            # All real FOVs; a fixed, preselected first subset for simulation panels.
                            if domain == "experimental" or i < c["evaluation"]["save_examples"]:
                                predictions.append(pred)
                    stem = f"{depth_tag(depth)}_{variant}_seed{seed}_on_{domain}"
                    np.savez_compressed(out / f"{stem}.npz", prediction=np.asarray(predictions),
                                        sample_id=ids[:len(predictions)])
                    data.close()
                    print(f"EVALUATED {stem}: {len(ids)} samples", flush=True)
    fields = ["depth_sls", "depth_um", "train_psf", "test_domain", "seed", "sample_id",
              "has_ground_truth", "mse", "psnr", "ssim"]
    with open(out / "per_sample.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    write_json(out / "coverage.json", {"missing_experimental_depths_sls": missing_real,
               "experimental_gt_note": "No matched ground truth is supplied by the mouse adapter; no reference metrics are fabricated.",
               "metric_range": 1.0, "prediction_clipping": False})
