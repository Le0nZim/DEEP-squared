"""Setup, download, native MATLAB generation, sequential training, comparison."""
from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

from .common import ASSETS, ROOT, VARIANTS, configuration, file_hash, save_resolved, write_json


def plan(c):
    n = len(c["depths_sls"])
    h = c["data"]["crop_size"] or 326
    gib = len(VARIANTS)*n*sum(c["data"]["counts"].values())*33*h*h*4/1024**3
    print(f"Run directory: {c['run_dir']}")
    print(f"Depths: {c['depths_sls']} SLS; {[d*50 for d in c['depths_sls']]} um at mus=200 cm^-1")
    print(f"Trainings: {len(VARIANTS)*n*len(c['seeds'])} = 4 PSF/camera conditions x {n} depths x {len(c['seeds'])} seeds")
    for name, factors in c["conditions"].items():
        print(f"  {name}: PSFs={factors['psf']}, camera={factors['camera']}")
    print(f"Legacy read noise: one scalar per {c['camera']['legacy_batch_samples']} generated samples (separate from training batch size)")
    print(f"{c['training']['epochs']} epochs; scSE-UNet; 32 patterns; loss={c['training']['loss']}; signal={c['signal_mode']}")
    print(f"Raw dataset estimate: {gib:.1f} GiB at {h}x{h} (compression can reduce it); plus PSFs and checkpoints")
    print("Evaluation: all 16 train/test condition combinations; same experimental FOVs for all four models")
    return gib


def doctor(c, matlab):
    plan(c)
    missing = [p for p in ("torch", "numpy", "scipy", "h5py", "skimage", "matplotlib") if importlib.util.find_spec(p) is None]
    print(f"Python: {sys.version.split()[0]}; missing packages: {missing or 'none'}")
    print(f"MATLAB: {shutil.which(matlab) or 'NOT FOUND (required for native generation)'}")
    if importlib.util.find_spec("torch"):
        import torch
        print(f"PyTorch {torch.__version__}; CUDA available: {torch.cuda.is_available()}")
    data = Path(c["data_dir"])
    for name in ASSETS:
        print(f"Asset {name}: {'present' if (data/name).exists() else 'download required'}")
    print(f"Free disk near repository: {shutil.disk_usage(ROOT).free/1024**3:.1f} GiB")
    print("MATLAB requires Image Processing, Statistics and Machine Learning, and Parallel Computing toolboxes.")
    print("Use a CUDA-capable MATLAB GPU for generation; --device controls Python training only.")


def download(c):
    directory = Path(c["data_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    for name, expected in ASSETS.items():
        target = directory / name
        if target.exists():
            if file_hash(target, "md5") != expected:
                raise ValueError(f"Checksum mismatch: {target}. Move the incorrect file aside and rerun download.")
            print(f"VERIFIED {target}", flush=True)
            continue
        temporary = target.with_suffix(".mat.partial")
        url = f"https://zenodo.org/records/8161051/files/{name}?download=1"
        print(f"DOWNLOAD {url}", flush=True)
        offset = temporary.stat().st_size if temporary.exists() else 0
        headers = {"User-Agent": "DEEP2-depth-study/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=120) as response:
            append = response.status == 206 and response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")
            if response.status == 206 and not append:
                raise ValueError("Unexpected download range; remove the incomplete .partial file and retry")
            transferred = offset if append else 0
            next_progress = transferred + 64*1024**2
            with open(temporary, "ab" if append else "wb") as f:
                while block := response.read(8*1024**2):
                    f.write(block)
                    transferred += len(block)
                    if transferred >= next_progress:
                        print(f"  {name}: {transferred/1024**2:.0f} MiB", flush=True)
                        next_progress = transferred + 64*1024**2
        if file_hash(temporary, "md5") != expected:
            raise ValueError(f"Downloaded checksum mismatch: {temporary}; move the incomplete file aside and run download again")
        os.replace(temporary, target)
    write_json(directory / "assets.json", {"source": "https://zenodo.org/records/8161051", "md5": ASSETS})


def prepare(c, matlab):
    if not shutil.which(matlab):
        raise RuntimeError("MATLAB not found. Run on a workstation with MATLAB and the documented toolboxes; no Python surrogate is substituted.")
    for name, expected in ASSETS.items():
        path = Path(c["data_dir"]) / name
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run the download command first")
        if file_hash(path, "md5") != expected:
            raise ValueError(f"Asset checksum mismatch: {path}")
    resolved = save_resolved(c)
    quote = lambda s: str(s).replace("'", "''")
    expression = f"addpath('{quote(ROOT/'experiments/depth/matlab')}'); depth_generate('{quote(resolved)}');"
    log = Path(c["run_dir"]) / "prepare.log"
    with open(log, "a") as f:
        process = subprocess.Popen([matlab, "-batch", expression], cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end="", flush=True)
            f.write(line)
            f.flush()
        returncode = process.wait()
    if returncode:
        raise RuntimeError(f"MATLAB generation failed (exit {returncode}); see {log}. Rerun the same command to resume.")
    import h5py
    with h5py.File(Path(c["run_dir"]) / "axis_contract.h5", "r") as f:
        if f["input"].shape != (2, 3, 5, 7) or f["input"][1, 2, 1, 5] != 123:
            raise ValueError("MATLAB/Python HDF5 axis contract failed")
    from .data import audit_pairs
    audit_pairs(c)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["plan", "doctor", "download", "prepare", "audit", "train", "evaluate", "report", "run"])
    p.add_argument("--config", default="experiments/depth/configs/paper_depths.json")
    p.add_argument("--device", default="cuda:0", help="PyTorch device; honors CUDA_VISIBLE_DEVICES")
    p.add_argument("--matlab", default="matlab", help="MATLAB executable name or path")
    args = p.parse_args(argv)
    try:
        c = configuration(args.config)
        if args.command == "plan":
            plan(c)
            return
        if args.command == "doctor":
            doctor(c, args.matlab)
            return
        save_resolved(c)
        if args.command == "run" and not shutil.which(args.matlab):
            raise RuntimeError("MATLAB not found. Run doctor and use the documented MATLAB workstation; no training was launched.")
        if args.command in ("run", "train", "evaluate"):
            import torch
            device = torch.device(args.device)
            # Fail before downloading/generating days of data on a bad setup.
            if device.type == "cuda" and (not torch.cuda.is_available() or
                    (device.index or 0) >= torch.cuda.device_count()):
                raise RuntimeError("Requested CUDA device unavailable; fix PyTorch/visibility or explicitly select --device cpu")
        if args.command in ("download", "run"):
            download(c)
        if args.command in ("prepare", "run"):
            prepare(c, args.matlab)
        if args.command in ("audit", "train"):
            from .data import audit_pairs
            audit_pairs(c)
            print("PAIRED DATA AUDIT PASSED", flush=True)
        if args.command in ("train", "run"):
            from .train import train_all
            train_all(c, args.device)
        if args.command in ("evaluate", "run"):
            from .evaluate import evaluate_all
            evaluate_all(c, args.device)
        if args.command in ("report", "run"):
            from .report import make_report
            make_report(c)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        p.exit(2, f"ERROR: {exc}\n")
