"""Configuration, provenance and atomic files. No GPU dependency at import time."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = {
    "legacy": {"psf": "legacy", "camera": "legacy"},
    "psf_only": {"psf": "corrected", "camera": "legacy"},
    "camera_only": {"psf": "legacy", "camera": "corrected"},
    "corrected": {"psf": "corrected", "camera": "corrected"},
}
VARIANTS = tuple(CONDITIONS)
SPLITS = ("train", "val", "test")
ASSETS = {
    "BV_03102021.mat": "213e005f11eec07907a62c15cd5e1898",
    "dmd_exp_tfm_mouse_20201224.mat": "8634a1a36ba213f91c5793601132a5dd",
}


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def file_hash(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def source_hashes():
    paths = [ROOT / "depth_study.py", ROOT / "Modules/model.py"]
    paths += list((ROOT / "experiments/depth").rglob("*.py"))
    paths += list((ROOT / "experiments/depth/matlab").rglob("*.m"))
    paths += list((ROOT / "fwd_model").rglob("*.m"))
    paths += list((ROOT / "experiments/depth/matlab").rglob("PROVENANCE.json"))
    paths += [ROOT / "fwd_model/_emhist/emhist_29-Apr-2021_02_09_25.mat"]
    paths = [p for p in paths if "tests" not in p.parts]
    return {p.relative_to(ROOT).as_posix(): file_hash(p) for p in sorted(set(paths))}


def verify_frozen_sources():
    for name in ("legacy_mc", "legacy_camera"):
        folder = ROOT / "experiments/depth/matlab" / f"+{name}"
        manifest = read_json(folder / "PROVENANCE.json")
        for file, expected in manifest["files"].items():
            if file_hash(folder / file) != expected:
                raise ValueError(f"Frozen legacy source changed: {folder/file}")
        if "lut_path" in manifest and file_hash(ROOT / manifest["lut_path"]) != manifest["lut_sha256"]:
            raise ValueError("Frozen legacy camera table changed")


def configuration(path):
    verify_frozen_sources()
    c = read_json(path)
    required = {"name", "data_dir", "output_dir", "depths_sls", "seeds", "data", "mc", "optics", "training", "signal_mode", "extension", "evaluation"}
    missing = required - c.keys()
    if missing:
        raise ValueError(f"Missing configuration fields: {sorted(missing)}")
    if c["signal_mode"] not in ("paper_peak", "fixed_source"):
        raise ValueError("signal_mode must be paper_peak or fixed_source")
    if c.get("conditions") != CONDITIONS:
        raise ValueError("Use all four explicit PSF/camera conditions from the updated supplied configs")
    camera = c.get("camera", {})
    for key in ("legacy_batch_samples", "lut_trials"):
        value = camera.get(key)
        if type(value) is not int or value < 1:
            raise ValueError(f"camera.{key} must be a positive integer")
    depths = c["depths_sls"]
    if not depths or len(depths) != len(set(depths)) or any(d <= 0 for d in depths):
        raise ValueError("depths_sls must contain unique positive depths")
    if not c["seeds"] or len(set(c["seeds"])) != len(c["seeds"]):
        raise ValueError("seeds must be nonempty and unique")
    if c["data"]["n_patterns"] != 32:
        raise ValueError("This protocol uses the paper's 32 patterns; do not silently alter it")
    if c["data"]["pattern_first_matlab"] != 21:
        raise ValueError("The source's mouse adapter uses patterns 21:52 (MATLAB indexing)")
    for key in SPLITS:
        if c["data"]["counts"][key] < 1:
            raise ValueError(f"Empty {key} split")
    for key in ("photons_per_batch", "batches", "max_hops"):
        if c["mc"][key] < 1:
            raise ValueError(f"mc.{key} must be positive")
    if c["training"]["loss"] not in ("generalized_kl", "mse", "legacy_kl"):
        raise ValueError("Unsupported loss")
    if c["training"]["epochs"] < 1 or c["training"]["batch_size"] < 1:
        raise ValueError("epochs and batch_size must be positive")
    if c["optics"]["mus_cm_inv"] != 200:
        raise ValueError("The mouse data depth mapping requires mus=200 cm^-1 (1 SLS=50 um)")
    fractions = c["data"]["split_fractions"]
    if len(fractions) != 3 or any(f <= 0 for f in fractions) or abs(sum(fractions)-1) > 1e-9:
        raise ValueError("split_fractions must contain three positive fractions summing to one")
    if c["extension"]["unmeasured_peak_reference_sls"] not in (2, 4, 6, 7, 8):
        raise ValueError("The extension calibration reference must exist in the mouse adapter")
    if c["optics"]["axial_planes"] % 2 or c["optics"]["axial_planes"] < 4:
        raise ValueError("optics.axial_planes must be even and at least 4")
    if c["evaluation"]["save_examples"] < 1:
        raise ValueError("evaluation.save_examples must be positive")
    # Stable sources + full config, not a timestamp or a mutable git branch name.
    c["source_hashes"] = source_hashes()
    signature = hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()
    c["experiment_id"] = signature
    c["repo_root"] = str(ROOT)
    c["data_dir"] = str((ROOT / c["data_dir"]).resolve())
    c["run_dir"] = str((ROOT / c["output_dir"] / f"{c['name']}-{signature[:12]}").resolve())
    return c


def depth_tag(depth):
    return f"{float(depth):g}sls"


def dataset_path(c, variant, depth, split):
    return Path(c["run_dir"]) / "datasets" / depth_tag(depth) / variant / f"{split}.h5"


def checkpoint_dir(c, variant, depth, seed):
    return Path(c["run_dir"]) / "models" / depth_tag(depth) / variant / f"seed{seed}"


def save_resolved(c):
    out = Path(c["run_dir"])
    out.mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    write_json(out / "resolved.json", c)
    write_json(out / "provenance.json", {"git_commit": commit, "source_hashes": c["source_hashes"], "experiment_id": c["experiment_id"]})
    return out / "resolved.json"
