"""Lazy HDF5 reads and strict pairing; MATLAB writes dimensions in reverse order."""
from pathlib import Path

import h5py
import numpy as np
import torch

from .common import SPLITS, VARIANTS, dataset_path, read_json, write_json


def check_file(path, c, require_gt=True):
    with h5py.File(path, "r") as f:
        if f.attrs.get("experiment_id") != c["experiment_id"]:
            raise ValueError(f"Wrong experiment identity: {path}")
        if not f.attrs.get("complete", 0):
            raise ValueError(f"Incomplete dataset: {path}; rerun prepare")
        x = f["input"]
        if x.ndim != 4 or x.shape[1] != 32 or min(x.shape[-2:]) < 32:
            raise ValueError(f"Expected input [N,32,H,W], H,W>=32: {path}: {x.shape}")
        if require_gt and ("gt" not in f or f["gt"].shape != (x.shape[0], 1, *x.shape[2:])):
            raise ValueError(f"Missing/malformed gt [N,1,H,W]: {path}")
        ids = np.asarray(f["sample_id"]).reshape(-1)
        if len(ids) != x.shape[0] or len(set(ids.tolist())) != len(ids):
            raise ValueError(f"Missing or duplicate sample IDs: {path}")
        return ids


def audit_pairs(c):
    reference_ids = {}
    for depth in c["depths_sls"]:
        for split in SPLITS:
            a, b = [dataset_path(c, v, depth, split) for v in VARIANTS]
            ids_a, ids_b = check_file(a, c), check_file(b, c)
            if not np.array_equal(ids_a, ids_b):
                raise ValueError(f"Unpaired object IDs: {a}, {b}")
            if len(ids_a) != c["data"]["counts"][split]:
                raise ValueError(f"Unexpected sample count: {a}")
            if split in reference_ids and not np.array_equal(reference_ids[split], ids_a):
                raise ValueError("Object IDs must also match across depths")
            reference_ids[split] = ids_a
            with h5py.File(a, "r") as fa, h5py.File(b, "r") as fb:
                for i in range(len(ids_a)):
                    if not np.array_equal(fa["gt"][i], fb["gt"][i]):
                        raise ValueError(f"Ground truth differs between PSF variants at {depth}, {split}, {i}")
                    for f in (fa, fb):
                        if not np.isfinite(f["input"][i]).all() or not np.isfinite(f["gt"][i]).all():
                            raise ValueError(f"Nonfinite data in {f.filename}")
                        if np.min(f["gt"][i]) < 0 or np.max(f["gt"][i]) > 1.00001:
                            raise ValueError("Ground truth must be in [0,1]")
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            if np.intersect1d(reference_ids[a], reference_ids[b]).size:
                raise ValueError(f"Split leakage: {a}/{b}")
    manifest_file = Path(c["run_dir"]) / "objects/manifest.json"
    if not manifest_file.exists():
        raise FileNotFoundError(f"Missing source split manifest: {manifest_file}")
    objects = read_json(manifest_file)["objects"]
    bounds = {}
    for split in SPLITS:
        entries = [e for e in objects if e["split"] == split]
        expected = [e["sample_id"] for e in entries]
        if not np.array_equal(expected, reference_ids[split]):
            raise ValueError(f"Manifest/sample ID mismatch: {split}")
        bounds[split] = (min(e["z_first"] for e in entries), max(e["z_last"] for e in entries))
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i+1:]:
            if max(bounds[a][0], bounds[b][0]) <= min(bounds[a][1], bounds[b][1]):
                raise ValueError(f"Source z-block leakage: {a}/{b}")
    write_json(Path(c["run_dir"]) / "pair_audit.json", {"passed": True, "source_z_bounds": bounds,
               "checks": ["exact paired targets", "IDs across PSF variants and depths", "disjoint source z blocks", "finite data"]})


def training_scales(c, depth):
    """One input scale shared by both arms, estimated exclusively from training."""
    paths = [dataset_path(c, variant, depth, "train") for variant in VARIANTS]
    signature = [[p.stat().st_size, p.stat().st_mtime_ns] for p in paths]
    cache = Path(c["run_dir"]) / "normalization" / f"{float(depth):g}sls.json"
    if cache.exists():
        saved = read_json(cache)
        if saved.get("training_file_stats") == signature:
            return saved
    maximum = 0.0
    for variant in VARIANTS:
        with h5py.File(dataset_path(c, variant, depth, "train"), "r") as f:
            for i in range(len(f["input"])):
                maximum = max(maximum, float(np.max(f["input"][i])))
    if not np.isfinite(maximum) or maximum <= 0:
        raise ValueError("Nonpositive training normalization scale")
    scales = {"input": maximum, "target": 1.0, "source": "maximum over both training arms only",
              "training_file_stats": signature}
    write_json(cache, scales)
    return scales


class MeasurementDataset(torch.utils.data.Dataset):
    def __init__(self, path, scale):
        self.path = str(path)
        self.scale = float(scale)
        self._f = None
        with h5py.File(path, "r") as f:
            self.length = len(f["input"])
            self.has_gt = "gt" in f

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        if self._f is None:
            self._f = h5py.File(self.path, "r")
        x = np.asarray(self._f["input"][i], dtype=np.float32) / self.scale
        y = np.asarray(self._f["gt"][i], dtype=np.float32) if self.has_gt else np.zeros((1, *x.shape[-2:]), np.float32)
        return torch.from_numpy(x), torch.from_numpy(y), i

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_f"] = None
        return state

    def close(self):
        if self._f is not None:
            self._f.close()
            self._f = None
