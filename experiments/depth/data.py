"""Lazy HDF5 reads and strict pairing; MATLAB writes dimensions in reverse order."""
from contextlib import ExitStack
from pathlib import Path

import h5py
import numpy as np
import torch

from .common import CONDITIONS, SPLITS, VARIANTS, dataset_path, read_json, write_json


def text_attribute(f, key):
    value = f.attrs.get(key)
    # h5writeatt releases can produce fixed-length byte strings; h5py's own
    # variable-length attributes decode to str. Accept both representations.
    return value.decode("utf-8") if isinstance(value, bytes) else value


def check_file(path, c, require_gt=True):
    with h5py.File(path, "r") as f:
        if text_attribute(f, "experiment_id") != c["experiment_id"]:
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


def read_noise_schedule(c, variant, depth, split, n):
    group_size = c["camera"]["legacy_batch_samples"] if CONDITIONS[variant]["camera"] == "legacy" else 1
    groups = np.arange(n, dtype=np.int64) // group_size + 1
    seeds = (c["data"]["seed"] + 700000000 + round(depth*100000) +
             (SPLITS.index(split)+1)*1000000 + groups) % (2**32-1)
    return groups, seeds


def audit_pairs(c):
    reference_ids = {}
    for depth in c["depths_sls"]:
        for split in SPLITS:
            paths = [dataset_path(c, v, depth, split) for v in VARIANTS]
            all_ids = [check_file(p, c) for p in paths]
            ids_a = all_ids[0]
            if any(not np.array_equal(ids_a, ids) for ids in all_ids[1:]):
                raise ValueError(f"Unpaired object IDs at {depth}, {split}")
            if len(ids_a) != c["data"]["counts"][split]:
                raise ValueError(f"Unexpected sample count: {paths[0]}")
            if split in reference_ids and not np.array_equal(reference_ids[split], ids_a):
                raise ValueError("Object IDs must also match across depths")
            reference_ids[split] = ids_a
            with ExitStack() as stack:
                files = [stack.enter_context(h5py.File(p, "r")) for p in paths]
                reference = stack.enter_context(h5py.File(dataset_path(c, "legacy", c["depths_sls"][0], split), "r"))
                for variant, f in zip(VARIANTS, files):
                    expected = {"variant": variant, **CONDITIONS[variant]}
                    if any(text_attribute(f, key) != value for key, value in expected.items()):
                        raise ValueError(f"Simulator factor metadata mismatch: {f.filename}")
                    groups, seeds = read_noise_schedule(c, variant, depth, split, len(ids_a))
                    for key, value in (("read_noise_group", groups), ("read_noise_seed", seeds)):
                        if key not in f or not np.array_equal(np.asarray(f[key]).reshape(-1), value):
                            raise ValueError(f"Incorrect {key}: {f.filename}")
                for i in range(len(ids_a)):
                    gt = reference["gt"][i]
                    for f in files:
                        if not np.array_equal(gt, f["gt"][i]):
                            raise ValueError(f"Ground truth differs across conditions/depths at {depth}, {split}, {i}")
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
               "checks": ["exact paired targets", "IDs across conditions and depths", "disjoint source z blocks", "camera factors and noise grouping", "finite data"]})


def training_scales(c, depth):
    """One input scale shared by all conditions, estimated exclusively from training."""
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
    scales = {"input": maximum, "target": 1.0, "source": "maximum over all training conditions only",
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
