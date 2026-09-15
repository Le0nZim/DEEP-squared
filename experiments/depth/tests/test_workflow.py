"""Meaningful CPU integration checks; fixtures are not scientific PSF results."""
import copy
import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from experiments.depth.common import CONDITIONS, ROOT, SPLITS, VARIANTS, configuration, dataset_path, write_json
from experiments.depth.data import audit_pairs, read_noise_schedule, training_scales
from experiments.depth.evaluate import evaluate_all, metrics
from experiments.depth.report import make_report, paired_contrasts
from experiments.depth.train import loss_value, state_hash, train_one


@pytest.fixture
def fixture_run(tmp_path):
    torch.set_num_threads(1)
    c = configuration(ROOT / "experiments/depth/configs/pilot.json")
    c.update(run_dir=str(tmp_path), depths_sls=[2], seeds=[100])
    c["data"]["counts"] = {s: 2 for s in SPLITS}
    c["training"].update(epochs=1, batch_size=2, workers=0)
    c["evaluation"]["save_examples"] = 1
    objects = []
    rng = np.random.default_rng(7)
    for i, split in enumerate(SPLITS):
        target = rng.random((2, 1, 32, 32), dtype=np.float32)
        measurement = np.repeat(target, 32, axis=1)
        ids = np.arange(i*2, i*2+2)
        for sample in ids:
            objects.append({"sample_id": int(sample), "split": split, "z_first": i*20+1, "z_last": i*20+5})
        for v in VARIANTS:
            path = dataset_path(c, v, 2, split)
            path.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(path, "w") as f:
                f.create_dataset("input", data=measurement)
                f.create_dataset("gt", data=target)
                f.create_dataset("sample_id", data=ids)
                f.attrs["experiment_id"] = c["experiment_id"]
                f.attrs["complete"] = 1
                f.attrs["variant"] = v
                for key, value in CONDITIONS[v].items():
                    f.attrs[key] = value
                groups, seeds = read_noise_schedule(c, v, 2, split, len(ids))
                f.create_dataset("read_noise_group", data=groups)
                f.create_dataset("read_noise_seed", data=seeds)
    write_json(tmp_path / "objects/manifest.json", {"objects": objects})
    return c


def test_old_transport_is_byte_exact():
    folder = ROOT / "experiments/depth/matlab/+legacy_mc"
    provenance = json.loads((folder / "PROVENANCE.json").read_text())
    for name, expected in provenance["files"].items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == expected


def test_pairing_and_training_only_scaling(fixture_run):
    c = fixture_run
    audit_pairs(c)
    before = training_scales(c, 2)
    with h5py.File(dataset_path(c, "corrected", 2, "test"), "r+") as f:
        f["input"][:] = 100000
    assert training_scales(c, 2) == before  # Test brightness cannot set normalization.
    with h5py.File(dataset_path(c, "corrected", 2, "test"), "r+") as f:
        f["gt"][0, 0, 0, 0] = .777
    with pytest.raises(ValueError, match="Ground truth differs"):
        audit_pairs(c)


def test_reject_source_leakage(fixture_run):
    p = Path(fixture_run["run_dir"]) / "objects/manifest.json"
    manifest = json.loads(p.read_text())
    manifest["objects"][2]["z_first"] = 1
    write_json(p, manifest)
    with pytest.raises(ValueError, match="Source z-block leakage"):
        audit_pairs(fixture_run)


def test_generalized_kl_has_finite_correct_minimum():
    y = torch.tensor([0., .2, .8])
    p = y.clone().requires_grad_()
    loss = loss_value(p, y, "generalized_kl")
    loss.backward()
    assert abs(loss.item()) < 1e-7
    assert torch.isfinite(p.grad).all()
    assert loss_value(y+0.1, y, "generalized_kl") > loss


def test_unclipped_metrics():
    y = np.ones((16,16), np.float32)
    assert metrics(2*y, y)["mse"] == 1  # A bad overbright result is not clipped away.
    assert metrics(y, y)["ssim"] == 1


def test_full_model_train_resume_and_comparison(fixture_run):
    c = fixture_run
    audit_pairs(c)
    for v in VARIANTS:
        train_one(c, v, 2, 100, torch.device("cpu"))
    paths = [Path(c["run_dir"]) / "models/2sls" / v / "seed100/last.pt" for v in VARIANTS]
    saved = [torch.load(p, weights_only=False) for p in paths]
    assert len({a["initial_state_sha256"] for a in saved}) == 1
    assert len({state_hash(a["model"]) for a in saved}) == 1  # Identical fixtures => identical paired training.
    mtime = paths[0].stat().st_mtime_ns
    (paths[0].parent / "complete.json").unlink()
    with pytest.raises(RuntimeError, match="Training incomplete"):
        evaluate_all(c, "cpu")
    train_one(c, "legacy", 2, 100, torch.device("cpu"))
    assert paths[0].stat().st_mtime_ns == mtime  # Completed run is not retrained.
    # Validate real-data inference without inventing reference metrics.
    real = Path(c["run_dir"]) / "experimental/2sls.h5"
    real.parent.mkdir()
    with h5py.File(real, "w") as f:
        f.create_dataset("input", data=np.ones((1,32,32,32), np.float32))
        f.create_dataset("sample_id", data=[1000001])
        f.attrs["experiment_id"] = c["experiment_id"]
        f.attrs["complete"] = 1
    evaluate_all(c, "cpu")
    make_report(c)
    report = Path(c["run_dir"]) / "report/REPORT.md"
    assert report.is_file()
    import csv
    with open(Path(c["run_dir"]) / "evaluation/per_sample.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 36  # Sixteen synthetic combinations * two objects + four real predictions.
    assert {r["train_condition"] for r in rows} == set(VARIANTS)
    with open(Path(c["run_dir"]) / "report/paired_differences.csv") as f:
        assert all(float(r["difference"]) == 0 for r in csv.DictReader(f))
    assert all(r["mse"] == "" for r in rows if r["test_domain"] == "experimental")


def test_epoch_resume_matches_uninterrupted(fixture_run, tmp_path):
    c = fixture_run
    train_one(c, "legacy", 2, 100, torch.device("cpu"))
    resumed = copy.deepcopy(c)
    resumed["training"]["epochs"] = 2  # Internal test only; CLI config edits create a new run identity.
    train_one(resumed, "legacy", 2, 100, torch.device("cpu"))
    train_one(resumed, "corrected", 2, 100, torch.device("cpu"))
    paths = [Path(c["run_dir"]) / "models/2sls" / v / "seed100/last.pt" for v in ("legacy", "corrected")]
    a, b = [torch.load(p, weights_only=False) for p in paths]
    assert state_hash(a["model"]) == state_hash(b["model"])
    assert a["history"] == b["history"]


def test_legacy_camera_and_table_are_byte_exact():
    folder = ROOT / "experiments/depth/matlab/+legacy_camera"
    provenance = json.loads((folder / "PROVENANCE.json").read_text())
    for name, expected in provenance["files"].items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == expected
    assert hashlib.sha256((ROOT / provenance["lut_path"]).read_bytes()).hexdigest() == provenance["lut_sha256"]


def test_old_reader_has_no_boundary_leakage(tmp_path):
    from Modules.support_train import read_h5py
    p = tmp_path / "boundary.h5"
    with h5py.File(p, "w") as f:
        f["gt"] = np.arange(7)
        f["input"] = np.arange(7)
        f["aaa_metadata"] = [123]  # Keys must be named, not chosen alphabetically.
    val, _ = read_h5py(p, True)
    train, _ = read_h5py(p, False)
    assert set(val).isdisjoint(train)
    assert sorted([*val, *train]) == list(range(7))


def test_camera_group_schedule_and_metadata_are_enforced(fixture_run):
    c = fixture_run
    group, seed = read_noise_schedule(c, "legacy", 2, "train", 34)
    assert group.tolist() == [1]*16 + [2]*16 + [3]*2
    assert np.array_equal(seed, read_noise_schedule(c, "psf_only", 2, "train", 34)[1])
    assert len(set(read_noise_schedule(c, "corrected", 2, "train", 34)[1])) == 34
    assert not np.intersect1d(seed, read_noise_schedule(c, "legacy", 2, "test", 34)[1]).size
    with h5py.File(dataset_path(c, "psf_only", 2, "train"), "r+") as f:
        f.attrs["camera"] = "corrected"
    with pytest.raises(ValueError, match="factor metadata"):
        audit_pairs(c)


def test_factorial_contrasts_resolve_interaction():
    # Known means: PSF=2, camera=3, extra interaction=5; combined=10.
    value = {"legacy": 0, "psf_only": 2, "camera_only": 3, "corrected": 10}
    means = {(2, arm, domain, s): np.full(3, v+s) for arm, v in value.items()
             for domain in VARIANTS for s in (1, 2, 3)}
    rows = paired_contrasts({"depths_sls": [2], "seeds": [1, 2, 3]}, means)
    expected = {"combined": 10, "psf_with_old_camera": 2, "psf_with_corrected_camera": 7,
                "camera_with_old_psf": 3, "camera_with_corrected_psf": 8, "interaction": 5}
    assert len(rows) == 4*6*3
    assert all(r["difference"] == expected[r["contrast"]] for r in rows)


def test_old_two_arm_config_cannot_silently_run(tmp_path):
    c = json.loads((ROOT / "experiments/depth/configs/pilot.json").read_text())
    del c["conditions"]
    p = tmp_path / "old.json"
    write_json(p, c)
    with pytest.raises(ValueError, match="four explicit"):
        configuration(p)


def test_matlab_byte_string_attributes_are_accepted(fixture_run):
    c = fixture_run
    for v in VARIANTS:
        for split in SPLITS:
            with h5py.File(dataset_path(c, v, 2, split), "r+") as f:
                for key in ("experiment_id", "variant", "psf", "camera"):
                    f.attrs[key] = np.bytes_(f.attrs[key])
    audit_pairs(c)
