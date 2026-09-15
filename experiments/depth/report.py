"""Paired numerical summaries and fixed-example comparison panels."""
import csv
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

from .common import VARIANTS, dataset_path, depth_tag, read_json


def make_report(c):
    base = Path(c["run_dir"])
    out = base / "report"
    out.mkdir(exist_ok=True)
    with open(base / "evaluation/per_sample.csv") as f:
        rows = list(csv.DictReader(f))
    by_seed = defaultdict(list)
    for r in rows:
        if int(r["has_ground_truth"]):
            key = (float(r["depth_sls"]), r["train_psf"], r["test_domain"], int(r["seed"]))
            by_seed[key].append([float(r[m]) for m in ("mse", "psnr", "ssim")])
    means = {k: np.mean(v, axis=0) for k, v in by_seed.items()}
    aggregate = defaultdict(list)
    for (d, train, test, seed), value in means.items():
        aggregate[(d, train, test)].append(value)
    summary = []
    for (d, train, test), values in sorted(aggregate.items()):
        arr = np.asarray(values)
        row = {"depth_sls": d, "train_psf": train, "test_domain": test, "n_seeds": len(arr)}
        for j, metric in enumerate(("mse", "psnr", "ssim")):
            row[metric + "_mean"] = float(arr[:, j].mean())
            row[metric + "_seed_sd"] = float(arr[:, j].std(ddof=1)) if len(arr) > 1 else ""
        summary.append(row)
    if not summary:
        raise ValueError("No completed synthetic evaluations; run evaluate first")
    with open(out / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    deltas = []
    for d in c["depths_sls"]:
        for domain in VARIANTS:
            delta = np.array([means[(d, "corrected", domain, s)] - means[(d, "legacy", domain, s)] for s in c["seeds"]])
            for j, metric in enumerate(("mse", "psnr", "ssim")):
                mean = float(delta[:, j].mean())
                half = float(t.ppf(.975, len(delta)-1) * delta[:, j].std(ddof=1) / np.sqrt(len(delta))) if len(delta) > 1 else None
                deltas.append({"depth_sls": d, "test_domain": domain, "metric": metric,
                               "corrected_minus_legacy": mean,
                               "ci95_low": mean-half if half is not None else "",
                               "ci95_high": mean+half if half is not None else ""})
    with open(out / "paired_differences.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(deltas[0]))
        writer.writeheader()
        writer.writerows(deltas)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for row_no, domain in enumerate(VARIANTS):
        for col, metric in enumerate(("mse", "psnr", "ssim")):
            ax = axes[row_no, col]
            for variant, color in (("legacy", "#a65736"), ("corrected", "#167c80")):
                entries = [r for r in summary if r["train_psf"] == variant and r["test_domain"] == domain]
                xs = [r["depth_sls"] for r in entries]
                ys = [r[metric+"_mean"] for r in entries]
                sd = [r[metric+"_seed_sd"] or 0 for r in entries]
                ax.errorbar(xs, ys, yerr=sd, marker="o", color=color, label=variant)
            ax.axvline(4, color="gray", linestyle=":", alpha=.5)
            ax.set(xlabel="Depth (SLS)", ylabel=metric.upper(), title=f"Test simulator: {domain}")
            ax.grid(alpha=.2)
            ax.legend(title="Training PSFs")
    fig.suptitle("Shared test data; error bars show variation across training seeds")
    fig.savefig(out / "depth_curves.png", dpi=180)
    plt.close(fig)
    panels = []
    seed = c["seeds"][0]
    for d in c["depths_sls"]:
        for domain in (*VARIANTS, "experimental"):
            path = base / "experimental" / f"{depth_tag(d)}.h5" if domain == "experimental" else dataset_path(c, domain, d, "test")
            if not path.exists():
                continue
            preds = [np.load(base / "evaluation" / f"{depth_tag(d)}_{v}_seed{seed}_on_{domain}.npz")["prediction"] for v in VARIANTS]
            n = min(c["evaluation"]["save_examples"], len(preds[0]))
            if n == 0:
                continue
            with h5py.File(path, "r") as f:
                has_gt = "gt" in f
                fig, axes = plt.subplots(n, 4 if has_gt else 3, figsize=(12, 3*n), squeeze=False, constrained_layout=True)
                for i in range(n):
                    avg = np.mean(f["input"][i], axis=0)
                    images = [avg, preds[0][i], preds[1][i]]
                    titles = ["Mean of 32 measurements", "Trained with old PSFs", "Trained with corrected PSFs"]
                    if has_gt:
                        images.append(f["gt"][i, 0])
                        titles.append("Ground truth")
                    for j, (im, title) in enumerate(zip(images, titles)):
                        axes[i, j].imshow(im, cmap="magma", vmin=0, vmax=max(float(avg.max()), 1e-8) if j == 0 else 1)
                        axes[i, j].set_title(title)
                        axes[i, j].axis("off")
            fig.suptitle(f"{d:g} SLS ({d*50:g} um); test: {domain}; seed {seed}; first {n} samples")
            name = f"{depth_tag(d)}_{domain}.png"
            fig.savefig(out / name, dpi=140)
            plt.close(fig)
            panels.append(name)
    coverage = read_json(base / "evaluation/coverage.json")
    lines = ["# DEEP2 PSF depth comparison", "", f"Experiment: `{c['experiment_id']}`", "",
             f"Signal protocol: **{c['signal_mode']}**. Loss: **{c['training']['loss']}**.", "",
             "The paper succeeded experimentally at 2/4 SLS and failed at 6 SLS; its simulations already worked at 6 SLS. "
             "8/10 SLS extend beyond its simulated validation range.", "",
             "These results do not automatically establish that the experimental model mismatch is fixed. "
             "Corrected simulated data are a surrogate, not independent physical ground truth. "
             "The supplied experimental FOVs have no registered reference ground truth; their panels are qualitative. "
             "Do not treat widefield images or another reconstruction as ground truth.", "",
             f"Missing experimental depths (SLS): {coverage['missing_experimental_depths_sls']}", "",
             "![Depth curves](depth_curves.png)", "",
             "| SLS | Train PSF | Test simulator | MSE | PSNR | SSIM |",
             "|---:|---|---|---:|---:|---:|"]
    for r in summary:
        lines.append(f"| {r['depth_sls']:g} | {r['train_psf']} | {r['test_domain']} | {r['mse_mean']:.5g} | {r['psnr_mean']:.3f} | {r['ssim_mean']:.4f} |")
    lines += ["", "`paired_differences.csv` reports corrected-minus-old differences on identical test inputs. "
              "95% t intervals are over training-seed means (unavailable with one seed). They do not measure biological "
              "uncertainty; source-volume crops are correlated. Predictions are not clipped or rescaled for metrics. "
              "Panels use the first configured samples, and a shared [0,1] range for reconstructions/targets.", "",
              "Inspect PSF convergence/collection diagnostics in `../psfs/` before scientific interpretation. "
              "This remains homogeneous scattering with finite transport time; excitation scattering, tissue "
              "heterogeneity, and absorption are not introduced by the angular fixes.", ""]
    for p in panels:
        lines += [f"![{p}]({p})", ""]
    (out / "REPORT.md").write_text("\n".join(lines))
    print(f"REPORT {out / 'REPORT.md'}", flush=True)
