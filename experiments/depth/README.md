# Old versus corrected PSFs: DEEP2 depth study

This workflow trains the repository's **original scSE-UNet** separately on old
and corrected scattering PSFs, with everything else paired, then evaluates the
models on shared test inputs. It is a controlled new experiment, not a claim to
have reproduced the paper's original checkpoints or resolved its mismatch.

## What depths are being tested?

| Depth (SLS) | Depth at mus=200 cm^-1 | Role |
|---:|---:|---|
| 2 | 100 um | Experimental success in the supplied Fig. 6 |
| 4 | 200 um | Deepest experimental success reported in the paper |
| 6 | 300 um | Reported experimental failure; simulation already succeeded |
| 8 | 400 um | Beyond the paper's simulated validation range; source adapter has experimental FOVs |
| 10 | 500 um | Further extension; simulation only with the supplied data |

The paper's Discussion (pp. 8-9), Fig. 6, Table 1, and dataset section establish
the distinction between experimental and simulated depth limits. One SLS here
is **1/mus = 50 um**, not the transport mean free path 1/[mus(1-g)] = 500 um.

Sources: [paper](https://doi.org/10.1038/s41377-023-01248-6),
[original dataset](https://zenodo.org/records/8161051),
[original mouse adapter](../../fwd_model/f_get_extPettern.m).

## Copy and run: Linux workstation

Requires Python 3.10+, PyTorch 2.2+, MATLAB with Image Processing, Statistics and
Machine Learning, and Parallel Computing toolboxes, and a MATLAB-compatible
NVIDIA GPU. A recent MATLAB release supported by the workstation's GPU is
recommended. The Python runner does not install or license MATLAB.

```bash
git clone --branch experiments/psf-depth-comparison https://github.com/Le0nZim/DEEP-squared.git
cd DEEP-squared
bash experiments/depth/setup.sh
source .venv-depth/bin/activate
python depth_study.py plan --config experiments/depth/configs/paper_depths.json
python depth_study.py run --config experiments/depth/configs/paper_depths.json --device cuda:0
```

For an existing checkout:

```bash
git fetch origin
git switch experiments/psf-depth-comparison
```

`setup.sh` reuses an existing PyTorch installation if importable; otherwise it
installs PyTorch. **Check `doctor` for CUDA availability.** If an existing install
is CPU-only, install a suitable CUDA PyTorch wheel into `.venv-depth` using the
official [PyTorch installer](https://pytorch.org/get-started/locally/). Do not
change GPU drivers automatically. On Windows, create/activate the venv with the
usual `python -m venv --system-site-packages .venv-depth` and
`.venv-depth\Scripts\Activate.ps1`, then install the same requirements and run
the same Python commands. `--matlab` accepts an absolute path to MATLAB.

To use physical GPU 1 on Linux:

```bash
CUDA_VISIBLE_DEVICES=1 python depth_study.py run --config experiments/depth/configs/paper_depths.json --device cuda:0
```

The Python device is zero-based among visible GPUs. MATLAB's `optics.gpu_index`
is one-based among the GPUs it can see. Check visibility on that workstation;
do not assume the Python and MATLAB physical GPU numbering is identical.
The new generator uses one selected GPU and never resets every GPU or starts
one worker per GPU.

## Run sequence and resume

The `run` command performs these stages sequentially:

1. Download the two required Zenodo MAT files (about 1.4 GB), or verify existing
   copies with the published MD5 checksums. Default location: `data/deep2_zenodo`.
2. Run native scattering regression checks and an optical CPU/GPU agreement
   check; record MATLAB's release and a MATLAB/Python HDF5 axis sentinel.
3. Generate shared exPSF/emPSF, source objects and split manifest, both scattering
   PSF banks, camera tables, and paired measurements. Export experimental stacks.
4. Audit exact paired targets, IDs, shapes, finite values, and source split bounds.
5. Train **30 models**: 2 PSF versions x 5 depths x seeds 100/101/102. Each uses
   Adam, lr=1e-3, betas=(0.9,0.999), batch=10, 100 epochs; StepLR(20,0.3), as in
   the upstream training script. No early stopping, silent epoch reduction, or
   mixed precision. Select the best checkpoint using its validation loss only.
6. Evaluate old/corrected training against old/corrected held-out simulations,
   plus identical experimental stacks when available; generate tables and panels.

For a short installation/pipeline run use `configs/pilot.json` (5 epochs, one
seed, 256/64/64 objects, 400,000 photons per plane). Its results are diagnostic.
It deliberately retains the 32-pattern architecture and source data; it is
still real MATLAB generation and can take appreciable time.

Rerun the **same command** after an interruption. PSF generation resumes at the
last saved batch (every eight batches); completed planes are reused. Data
generation resumes at the last written object. Training resumes at the last
completed epoch, including optimizer, scheduler, data-order and RNG states.
Completed models are skipped. Evaluation/report generation can be repeated.
Atomic `.partial` files are not considered completed artifacts.

Each stage is also callable independently:

```bash
python depth_study.py doctor
python depth_study.py download
python depth_study.py prepare
python depth_study.py audit
python depth_study.py train --device cuda:0
python depth_study.py evaluate --device cuda:0
python depth_study.py report
```

Pass the **same `--config` to every stage** if not using the default. Preparing
on one machine and training on another is supported: copy the run directory and
data, keep the config/source identical, and use the corresponding absolute or
repo-relative `output_dir`. The run ID excludes resolved machine-specific paths.
Keep enough disk space for the actual source resolution: the default 1,280
objects x 10 conditions at 326x326 requires about **167 GiB uncompressed** for
input/target datasets, plus PSFs and checkpoints. Compression reduces actual
usage. `plan` prints this estimate; `doctor` prints available storage. The source
volume and PSFs also need substantial host RAM (allow roughly 32 GB or more).
The original-scale Monte Carlo setting is **128 million photons per plane**, up
to 1,000 hops; do not mistake it for a quick run. No runtime estimate is asserted.

## Which parts are old, corrected, or held common?

| Component | Protocol |
|---|---|
| Old scattering | Byte-exact copies of `f_launch`, `f_hop`, `f_spin`, `f_backProp` from pre-correction commit `757aefd`; hashes and source provenance in `matlab/+legacy_mc/PROVENANCE.json` |
| Corrected scattering | Calls the existing corrected MATLAB functions in `fwd_model/_submodules/MC_LightScattering` |
| Tissue forwarding | Old arm retains the original unforwarded g/nt/nm defaults; corrected arm forwards them. Both coincide at the default g=0.9, n=1.33 |
| Monte Carlo budget | Same photon/hop budgets and seed formula for both arms; streams independent between training and test PSF banks |
| Scattering mass | Histograms normalized by launched photons; overflow edge bins discarded as upstream; no unit-sum normalization |
| Optical PSFs | Common upstream vectorial Debye equations and grids; excitation intensity squared for two-photon excitation; row blocks reduce memory |
| Forward operator | Original 3D/2D convolution helpers, same operator order, sampling and focal-plane convention; one prescribed object per call |
| Patterns | Same measured 32 patterns, MATLAB indices 21:52, same ordering as the source's mouse adapter; no generated replacement patterns |
| Source objects | `Data.cell` from BV MAT; source XY spacing from metadata, axial step 1.5 um from paper; resampled to the measurement grid and configured dz |
| Data splits | Disjoint contiguous source-z blocks (60/20/20), two-plane boundary guards, then seeded spatial crops; same manifest in all conditions |
| Ground truth | Unscattered excitation-PSF image of the same object; common per-object peak normalization to [0,1], identical between arms/depths |
| Input normalization | Single maximum over the two training arms at each depth; stored in checkpoints and used unchanged for validation/test/experimental inputs |
| Architecture | Original `Modules.model.UNet`, 13,424,353 parameters. Only deprecated addition/interpolation syntax was modernized, preserving operations and state-dict keys |
| Initialization and order | Same model seed and DataLoader seed for paired models; exact resume on the tested CPU setup; CUDA bilinear backward can be nondeterministic |

These are explicit **common repairs to the experiment harness**, not additional
PSF differences between the arms:

- The historical `run.py` ignores supplied CLI arguments (`parse_args(args=[])`),
  hard-codes lab paths, overlaps a split boundary for some dataset sizes, and
  feeds raw nonnegative predictions to `KLDivLoss`, which expects logarithmic
  input. The new runner uses a proper CLI, separate datasets, sample-weighted
  loss averages, lazy HDF5 loading, explicit devices, and full checkpoints.
- Default `generalized_kl` is the intensity I-divergence
  `mean(y*log(y/(prediction+eps)) - y + prediction + eps)`, with a stable zero-target
  implementation. It is nonnegative and suitable for unnormalized intensities.
  It is **not a literal reproduction** of the source's erroneous KL invocation
  or the paper's written expression without the linear terms. Both arms use it.
  `mse` is available; `legacy_kl` explicitly reproduces the historical raw-input
  call for diagnostics and can be unbounded. Changing loss creates a new run.
- The source EMCCD routine caps photon counts at its lookup-table size and adds
  a scalar read-noise draw to the entire image. The common new camera code
  generates conditional tables from the same Bernoulli multiplication process
  for the actual gain, extends them when necessary, and draws read noise per
  pixel. Its finite table has 10,000 trials per input count. It does not silently
  switch to Gaussian shot noise or clip high photon counts.
- Explicit source resampling, guarded splits, and fixed sample counts replace
  the original ad hoc crop/selection loop. This tests within-volume held-out
  regions, **not generalization to new animals**. Source crops can be correlated.

Compare the two arms **within this protocol** to isolate the PSF change. Comparing
these results directly with published scores also includes these common changes,
different source sampling, and possibly a different software environment.

## Brightness and going deeper

`paper_depths.json` uses `paper_peak`: each simulated stack is scaled to the
depth-specific experimental peak calibration, as in the original generator.
That asks whether **PSF shape** improves reconstruction under matched brightness.
It removes collection-efficiency differences from the noiseless peak by design.
At 10 SLS, there is no calibration in the supplied mouse adapter: the config
explicitly holds the 8-SLS brightness and camera settings. This is an optimistic
brightness-controlled simulation, not a measured 500-um signal level.

Run the complementary source-brightness control:

```bash
python depth_study.py run --config experiments/depth/configs/fixed_source.json --device cuda:0
```

`fixed_source` calibrates one gain using the median peak of the first 16
**training** objects under old PSFs at 2 SLS. It applies that same multiplier to
all objects, depths and both PSF variants, allowing collection loss to affect
photon counts and SNR. Both arms at a given depth use identical camera settings.
It models fixed emitted source strength under this forward operator; it does
not add attenuation/scattering of the excitation or biological absorption.

For production, the center scattering plane is also simulated with twice the
hop limit using the same initial streams. Per-condition `convergence.json` files
report changes in in-FOV mass and coarse normalized shape. Inspect them and the
late-escape fractions. Independent train/test PSF banks expose photon sampling
variation, but **a full photon-count/support convergence study is still needed**
before claiming a physical depth limit. Change the config for extra budgets;
never silently replace a failed convergence result with a favorable one.

## Outputs and interpretation

`plan` prints `results/depth_study/<name>-<fingerprint>/`. Changes to config or
relevant source code produce a new fingerprint; old/new PSFs cannot be mixed
by reusing a filename. `resolved.json` and `provenance.json` record source hashes.
The raw Zenodo assets are checksum verified before generation.

- `objects/manifest.json`: sample IDs, source bounds, split and voxel spacings.
- `psfs/`: shared optics, old/new train/test kernels, absolute collection and
  convergence diagnostics. `calibration/`: depth-specific brightness/camera settings.
- `datasets/<depth>/<variant>/{train,val,test}.h5`: streaming float32 measurements
  and common targets; per-object brightness multipliers and noiseless peaks.
- `experimental/<depth>.h5`: the same real FOVs for both models, with their names.
  Availability is checked during source loading; the source advertises 2/4/6/8
  SLS (and 7 if explicitly configured). No 10-SLS file is invented.
- `models/<depth>/<variant>/seed<seed>/{best,last}.pt`, `history.csv`, `metadata.json`.
- `evaluation/per_sample.csv`: MSE/PSNR/SSIM for simulations, plus coverage and
  saved predictions. No ground-truth metrics are assigned to unreferenced real data.
- `report/REPORT.md`, `summary.csv`, `paired_differences.csv`, `depth_curves.png`,
  and fixed-example old/new image panels.

The four synthetic comparisons answer different questions:

| Train | Shared test | Interpretation |
|---|---|---|
| Old | Old | Baseline under old simulator |
| Corrected | Old | Reverse mismatch control |
| Old | Corrected | Old-trained model under the corrected simulator |
| Corrected | Corrected | Corrected-trained model under the corrected simulator |

The most relevant **simulated mismatch** contrast is old-trained versus
corrected-trained on the same corrected test data. Real-data panels compare
both models on identical experimental inputs, selected before seeing results.
Source widefield/average images and conventional DEEP reconstructions are not
registered ground truth. Qualitative sharper output can contain hallucinated
structures; matched physical reference measurements would be needed to verify
those structures and quantify an experimental depth improvement.

Report means over test objects within each training seed, then across seeds.
Paired corrected-minus-old differences use identical objects/inputs. The 95%
t intervals cover variation in training-seed means, not biological uncertainty.
No per-prediction rescaling/clipping is applied to numerical metrics; [0,1] is
the fixed target range. Panels use common reconstruction limits and the first
configured examples, not cherry-picked best cases.

## Validation status

The Python integration tests exercise the **full original architecture**, one
training epoch, continuation from an epoch checkpoint, paired reproducibility,
all four evaluation combinations, unreferenced experimental inference, leakage
rejection, normalization isolation, zero-target loss gradients, and report
generation. They use explicit small synthetic fixtures to test the software.

```bash
python -m pytest -q experiments/depth/tests
```

The development environment has no MATLAB or CUDA. **Native MATLAB generation,
the production photon budget, and scientific training results have not been
executed here.** `prepare` runs the supplied native checks on the workstation
before production; it stops on missing dependencies or failed checks. The
NumPy walkthrough notebook is not used as a production PSF generator.
