# Camera and pipeline re-audit

This audit updates the depth-study branch after the scalar read-noise bug was
reported. The earlier two-condition runner already used independent pixel read
noise in **both** arms, so its `legacy` label meant old scattering, not the old
camera. That was inadequate for an old-simulator versus corrected-simulator
comparison. The new protocol explicitly crosses scattering and camera versions.

## Confirmed camera defects and their corrections

### One Gaussian for an entire batch

The original `f_simulateIm_emCCD.m` used:

```matlab
Xhat = (Xhat + normrnd(0,pram.cam_sigma_rd))/pram.cam_EMgain;
```

Both arguments to `normrnd` are scalars, so this draws **one number**. MATLAB
broadcasts it over every Y/X pixel, excitation pattern, and object in the input
array. It represents a common batch offset, not independent sensor read noise.
See the primary [MATLAB normrnd documentation](https://www.mathworks.com/help/stats/normrnd.html)
for scalar versus array-sized draws.

The corrected production routine uses:

```matlab
read_noise = pram.cam_sigma_rd * randn(size(electrons));
Xhat = (electrons + read_noise) / pram.cam_EMgain;
```

The shape includes **all** axes; drawing only a Y-by-X array and broadcasting it
across patterns/batch would still be wrong. Negative input-equivalent measured
values from read noise remain possible and are not clipped away. The ADU
conversion retains the original gain, ADC factor and bias convention.

Let `s = cam_sigma_rd / cam_EMgain` in input-equivalent electrons. For two
distinct elements in the same generated batch:

| Statistic, considering read noise alone | Original | Corrected |
|---|---:|---:|
| Marginal mean per element | 0 | 0 |
| Marginal variance per element | s² | s² |
| Covariance between distinct elements | s² | 0 |
| Read-noise variance of the mean of 32 patterns | s² | s²/32 |
| Read-noise standard deviation of that mean | s | s/sqrt(32) |

Thus the bug is the correlation structure, not simply the noise amplitude.
An apparent smooth offset is statistically different from spatial noise a
network must remove. These statements concern the read component, not total
shot-plus-EM noise or a measured reconstruction improvement.

### Silent photon-count clipping

The original camera replaces every Poisson count above the lookup-table height
with that height. The supplied table is 100-by-10,000, so a 150-electron event
is sampled as a 100-electron event. This biases both the mean and distribution;
the lookup boundary is not a modeled camera full-well limit.

The corrected production camera samples the same Bernoulli multiplication
register directly for counts outside a supplied table. The study runner instead
extends its deterministic conditional table to cover the observed counts.
Neither path caps the input count or substitutes Gaussian shot noise.

### Unverifiable table reuse and trial-count truncation

The historical entry script loads the same table after selecting different
camera settings. The distributed MAT contains only `emhist`, with no gain
metadata. Its mean response at 100 input electrons is about 50,165 output
electrons, consistent with an effective gain near 500; this does **not** verify
compatibility with each experimental setting.

Corrected tables carry `cam_N_gainStages` and `cam_Brnuli_alpha`; incompatible
or untagged tables are rejected. `f_genEmhist` now returns that structure,
respects the caller's RNG, and produces exactly the requested number of trials.
Previously, its 100-trial chunk loop dropped a remainder and did not initialize
a table for fewer than 100 trials. The study cache key includes the full
precision multiplication probability, stage count, and trial count; extending
or reloading a cache preserves earlier rows and the caller's RNG state.

Migration for a direct MATLAB caller:

```matlab
rng(100,'twister');
emhist = f_genEmhist(100,10000,pram); % metadata-bearing struct
[Y,Yadu] = f_simulateIm_emCCD(expected_photons,emhist,pram);
% Alternatively pass [] to simulate the Bernoulli register directly.
```

`main.m` now generates compatible tables instead of loading the untagged table.
The study's corrected camera uses the same repaired production camera function.
CPU random sampling gives explicit MATLAB `rng` control even if array storage
is on the GPU. This changes historical random sequences, which are not claimed
to be reproduced.

## Other concrete findings

| Location | Finding | Resolution |
|---|---|---|
| `Modules/support_train.py` | Validation ends at ceil(N/5), training starts at floor(N/5): one shared sample when N is not divisible by five. Alphabetic HDF5 key selection can also select metadata. | One shared split boundary; explicit `gt`/`input` keys; regression with seven samples and an extra metadata key. |
| `fwd_model/main.m` | Test uses MATLAB 1:128 and training starts at 128. | Training starts at 129. The study uses disjoint source-z blocks before cropping. |
| `f_fwd3D.m` | Scan omits the final valid slab, including the only slab of an exact-fit volume. A strict mean comparison also discards constant positive volumes. | Inclusive window endpoint and nonblank, non-strict mean selection. |
| `f_fwd3D.m` | Y crop uses Nx; quarter-turn augmentation changes rectangular dimensions; empty output is a misleading scalar zero. | Use Ny, shape-preserving rotations, correctly shaped empty arrays, and explicit blank-normalization failure. |
| `Efficient_PSF.m`, study optical helper | Radius coordinates and `calculate_phi` differ by half a sample; odd grids lack the intended common zero. The original row loop also assumes Nx=Ny. | Radius and azimuth share integer-centered x/y coordinates; row loop uses Ny. **Common repair in every condition.** |
| `depth_forward.m` | Earlier runner selected Nz/2+1 for even support while `f_fwd3D` uses ceil(Nz/2). | Restore the declared upstream convention, with a native CPU/GPU forward-target agreement check on an even-Z object. This establishes agreement, not independent physical registration. |
| `depth_study.py` workflow | CUDA availability previously checked only after expensive generation; MATLAB HDF5 strings may be bytes. | Check requested CUDA device before `run`; accept fixed byte-string and variable string attributes; verify condition labels and read-noise grouping. |

The supported runner already repaired the historical raw-output `KLDivLoss`
misuse, sample weighting, hard-coded paths, checkpoint/RNG resume, and
prediction-by-prediction metric rescaling. `run.py`, `validation.py`, and
`Modules/quantitative_metrics.py` remain historical interfaces and are **not**
the experiment entry points. Use `depth_study.py`; its metrics retain amplitude
errors and never invent reference metrics for experimental FOVs.

## What exactly is preserved in the old control?

`experiments/depth/matlab/+legacy_mc` holds byte-exact old transport routines;
`+legacy_camera` now holds the byte-exact old camera routine. Their provenance
manifests and the archived camera table's SHA-256 are checked before every CLI
command. Do not fix those preserved copies or rewrite their expected hashes.

The legacy camera wrapper calls the old shot/EM routine with read noise disabled,
then adds its single Gaussian with a separately seeded stream. This preserves
the old noise **law**, including table reuse and clipping, while allowing
streaming writes without losing the original across-sample correlation.
`camera.legacy_batch_samples=16` defines a generation batch; the same scalar is
used for those 16 objects and every pixel/pattern within them. Groups and seeds
are stored in HDF5 and stay the same after resume. The final group may be shorter.
This setting is not the optimizer minibatch size.

Historical generation used a varying number of accepted slabs per `f_fwd3D`
call; there is no universal published batch size or original RNG trace to
recover. Sixteen is an explicit controlled-experiment choice, not a claim about
the paper's actual batch sizes. Change it in a copied config for a sensitivity
study; never silently switch the legacy baseline to independent scalars per
object. Splits and depths have separate group streams.

The four conditions are `legacy`, `psf_only`, `camera_only`, and `corrected`.
All share the corrected optical grid, preprocessing, split/normalization rules,
architecture, and repaired training procedure. Consequently this is a factorial
comparison of **scattering transport and camera implementations under one common
protocol**, not a literal rerun of every historical defect. Camera effects
include independent read noise, table provenance/gain matching, and removal of
clipping; a camera-effect score alone cannot attribute an improvement solely to
read noise. Native zero-input tests isolate that specific correction.

## Validation and scientific limits

The Python suite exercises the full original 13,424,353-parameter scSE-UNet,
all 16 synthetic train/test combinations, four predictions per experimental
FOV, epoch resume, immutable legacy sources, split rejection, MATLAB byte-string
attributes, camera factor/group validation, and known factorial contrasts.
Fixtures in those tests are software checks, not generated tissue PSFs.

Native tests in `tests/test_camera_and_forward.m` call actual production code:
per-element covariance and 32-pattern averaging, original scalar broadcasting,
counts above the LUT limit, invalid tables/rates, analytical Bernoulli-register
moments, arbitrary table widths, cache/resume behavior, blank/exact-fit/final
slabs, rectangular crops, and optical reflection symmetry. `depth_native_checks`
also checks scattering, optical CPU/GPU agreement and even-Z forward agreement
before generation. Native checks must pass on the launch workstation.

This development environment has no MATLAB/CUDA. MATLAB files were syntax
parsed; the native checks and scientific training have **not** run here. Passing
the Python suite does not substitute for those gates.

Remaining physics questions include absolute optical calibration and wavelength
convention (`2*pi/lambda` is retained from upstream; adding an index factor needs
an explicit vacuum/in-medium interpretation), axial registration and PSF
resampling/throughput, finite photon/support/hop convergence, heterogeneous
tissue, excitation scattering and absorption. They are not claimed fixed by
this audit. The corrected simulator remains a hypothesis to test against shared
experimental measurements. In particular, success on simulated 6/8/10-SLS data
cannot establish that the paper's experimental mismatch has been removed.
