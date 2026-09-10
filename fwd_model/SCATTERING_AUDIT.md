# Monte Carlo scattering audit and corrections

## Scope and source revisions

Audited DEEP-squared at
[`068c1aa17ba294faad523f4942f2052276ef36b4`](https://github.com/wadduwagelab/DEEP-squared/tree/068c1aa17ba294faad523f4942f2052276ef36b4).
Its scattering implementation is the separate MC_LightScattering dependency,
pinned to
[`7b0ed3d2a5c2c48178e7aa15591208b93294be26`](https://github.com/dushanw/MC_LightScattering/tree/7b0ed3d2a5c2c48178e7aa15591208b93294be26).
This patch vendors that revision's MATLAB runtime, preserving its MIT license,
so the corrections belong to the DEEP-squared repository and survive a fresh
clone without a second fork. See the dependency's `VENDORED.md`.

The original simulator launches photons at a point below a planar surface,
draws exponential scattering distances with mean `1e4/mus` micrometers, and
updates directions using the Henyey–Greenstein (HG) phase function. Photons
that reach the surface are held there. The code then refracts and filters
these rays through the objective aperture, projects them backward to the
source plane, and bins their positions into a scattering PSF. The histogram
is divided by **all launched photons**, preserving collection losses.
`f_simPSFs3D.m` repeats this at multiple source depths.

## Confirmed defects and fixes

| Location | Defect | Correction | Relevant with the supplied default parameters? |
| --- | --- | --- | --- |
| `f_launch.m` | Uniform polar angle oversamples directions near the optical axis. | Sample `cos(theta)` uniformly on `[-1,1]` and azimuth uniformly on `[0,2*pi)`. | Yes. |
| `f_backProp.m`, objective filter | Uses `uz/abs(u)`, a cosine, as the sine in the NA condition. | Test the transverse magnitude of the normalized refracted direction against `NA/nm`. | Yes. |
| `f_backProp.m`, refraction | Applies Snell's law to the longitudinal component and reconstructs an incorrect direction. | Scale the normalized transverse components by `nt/nm`; obtain positive `uz` from the unit-vector constraint. Reject nontransmitting and downward directions. | Oblique refraction is wrong for unequal indices; the original also has an exactly axial singularity at matched indices. |
| `f_spin.m` | Divides by `g` at `g=0` and by zero for an exactly axial incident direction. | Handle isotropic and perfectly forward/backward scattering explicitly; use an axial basis, stable small-`g` algebra, and normalized real unit directions. | The exact-axis singularity is an edge case; `g=0` is outside the default. |
| `f_simPSFs3D.m` | Does not forward `g`, `nt`, or `nm` to the simulator. | Copy all three supplied parameters. | Default values happen to agree, but edits were silently ignored. |
| `f_hop.m` | Surface roundoff can produce a tiny negative hop and change an already escaped ray's position/path length. | Require upward travel, clamp the boundary distance to nonnegative, and set escaped `z` exactly to zero. | Numerical edge case. |

The collection routine also returns a correctly sized zero histogram when no
rays are collected and uses `(z0_um-z)/uz` for projection to the target plane.
The existing histogram normalization and outer guard-bin convention are retained.

### The objective-angle error

For a unit direction and angle measured from the positive optical axis,

```text
cos(alpha) = uz
sin(alpha) = sqrt(ux^2 + uy^2)
NA = nm * sin(alpha_max)
```

For `NA=1`, `nm=1.33`, and matched tissue/medium indices:

| Rule | Accepted upward angles from the optical axis |
| --- | --- |
| Correct aperture | 0 to 48.7535 degrees |
| Original aperture condition | 41.2465 degrees to nearly 90 degrees |

The intervals overlap, but the original condition excludes the central cone
and includes overly oblique rays. This follows directly from the
[original collection code](https://github.com/dushanw/MC_LightScattering/blob/7b0ed3d2a5c2c48178e7aa15591208b93294be26/f_backProp.m)
and the [definition of numerical aperture](https://www.microscopyu.com/microscopy-basics/numerical-aperture).
Exactly axial and exactly grazing inputs additionally expose singular cases
in the original refraction algebra.

### The refraction error

For unit input direction, the corrected geometry is

```text
ux_out = (nt/nm)*ux
uy_out = (nt/nm)*uy
uz_out = sqrt(1 - ux_out^2 - uy_out^2)
```

Only upward rays with `ux_out^2+uy_out^2 < 1` propagate out toward the objective.
For `nt=1.5`, `nm=1`, a 30-degree incident ray must transmit at 48.5904 degrees;
a 50-degree ray must undergo total internal reflection. The original code
reverses those two transmission classifications. The
[MCML reference implementation](https://omlc.org/software/mc/mcml/mcml-src/mcmlgo.c)
uses the sine in Snell's law and scales transverse direction cosines.

### What was already correct

The ordinary HG inverse-CDF expression used at `g=0.9` agrees with
[OMLC's MCML implementation](https://omlc.org/software/mc/mcml/mcml-src/mcmlgo.c).
Its deflection angle is relative to the **incoming photon direction**.
The exponential free-path formula and conversion from centimeters to
micrometers are also correct for positive `mus`.

Uniform solid-angle launch follows the isotropic source in
[OMLC's mc321.c](https://omlc.org/software/mc/mc321.c).
Uniform polar angle would require a specifically intended nonisotropic
source; no such source model is specified in this implementation.

## Validation performed

Ran `tests/angular_reference_checks.py` with NumPy, seed `20260910`, and
1,000,000 samples. These are independent numerical checks of the equations;
**they do not execute the modified MATLAB functions**. The recorded output
is `tests/angular_reference_results.json`.

| Check | Numerical result | Analytical target |
| --- | --- | --- |
| Original launch `E[uz^2]` | 0.499824 | 1/3 for isotropic emission; fails |
| Corrected launch `E[uz^2]` | 0.333209 | 0.333333 |
| Corrected launch `E[ux^2]`, `E[uy^2]` | 0.333221, 0.333570 | 0.333333 each |
| Corrected ballistic collection fraction, NA=1 in index 1.33 | 0.170158 | 0.170350 |
| Ordinary HG `E[cos(theta)]`, g=0.9 | 0.900125 | 0.9 |
| Ordinary HG `E[P2(cos(theta))]`, g=0.9 | 0.810156 | 0.81 |
| Small-g rearrangement versus extended-precision original formula at g=+/-0.0001 | max difference 1.26e-15 | algebraic equivalence |

The ballistic fraction is `(1-sqrt(1-(NA/nm)^2))/2`, measured relative to all
photons from an isotropic source with no scattering and no field-of-view
loss. It is a geometry check, **not a tissue collection prediction**.

Added 14 CPU MATLAB regression tests in `tests/test_scattering.m`. They call
the production functions and cover launch isotropy, the accepted angular
cone, ballistic focusing/collection, Snell refraction in both index
directions, TIR rejection, scale-independent directions, empty collection,
HG moments, axial and nearly axial incidence, g=+/-1, frozen escaped rays,
hop statistics/units, surface stability, and a small transport run.

MATLAB and Octave were unavailable in the execution environment. These
14 tests have **not been run**, and neither the GPU path nor the full
PSF-generation/training pipeline has been validated here. Run on your MATLAB
installation, from the repository root:

```matlab
results = runtests('fwd_model/tests/test_scattering.m');
assertSuccess(results);
```

The CPU tests need Statistics and Machine Learning Toolbox for the existing
`hist3` call. They do not need a GPU, a parallel pool, or experimental data.
The independent numerical reference check can be rerun with:

```bash
python fwd_model/tests/angular_reference_checks.py --output angular_reference_results.json
```

## Model boundaries and consequences

The executed [visual walkthrough](notebooks/Scattering_Fixes_Visual_Walkthrough.ipynb)
adds six figures and a self-contained NumPy reproduction of the original and
corrected transport equations. Its seeded diagnostic uses 40,000 photons,
300 hops, a source depth of 350 micrometers, and the default tissue/NA values.
It compares launch statistics, aperture geometry, refraction, numerical edge
cases, four combinations of transport and collection, and absolute collection
versus conditional radial spread. Its outputs are embedded in the notebook.
This is a finite Python experiment, not an execution of the MATLAB code or a
replacement for a convergence study or production PSF regeneration.

- This remains a homogeneous, scattering-only half-space model with a finite
  number of hops. Absorption, heterogeneous tissue, and convergence with
  `NtimePts` were not added or validated.
- At unequal indices, the correction fixes the geometry of first-pass
  transmission. The original approximation still discards TIR rays instead
  of returning them to the tissue and omits Fresnel reflection/transmission
  weights. It is not a complete transport model of an index-mismatched
  interface. The supplied defaults have `nt=nm=1.33`.
- Optical PSF generation, detector noise, and the reconstruction network were
  outside this scattering patch. In particular, full PSF generation retains
  its existing GPU/toolbox requirements.
- Existing saved scattering PSFs and synthetic training measurements do not
  change automatically. Regenerate them with this patch before comparing
  reconstructions. No claim is made here about improved reconstruction
  accuracy or the magnitude of changes in the published results.
