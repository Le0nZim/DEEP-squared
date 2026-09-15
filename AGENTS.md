# Running the DEEP2 scattering and camera comparison

Read `experiments/depth/README.md` and `fwd_model/CAMERA_AND_PIPELINE_AUDIT.md`.
Follow the launch commands literally. The user
is testing whether the scattering corrections improve DEEP2 at and beyond
the paper's experimental depth limit.

- Work on `experiments/psf-depth-comparison`. Do not use `run.py`, `validation.py`,
  or `fwd_model/main.m` for this experiment; those are historical scripts.
- Use `depth_study.py`. Start with `doctor` and `plan`, then run the requested
  config. `run` downloads verified source data, generates old/new native MATLAB
  PSFs and all four camera/scattering datasets, trains every job sequentially, evaluates, and reports.
- MATLAB plus the documented toolboxes and a compatible NVIDIA GPU are required
  for generation. If MATLAB is missing, report that concrete dependency. Do not
  replace it with a Gaussian blur, the notebook's demonstration simulator,
  random data, pretrained weights, a different architecture, or a skipped stage.
- The network must remain `Modules.model.UNet`, the original 13,424,353-parameter
  scSE-UNet with 32 patterns. `+legacy_mc` contains byte-exact old transport code;
  do not repair it. The same applies to `+legacy_camera` and the archived camera
  MAT table: do not modify them or rewrite their expected hashes. Corrected transport is in the existing `fwd_model` subtree.
- Default depths are 2/4/6/8/10 SLS, or 100/200/300/400/500 um at mus=200 cm^-1.
  The figure shows 2/4 SLS. Experimental failure was reported at 6 SLS; the paper
  already succeeded in simulation at 6 SLS.
- Use all four supplied conditions: legacy (old scattering + old camera),
  psf_only (fixed scattering + old camera), camera_only (old scattering + fixed
  camera), corrected (both fixed). Production is 60 models per config; pilot is
  20. PSF banks are reused by conditions sharing scattering. Do not collapse
  these to two conditions or call the prior two-arm baseline fully old code.
- Keep seeds, objects, train/val/test splits, patterns, nominal camera parameters,
  common optical-coordinate repairs, loss, and training settings paired. Do not train on the test partition, compute
  normalization from test data, tune on experimental test FOVs, or choose a
  different checkpoint using their appearance.
- Legacy read noise is one Gaussian per configured generation group (default 16
  samples), across every pixel/pattern/object in that group. Corrected noise is
  independent per element. Keep group IDs/seeds on resume; generation groups
  are independent of optimizer minibatches. The documented group size is a
  controlled choice, not a recovered paper setting.
- Run the required native camera, scattering, optical, and forward checks before
  production. Do not bypass a failed check, accept incompatible camera tables,
  or clip photon counts in corrected conditions.
- The pilot config is only a pipeline check. Do not report its few-photon,
  five-epoch results as the completed scientific experiment. Do not silently
  reduce production photon counts, epochs, spatial resolution, depths, or seeds.
- Use the same config to resume. The runner records source/config fingerprints,
  completed samples, completed PSF batches, and epoch checkpoints. Editing the
  config/code creates a different run directory by design. Never mark unfinished
  files complete or reuse artifacts by manually changing their provenance.
- Report results on identical test inputs: all 16 train/test condition combinations,
  and the same experimental FOVs for all four models. Widefield images and DEEP
  reconstructions are not ground truth. Do not fabricate experimental PSNR/SSIM.
- Explain PSF, camera, combined and interaction contrasts on one shared test
  domain. The camera factor includes read-noise and table/clipping changes.
  The legacy conditions share repaired training/optics; they are not a literal
  run of every historical defect.
- Inspect PSF convergence and fixed-source results before claiming a depth gain.
  Simulation success alone cannot establish that experimental mismatch is fixed.
- Keep raw data, generated HDF5/MAT files, checkpoints, and results out of git.
  Summarize the completed job count, failures, report path, and remaining native
  validation requirements when handing work back.
