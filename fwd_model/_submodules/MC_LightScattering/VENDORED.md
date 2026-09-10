# Dependency provenance

The MATLAB runtime files in this directory come from
[dushanw/MC_LightScattering](https://github.com/dushanw/MC_LightScattering),
commit `7b0ed3d2a5c2c48178e7aa15591208b93294be26`, the exact revision pinned
by DEEP-squared commit `068c1aa17ba294faad523f4942f2052276ef36b4`.
The upstream MIT license and copyright notice are retained in `LICENSE`.

This directory is now tracked directly in DEEP-squared so its scattering
corrections can be committed and cloned with a single repository. It is no
longer a Git submodule. The original directory name is retained for existing
MATLAB paths. Runtime `.m` files and the upstream README are included. The
upstream reference PDF and empty editor marker are omitted; neither is used
at runtime. The optical PSF reference remains available in the upstream repo.

See `../../SCATTERING_AUDIT.md` for the corrections and validation limits.
