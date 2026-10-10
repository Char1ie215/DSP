# Third-Party Notices

DSP vendors a limited diffusion-policy and runtime subset from OAT under
`src/dsp_diffusion`. Preserve `OAT_LICENSE.txt` (Chaoqi Liu and contributors,
2026). File-level origins and source hashes are recorded in
`docs/diffusion_provenance.json`. The subset includes DSP's previously validated
integration patches and uses a new Python namespace. This is not an assertion
that the third-party implementation was authored by DSP's authors.

- OAT revision: `4bb0c01611c5b1ec8a6268cb376aca2cb519c881`.
  https://github.com/Chaoqi-LIU/oat
- OAT's top-level license is primarily MIT but identifies separately licensed
  non-commercial components. Its referenced NOTICE file was absent at this
  revision. The extracted subset excludes tokenizers, autoregressive policies,
  and the Apple/EPFL-derived `model/common/misc.py`; the trainer uses PyTorch's
  BF16 capability API instead. No retained source header identified a separate
  non-commercial license. This file-selection audit is not legal clearance and
  does not relicense any upstream code or erase attribution obligations.
- LIBERO's observed code license is MIT, copyright Lifelong Robot Learning (2023).
  Its external source and asset licenses continue to apply.
- The external dataset card identifies MIT licensing:
  https://huggingface.co/datasets/chaoqi-liu/libero10_N500.zarr

LIBERO is installed independently at revision
`6090ff21837566fed47b7c9061c9899f4749d36b`, not bundled with this source archive.

The project does not relicense external code, datasets, assets, or dependencies.
Authors have not yet selected a project-wide license. This candidate is prepared
for technical validation, not legal clearance for unrestricted redistribution.
