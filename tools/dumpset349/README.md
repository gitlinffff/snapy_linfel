# #349 bitwise dump set (reconstruction)

chengcli/snapy #349 describes a bitwise dump gate: 20 one-block cases, a deterministic initial
state, 20 rk3 steps at a fixed dt, saving the final `hydro_u` (and `scalar_s`). Its harness was
never committed, so this is a **reconstruction from the PR's case table**, not the original
script. The case names follow that table. Where the table leaves something open, these files
choose it: which solver goes with which box2d recon variant, the initial states, and dt.

It is the shared copy for the V3 wave 1 #349 gates (C1-C8 and later). It is tooling only,
outside any wave PR.

## Files

| file | what |
|---|---|
| `dumpset349.py` | the 20 cases, the runner and the dump writer (one child process per case) |
| `compare.py` | byte-compares two dump directories; exits 0 only if every case ran and every file is byte-identical, else lists the cases that differ |
| `extra_plumeroe.py` | OPTIONAL extra case `extra_column256_dc_plumeroe`, **outside #349**; see below |
| `SHA256SUMS.main-730b3500.cuda-sm61-fp64.txt` | checksums of main 730b3500's 23 dumps, CUDA float64, GTX 1080 Ti (sm_61) |
| `SHA256SUMS.main-730b3500.cpu-fp64.txt` | checksums of main 730b3500's 23 dumps, CPU float64, AMD Ryzen 9 7950X |

## Cases (all float64, 20 rk3 steps, fixed dt, nghost 3)

| case | grid | recon (vertical = horizontal) | Riemann | EOS / physics | dt | dumped |
|---|---|---|---|---|---|---|
| box2d_weno5s_lmars | Cartesian 64x128x1, gravity | weno5 scale | lmars | ideal-gas | 0.124275 | hydro_u |
| box2d_weno5_hllc | same | weno5 | hllc | ideal-gas | 0.124275 | hydro_u |
| box2d_weno3s_roe | same | weno3 scale | roe | ideal-gas | 0.124275 | hydro_u |
| box2d_weno3_lmars | same | weno3 | lmars | ideal-gas | 0.124275 | hydro_u |
| box2d_cp5_hllc | same | cp5 | hllc | ideal-gas | 0.124275 | hydro_u |
| box2d_cp3_roe | same | cp3 | roe | ideal-gas | 0.124275 | hydro_u |
| box2d_plm_lmars | same | plm | lmars | ideal-gas | 0.124275 | hydro_u |
| box2d_weno5ss_hllc | same | weno5 scale shock | hllc | ideal-gas | 0.124275 | hydro_u |
| box3d_weno5s_hllc | Cartesian 32x24x24, gravity | weno5 scale | hllc | ideal-gas | 0.248549 | hydro_u |
| gnomonic_weno5s_{lmars,hllc,roe} | gnomonic-equiangle face block 32x16x16, gravity | weno5 scale | as named | ideal-gas | 0.248549 | hydro_u |
| column2048_weno5s_lmars | Cartesian x1 column 2048x1x1, no gravity (the tiled CUDA path) | weno5 scale | lmars | ideal-gas | 0.00795358 | hydro_u |
| column2048_weno3s_hllc | same | weno3 scale | hllc | ideal-gas | 0.00795358 | hydro_u |
| column2048_cp5_lmars | same | cp5 | lmars | ideal-gas | 0.00795358 | hydro_u |
| moist_weno5ss_lim_sed_scalar | Cartesian 48x64x1, gravity | weno5 scale shock | lmars | ideal-moist dry/vapor/cloud, EOS limiter, const-vsed cloud sedimentation, passive scalar | 0.1657 | hydro_u, scalar_s |
| moist_weno5s_lim_sed_scalar | same | weno5 scale | lmars | same | 0.1657 | hydro_u, scalar_s |
| moist_weno5s_lim_sed_scalarhole | same | weno5 scale | lmars | same, scalar with a zero hole (tracer limiter) | 0.1657 | hydro_u, scalar_s |
| moist_weno5s_nolim_sed | same | weno5 scale | lmars | ideal-moist, no limiter, sedimentation | 0.1657 | hydro_u |
| shallow_weno5_sroe | Cartesian 128x64x1, periodic | weno5 | shallow-roe (dir xy) | shallow-water | 0.00857872 | hydro_u |

dt comes from the deck alone, not from the build: 0.3 min(dx) / (c_s + 30 m/s), with c_s for an
ideal gas at 300 K (shallow water: 0.3 dx / (sqrt(12) + 2)). Each run writes
`<case>.hydro_u.bin` / `<case>.scalar_s.bin`: the raw C-order bytes of the final arrays, ghost
cells included. That gives 23 files and 8,060,720 bytes per run. It also writes `<case>.json`
(shape, dtype, dt and the positivity counters) and `status.json`.

Notes:
- The column cases run without gravity. With const-gravity on main, the x1 column gave
  bitwise-equal cp5 and weno5s results, so the recon variant was not being tested. My untested
  guess is the x1 well-balanced path.
- The two moist `scalar` cases have the same `hydro_u`, because the scalar is passive. Their
  `scalar_s` differ. All other dumps are pairwise distinct.
- On main, the limiters fire: about 1210 hydro positivity hits in the limited moist cases,
  1315 without the limiter, and 5106 scalar hits in the hole case.

## How to run a gate

Build main and the step **on the same machine, with identical build flags**, and run both on the
same device. Compare CPU with CPU and CUDA with CUDA only, never across devices: CPU vs CUDA on
main differs in all 23 files at round-off (max |diff| about 3e-10).

```sh
# one run per build; <pkg> = the directory holding that build's installed snapy (and kintera, torch)
PYTHONPATH=<main-pkg>:<deps> python tools/dumpset349/dumpset349.py --device cuda --dtype float64 --out runs/main_cuda
PYTHONPATH=<step-pkg>:<deps> python tools/dumpset349/dumpset349.py --device cuda --dtype float64 --out runs/step_cuda
python tools/dumpset349/compare.py runs/main_cuda runs/step_cuda   # exit 0 = all 20 cases byte-identical
```

For CPU, use `--device cpu` (and `CUDA_VISIBLE_DEVICES=` if you want to hide the GPU).
`--case <name>` runs a single case. Run main twice and compare it with itself first, to show the
device is deterministic. For each failing case, compare prints the first differing element, its
field, and the max abs/rel difference.

Run the script from outside the source tree (e.g. `cd /tmp`), so `import snapy` picks up the
installed build and not the checkout's `python/` folder.

The checksum files hold this machine's main bits (`cd runs/main_cuda && sha256sum -c
<path>/SHA256SUMS...txt`). They match only on hardware and builds that give main identical bits.
The gate itself is main vs step on one machine, not against these files.

Reference on main 730b3500 (Release, default FP flags, CUDA=ON for sm_61, UCX=OFF, PNETCDF=OFF;
torch 2.10.0+cu126, nvcc 12.6.85, gcc 11.5.0, kintera v2.6.9 b8e726ce built with CUDA):
- 3 CUDA runs gave 23/23 byte-identical dumps, and so did 3 CPU runs.
- One full 20-case run takes about 42 s on CUDA (GTX 1080 Ti) and about 39 s on CPU (Ryzen 9
  7950X), mostly process start-up.

## Optional extra case, outside #349

`extra_plumeroe.py` runs `extra_column256_dc_plumeroe`: a 256-cell ideal-gas column, donor cell,
no gravity, using the plume-roe solver. It was added because V3 C1 edits `plume_roe.cpp`. It is
**not part of the 20-case gate**: `dumpset349.py`, `compare.py` and the checksums do not include
it. On main 730b3500 it does not run. Hydro always asks the solver for face pressure, and
PlumeRoeSolver throws "Face-pressure output is not implemented for PlumeRoeSolver", so the script
exits 2 and writes no dump. Give it its own output directory:

```sh
python tools/dumpset349/extra_plumeroe.py --device cuda --out runs/main_cuda_extra
```
