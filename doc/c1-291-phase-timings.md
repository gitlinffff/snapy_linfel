# chengcli/snapy#291 C1: per-phase GPU timings, snapy 2.10.9 vs 2.11.2

> **Hardware note.** These numbers come from an NVIDIA GeForce **GTX 1080 Ti** (Pascal, sm_61) with
> **torch 2.7.1+cu126**. torch 2.10 ships no sm_6x kernels and does not run on this card. The
> card has low FP64 throughput, and the FP64 implicit solve takes a large share of the step,
> so **these ratios are not the A100 numbers in the issue** (A100, torch 2.10.0+cu128, 1.54x).
> Use them for where the extra time goes, not for its size on an A100.

## Setup

- **Versions:** v2.10.9 = `914b43bc80a7ba5567890784623fce4d54b307d8`, v2.11.2 = `117e449a620bd7fd50abe19239ab660a56d5d4cb`.
- **Instrumentation:** one commit per version on its tag, branches `c1-291-timing-v2.10.9`
  (`df78e670a60d01a8a283f1dc97cedd94c698f86d`) and `c1-291-timing-v2.11.2`
  (`2b2d6d1f84e6fb23fceec145219a5d135c6b8926`). Each adds `src/utils/phase_scope.hpp`
  (`at::RecordFunction`, USER_SCOPE) and sequential `snap::*` scopes in `HydroImpl::forward`
  and `MeshBlockImpl::forward/advance_local`, with the same names in both versions where the
  code exists. Profiling only, not for merge.
- **Build:** CUDA source builds of both versions (CMake Release, `CUDA=ON`, sm_61), each in its own tree.
- **Software:** torch 2.7.1+cu126, CUDA toolkit 12.6, kintera 2.6.0 (`38eac220`), pyharp 2.7.2,
  pydisort 1.8.15 (all CUDA source builds), the same stack for both versions.
- **GPU:** NVIDIA GeForce GTX 1080 Ti (sm_61, 11 GiB), driver 580.178.04.

## How it was measured

- **Driver:** a Python driver following `run_hydro.cpp`'s order per step: `max_time_step`, three
  `block.forward` RK stages, refresh `hydro_w`, kintera kinetics (`forward`, `jacobian`,
  `evolve_implicit`), then `check_redo`. It adds `drv.*` scopes around each.
- **Step time:** wall time per step, CUDA-synchronised, profiler off. Two independent runs per version.
  - Small 2-D deck: 50 warm-up steps, then 200 timed steps per run.
  - 160^3 block: 3 warm-up steps, then 5 timed steps per run.
- **Per phase:** `torch.profiler` (CPU + CUDA) over 20 steps (small deck) or 3 steps (block)
  after the warm-up. Each device op (kernel, memcpy, memset) is charged to the innermost scope
  that encloses its launch, matched by correlation id. Per-phase values are GPU ms per step.
  GPU time outside every scope is 0.12 ms (small deck) and 14 ms (block) per step.
- **Deck:** `examples/earth_crm.yaml` from v2.11.2, the same file for both versions:
  `ideal-moist` with `limiter: true`; H2O + H2O(l) + H2O(l,p); weno5, lmars, rk3; constant
  gravity (2.11.2 defaults `gravity-work: cell`, fixer on); sedimentation, nucleation,
  coagulation and evaporation kinetics. Changes: `implicit-scheme: 1`, `cfl: 0.9` (as in the
  issue), no internal solid boundary. Initial state: moist adiabat (Ts 300 K, Ps 1e5 Pa,
  xH2O 0.02) plus seeded 0.01 m/s noise. No redos occurred in any run.

## Step totals

| deck | 2.10.9 ms/step | 2.11.2 ms/step | ratio | runs (2.10.9 / 2.11.2) |
|---|---|---|---|---|
| 2-D 120 (x1, vertical) x 178 (x2) | 17.27 | 23.57 | 1.36 | 17.25, 17.29 / 23.55, 23.59 |
| 160 x 160 x 160 block | 2008.50 | 2500.30 | 1.24 | 2006.16, 2010.85 / 2500.10, 2500.50 |

## Small 2-D deck: 120 (x1, vertical) x 178 (x2)

**Grid** 120 (x1) x 178 (x2) x 1 (x3), 2 runs per version, 200 timed and 20 profiled steps each after 50 warm-up steps; peak memory 0.04 / 0.04 GiB; redos 0 / 0; mean dt 0.2903 / 0.2903 s.

| per step | 2.10.9 | 2.11.2 | ratio |
|---|---|---|---|
| wall time per step, ms (CUDA-synchronised, no profiler) (runs: 17.25, 17.29 / 23.55, 23.59) | 17.27 | 23.57 | 1.36 |
| GPU busy per step, ms (profiled) | 15.63 | 20.51 | 1.31 |
| CUDA kernels per step | 1712.00 | 2598.00 | 1.52 |
| GPU time not inside any scope, ms | 0.12 | 0.12 | 1.00 |

| phase | 2.10.9 GPU ms | 2.11.2 GPU ms | ratio | share 2.10.9 | share 2.11.2 | kernels 2.10.9 / 2.11.2 | share of the slowdown |
|---|---|---|---|---|---|---|---|
| EOS: conserved -> primitive (hydro.eos) | 0.818 | 1.364 | 1.67 | 5.2 % | 6.6 % | 177 / 264 | +11.2 % |
| x1 well-balanced reference | 0.560 | 0.525 | 0.94 | 3.6 % | 2.6 % | 66 / 33 | -0.7 % |
| x1 reconstruction | 0.772 | 0.793 | 1.03 | 4.9 % | 3.9 % | 96 / 99 | +0.4 % |
| x1 Riemann solver | 0.581 | 0.616 | 1.06 | 3.7 % | 3.0 % | 171 / 177 | +0.7 % |
| x1 sedimentation flux | 0.110 | 0.238 | 2.17 | 0.7 % | 1.2 % | 6 / 9 | +2.6 % |
| x2/x3 reconstruction | 0.754 | 0.764 | 1.01 | 4.8 % | 3.7 % | 66 / 66 | +0.2 % |
| x2/x3 Riemann solver | 0.566 | 0.569 | 1.00 | 3.6 % | 2.8 % | 171 / 171 | +0.1 % |
| positivity: theta | 0.287 | 0.297 | 1.03 | 1.8 % | 1.4 % | 78 / 81 | +0.2 % |
| positivity: census (hits/severe/min) | 0.076 | 0.371 | 4.91 | 0.5 % | 1.8 % | 12 / 66 | +6.1 % |
| positivity: species enthalpy (2.11) | 0.000 | 0.188 | new | 0.0 % | 0.9 % | 0 / 57 | +3.8 % |
| positivity: theta ghost fill | 0.062 | 0.063 | 1.01 | 0.4 % | 0.3 % | 24 / 24 | +0.0 % |
| positivity: carry (2.11) | 0.000 | 0.758 | new | 0.0 % | 3.7 % | 0 / 168 | +15.5 % |
| positivity: scale | 0.065 | 0.071 | 1.09 | 0.4 % | 0.3 % | 18 / 18 | +0.1 % |
| positivity: limiter meters (2.11) | 0.000 | 0.211 | new | 0.0 % | 1.0 % | 0 / 18 | +4.3 % |
| flux divergence | 0.339 | 0.326 | 0.96 | 2.2 % | 1.6 % | 48 / 48 | -0.3 % |
| forcing (const gravity etc.) | 0.199 | 0.200 | 1.00 | 1.3 % | 1.0 % | 57 / 57 | +0.0 % |
| gravity work (face/cell correction) | 0.184 | 0.351 | 1.91 | 1.2 % | 1.7 % | 75 / 147 | +3.4 % |
| gravity-work fixer: stage E+PE (2.11) | 0.000 | 0.237 | new | 0.0 % | 1.2 % | 0 / 60 | +4.9 % |
| implicit (vertical) solve | 7.571 | 8.513 | 1.12 | 48.4 % | 41.5 % | 186 / 291 | +19.3 % |
| ghost exchange before stage | 0.032 | 0.083 | 2.55 | 0.2 % | 0.4 % | 12 / 48 | +1.0 % |
| RK stage averaging + stage-0 save | 0.135 | 0.141 | 1.04 | 0.9 % | 0.7 % | 15 / 21 | +0.1 % |
| conserved limiter (whole column) | 0.476 | 0.910 | 1.91 | 3.0 % | 4.4 % | 90 / 162 | +8.9 % |
| saturation adjustment | 0.826 | 0.975 | 1.18 | 5.3 % | 4.8 % | 115 / 140 | +3.1 % |
| gravity-work fixer: apply (2.11) | 0.000 | 0.073 | new | 0.0 % | 0.4 % | 0 / 21 | +1.5 % |
| boundary functions | 0.048 | 0.044 | 0.93 | 0.3 % | 0.2 % | 18 / 18 | -0.1 % |
| driver: kinetics (kintera) | 0.948 | 1.123 | 1.19 | 6.1 % | 5.5 % | 168 / 197 | +3.6 % |
| driver: check_redo | 0.029 | 0.500 | 17.02 | 0.2 % | 2.4 % | 4 / 93 | +9.6 % |
| driver: max_time_step | 0.074 | 0.093 | 1.27 | 0.5 % | 0.5 % | 18 / 23 | +0.4 % |
| other scopes (drv.stage0, drv.stage1, drv.stage2, snap::hydro.forward, snap::mb.scalar, snap::mb.solid, snap::mb.user_forcing, snap::x1.seam) and unscoped | 0.115 | 0.115 | | | | | +0.0 % |
| **GPU busy, total** | **15.627** | **20.513** | **1.31** | 100 % | 100 % | 1712 / 2598 | 100 % |

Without the implicit solve, GPU time per step goes 8.06 -> 12.00 ms (ratio 1.49).

## Production-size block: 160 x 160 x 160 (4.1 M cells)

**Grid** 160 (x1) x 160 (x2) x 160 (x3), 2 runs per version, 5 timed and 3 profiled steps each after 3 warm-up steps; peak memory 6.34 / 6.93 GiB; redos 0 / 0; mean dt 0.323 / 0.323 s.

| per step | 2.10.9 | 2.11.2 | ratio |
|---|---|---|---|
| wall time per step, ms (CUDA-synchronised, no profiler) (runs: 2006.16, 2010.85 / 2500.10, 2500.50) | 2008.50 | 2500.30 | 1.24 |
| GPU busy per step, ms (profiled) | 2016.42 | 2492.70 | 1.24 |
| CUDA kernels per step | 2098.00 | 3077.00 | 1.47 |
| GPU time not inside any scope, ms | 14.03 | 14.05 | 1.00 |

| phase | 2.10.9 GPU ms | 2.11.2 GPU ms | ratio | share 2.10.9 | share 2.11.2 | kernels 2.10.9 / 2.11.2 | share of the slowdown |
|---|---|---|---|---|---|---|---|
| EOS: conserved -> primitive (hydro.eos) | 93.810 | 134.891 | 1.44 | 4.7 % | 5.4 % | 177 / 264 | +8.6 % |
| x1 well-balanced reference | 64.077 | 75.599 | 1.18 | 3.2 % | 3.0 % | 66 / 33 | +2.4 % |
| x1 reconstruction | 129.447 | 130.569 | 1.01 | 6.4 % | 5.2 % | 96 / 99 | +0.2 % |
| x1 Riemann solver | 67.817 | 70.659 | 1.04 | 3.4 % | 2.8 % | 171 / 177 | +0.6 % |
| x1 sedimentation flux | 15.494 | 30.739 | 1.98 | 0.8 % | 1.2 % | 6 / 9 | +3.2 % |
| x2/x3 reconstruction | 236.859 | 237.518 | 1.00 | 11.7 % | 9.5 % | 132 / 132 | +0.1 % |
| x2/x3 Riemann solver | 137.262 | 137.306 | 1.00 | 6.8 % | 5.5 % | 342 / 342 | +0.0 % |
| positivity: theta | 57.378 | 59.240 | 1.03 | 2.8 % | 2.4 % | 99 / 102 | +0.4 % |
| positivity: census (hits/severe/min) | 3.124 | 20.995 | 6.72 | 0.2 % | 0.8 % | 12 / 66 | +3.8 % |
| positivity: species enthalpy (2.11) | 0.000 | 21.751 | new | 0.0 % | 0.9 % | 0 / 57 | +4.6 % |
| positivity: theta ghost fill | 1.156 | 1.159 | 1.00 | 0.1 % | 0.0 % | 72 / 72 | +0.0 % |
| positivity: carry (2.11) | 0.000 | 140.689 | new | 0.0 % | 5.6 % | 0 / 225 | +29.5 % |
| positivity: scale | 17.775 | 17.777 | 1.00 | 0.9 % | 0.7 % | 27 / 27 | +0.0 % |
| positivity: limiter meters (2.11) | 0.000 | 6.550 | new | 0.0 % | 0.3 % | 0 / 18 | +1.4 % |
| flux divergence | 83.117 | 83.114 | 1.00 | 4.1 % | 3.3 % | 66 / 66 | -0.0 % |
| forcing (const gravity etc.) | 25.407 | 25.422 | 1.00 | 1.3 % | 1.0 % | 57 / 57 | +0.0 % |
| gravity work (face/cell correction) | 17.228 | 26.420 | 1.53 | 0.9 % | 1.1 % | 75 / 147 | +1.9 % |
| gravity-work fixer: stage E+PE (2.11) | 0.000 | 12.585 | new | 0.0 % | 0.5 % | 0 / 60 | +2.6 % |
| implicit (vertical) solve | 717.316 | 794.676 | 1.11 | 35.6 % | 31.9 % | 186 / 291 | +16.2 % |
| ghost exchange before stage | 1.173 | 1.384 | 1.18 | 0.1 % | 0.1 % | 60 / 132 | +0.0 % |
| RK stage averaging + stage-0 save | 31.217 | 31.234 | 1.00 | 1.5 % | 1.3 % | 15 / 21 | +0.0 % |
| conserved limiter (whole column) | 48.742 | 87.099 | 1.79 | 2.4 % | 3.5 % | 90 / 162 | +8.1 % |
| saturation adjustment | 114.794 | 127.728 | 1.11 | 5.7 % | 5.1 % | 115 / 140 | +2.7 % |
| gravity-work fixer: apply (2.11) | 0.000 | 2.989 | new | 0.0 % | 0.1 % | 0 / 21 | +0.6 % |
| boundary functions | 1.970 | 1.967 | 1.00 | 0.1 % | 0.1 % | 18 / 18 | -0.0 % |
| driver: kinetics (kintera) | 130.600 | 144.453 | 1.11 | 6.5 % | 5.8 % | 168 / 197 | +2.9 % |
| driver: check_redo | 0.604 | 47.096 | 77.94 | 0.0 % | 1.9 % | 4 / 93 | +9.8 % |
| driver: max_time_step | 6.021 | 7.039 | 1.17 | 0.3 % | 0.3 % | 23 / 28 | +0.2 % |
| other scopes (drv.stage0, drv.stage1, drv.stage2, snap::hydro.forward, snap::mb.scalar, snap::mb.solid, snap::mb.user_forcing, snap::x1.seam) and unscoped | 14.026 | 14.051 | | | | | +0.0 % |
| **GPU busy, total** | **2016.415** | **2492.696** | **1.24** | 100 % | 100 % | 2098 / 3077 | 100 % |

Without the implicit solve, GPU time per step goes 1299 -> 1698 ms (ratio 1.31).

## Reading

- **Total.** The step-time ratio is 1.36 on the small deck and 1.24 on the 160³ block, against 1.54 in the issue. Kernels per step rise 1.52x and 1.47x; the issue counted 1.46x.
- **Why the total ratio is lower here.** On this Pascal card the FP64 implicit solve dominates the step: 48 % / 32 % of 2.10.9's step on the small deck / the block. It slows only 1.12x. Without it, the small deck's ratio is 1.49, close to the issue's.
- **Where the slowdown comes from** (share of the extra GPU time, small deck / block):
  - the new positivity work: carry 15.5 / 29.5 %, census 6.1 / 3.8 %, species enthalpy 3.8 / 4.6 %, limiter meters 4.3 / 1.4 %; together about 30 / 39 %;
  - the implicit solve: 19.3 / 16.2 %, with 186 -> 291 kernels;
  - EOS cons->prim: 11.2 / 8.6 %, with 177 -> 264 kernels;
  - check_redo: 9.6 / 9.8 %; its GPU time goes 0.03 -> 0.50 ms, from 4 -> 93 kernels. The 2.11.2 floor test runs a full cons->prim on a clone;
  - the whole-column conserved limiter: 8.9 / 8.1 %, with 90 -> 162 kernels;
  - gravity work plus the fixer: 9.8 / 5.1 %;
  - sedimentation (positivity carry split) and saturation adjustment: a few % each.
- **Unchanged.** x1/x2/x3 reconstruction, the Riemann solvers, divergence and forcing are within a few %.

## Not checked

- **Not the issue's hardware.** No A100 and no torch 2.10.0+cu128: that wheel has no sm_6x code and fails on this GPU. Pascal has low FP64 throughput, so absolute times and shares differ from the issue.
- **Same kintera for both.** kintera 2.6.0 was used for both versions (the issue paired 2.10.9 with kintera 2.5.0), so the kintera change itself is not measured.
- **Not the issue's driver.** There is no radiation and no restart. The issue's own deck and driver are not available; the grid orientation (178 horizontal, 120 vertical) is my reading of "178 x 120".
- **Instrumentation overhead.** Builds without the scopes were not timed, so this overhead is not measured. When the profiler is off, a RecordFunction scope only checks whether profiling is active.
- `.item()` and host-sync counts per step, the per-switch timings of the issue (fixer off, face work, wall clamp), and multi-block or multi-rank runs were not measured.
