## What V3 changes

Base: chengcli/snapy `main` 72bc3980d569e8e1cc2b2dd1f1d3c671ae02acff, which already holds waves 2-5
(#372, #373). Head: `v3/final` 73c45a7c8c991209924e1137992352c335575062, 194 commits on that base.
The result-changing rows below are measured against `main` deeabfee597cd83f68f4586d245b01f6091eaa6e
(v2.14.3), the named reference.

Lines of the closing batch (the fixes below, 4c023985..73c45a7c, `git diff --numstat 4c023985
73c45a7c`): `src` net −10 (+870/−880) and `python/csrc` net −3 (+6/−9), −13 together; `tests`
net +786 (+952/−166).

- **The flux register books every source.** `AxialMomentum` reads the flux register; `exact`
  replaces `full` and the closure key (3a6a2c46). `Hydro` holds `X1Momentum`, which builds the
  axial momentum (11b35d8a); `FluxRegister::post_split` books the step's direct part of the split
  source (5b68d667). Diffusion posts its energy flux and its radial halves, and `exact` allows a
  viscosity with the viscous torque booked (ee0ce297, c33a67f9; test e09ea710). `riemann.central`
  sits on the HLLC perturbation-flux path (2642c0a5). The outward step transfer (deaaa724) was
  dropped again (6f3dbfc9) because nothing read it. The stage's census masks, wall rates and status
  are batched once per stage and the stage amounts fold once per step on the device (3bf18620,
  ce297125, 871b90c0); the wall batch keeps fixed device storage and its slots follow the stage's
  push order (a78dc50f, fa734fff, 9766b2d7); the register re-prescribes after a source's post and
  rollback goes through the monitor (8a224e1b, 614c2772); only writers that can touch the
  prescribed faces mark a direction (89a38287, d7c7a93a); `place_amounts` checks all three amount
  tensors (939f7928) and casts on the host before a move, so MPS, which holds no float64, works
  (9003c961).
- **Output.** Writer core without `AthenaArray`; netcdf and pnetcdf sinks; vectorize and
  super-resolution removed (f28908d3); restart writer, bundle and reader in one module, the reader
  throws (40d54fb3); `combine_parts` replaces `mppnccombine` (d7cee8bb, 74cbc7b8); the slice bounds
  and order checks join the output table's generated checks (baf68074); the budget consumer,
  `BudgetSeries` and `BudgetCarry` (e8cbe791); `src/interface` deleted (73c8ee4f);
  `print_cycle_diagnostics` deleted (61c4b2e6).
- **Kintera retry and chemistry.** Kintera's failure counter is in the step's reset list and retry
  stack (7ce3e938); the `ThermoY` active set is snapshotted at step start and restored on a retry
  (aa7b41bf); the saturation adjustment runs inside the captured segment (08836980); the chemistry
  step is a registered source after the last stage (2555f25d); eos, scalar and sedimentation use
  fixed buffers and no stage host reads (ea53dfea); the ideal-moist EOS sums species explicitly
  (b316baf4, result-changing, see below).
- **Reconstruction, Riemann and FP32.** FP32 typed constants; uniform weno5 rows read `d`
  (e341ffdf, result-changing for FP32, see below); the block-independent FP32 sweep gauge with the
  x2/x3 walls in the exchanged ghost slabs (4d41d797); the reconstruction code under its line
  ceiling (d3178f20); one shock default (`false`) and density/pressure floors on the shock path
  (3bd63808); one reader for a reconstruct section (55cb9d0b); the stretched x1 reference rows live
  on the grid's device (069769ba, issue #316).
- **Layout and seams.** `Exchange::seam_slab`, diffusion in one round (b4355db1); the same-process
  exchange state keyed by the Mesh (295741a0); dead legacy ranges deleted (56569511); the
  panel-seam rotation operands are built once per grid and the cache holds its source grid storages
  (1a3c45e7, 70f6e15e).
- **Wall prescription and dt.** The wall prescription works on the wall face planes only
  (8b587ce7) and `FluxRegister::reset()` drops it too (c3f2ff8c); the positivity limiter keeps the
  x1 wall faces prescribed (0ec6b1d3); dt is a CUDA-graph input, not part of the graph key, with a
  per-block device dt buffer (b84b8bdb).
- **One home for each helper.** The first audit (33 commits, ff239db6..faa65213): one home for
  each duplicated helper across layout, coord, eos, riemann, recon, scalar and output; the aneos
  adiabatic index and sound speed in one place (be018f65); the deck readers take their defaults
  from the option tables. The second audit (0456087d, f8c0c66d, fec6b5e3, 14c7b221, b5212677):
  unreached code, unread fields, commented-out code, one-caller wrappers, a duplicate ctest entry.
  Option defaults read from their option structs (b9632696, 80ac12e9, c07afc5d, 1477afb0,
  d1eae7bf); the two unread output options (f4fa5f63) and the six unused `Quantity` rows (833aca42)
  removed. Comments name the issue, not people or iterations (d258cbf1).
- **Build and CI.** kintera from PyPI instead of a source build (b541a6db), then 2.6.12 on every
  leg (1497416a, 7d48cfa4), then 2.6.13 on every leg with the seam energy reference re-recorded
  (f7034194; see "kintera pin"). The parallel tests keep `snap_cu` linked (750a5b09). `SNAP_SQR` is
  a macro again so contracting compilers fuse as before ff66651a (7a70d38f) and
  `ANEOSThermoImpl::create` keeps the pointer it made (de04b2a5): the two macOS fixes. The graph
  tests skip unless the build has CUDA (b7d47b34).
- **CUDA graphs.** A Mesh owns one stage executor and runs its blocks' stages through it
  (1515539d); the stage is a schedule of segments keyed (block, stage, segment), with the
  stage-entry exchange as its own host-exchange segment and the tendencies, user forcings and
  combine as ops (7e487d25, d7b608f3, 629d1b58, 3d81ba4d, 52b7cdf7); a block's graphs share one
  memory pool and a block with no live graph captures into a fresh pool (1ffccfd0, 9bd69762); an
  outflow boundary and a user stage forcing are per-op refusals (f398794a, a647120f); a cut at an
  exchange lets a Mesh's blocks capture together, the cubed sphere's panel seams are cuts, the
  same-process seam copies are a graph, and a lone block's exchanges are no cut (30cf8389,
  28350ed5, 8c211e7c, 16256d62; the lone-block branch is replaced by the executor fix below); the vertical implicit solve reads the device dt and runs under
  cuda-graph (0c939da6, 743d2931); a registered chemistry step is an eager segment and a Mesh with
  chemistry captures its blocks' stages (051212e9, 6529741a); a Mesh whose blocks disagree runs
  eager (4b21ca22); the ideal-gas EOS, lmars and no forcing capture, and capture allows ideal-gas,
  ideal-moist, lmars and hllc only (cb80feaa, 43fae31b); the register monitor's amounts and the
  block's `EfStatus` slots, `conversion_trips` included, are in the capture save list (e6fb9967,
  7ac0d8b7, 66614c32, 4c023985); a stretched cubed sphere captures (069769ba);
  `integration.execution` defaults to cuda-graph on a CUDA device (02e4ef05). Gate on every row:
  graph equals eager bitwise in FP64 and FP32 with dt changing every step.

## Fixes landing with this PR

These are the 56 commits on 4c023985 (168de4ef is the clang-format commit), each checked on its
fork branch by a non-author before it was placed on `v3/final`. One entry per change. Each entry has a RED test where behaviour or
a reachable path changed, and says so where it has none; each passed the dump set byte-identical in
FP64 and FP32 on CPU and CUDA and the graph-equals-eager tests. The commits in parentheses are
the landed ones.

- **Executor: every capture on its own stream; split panels and many-block Meshes capture
  (issue #317). Landed: c7bbca02, c9977805, 838028c6, b81f1c1f, a7a9ac4a,
  e474ff90, 10027e12, 2a25a3ff.** A block now captures its CUDA graphs on a stream that
  the stage executor creates for it and keeps, and the seam copies capture on an owned stream too.
  The Mesh's block worker pool creates its own worker streams, one per block, instead of taking
  them from torch's shared pool of 32, so no two blocks share a stream however many blocks a Mesh
  has, and nothing in `src` draws from the shared pool any more. For a user this means a
  cubed sphere whose panels are split into several blocks (2x2, 4x4) and a Mesh of many blocks
  (8, 24, 40) run as CUDA graphs under the default execution and give the same bits as explicit
  eager; before, they aborted under capture or were refused. The split-panel refusal is deleted.
  The executor's exchange entry decides between a cut and a plain launch, and the lone-block
  special case in the exchange is gone. REDs: a many-block Cartesian Mesh aborts under capture;
  2x2 and 4x4 cubed spheres must capture; two blocks never share a worker stream at 33 or more
  blocks. The default execution's time against explicit eager on the tested GPU is in the issue's
  correction comment (UCzhangxi/snapy#317).
- **Executor: one stage schedule for Mesh and bare blocks. Landed: 17346e57, 5f49cf32, 29999ba8, e74972ff,
  1d4ddeb3, 671ac480, fc0d191d, 1527e95e.** A block in a
  Mesh and a bare block now run the same stage schedule, eagerly or as graphs, through one run
  path; the hand-written chain `advance_stage`/`advance_core` and the `with_host` flag are deleted
  (`advance_local` stays for Python and the tests). One capture helper begins, ends and abandons a
  graph for the stage segments, the error path and the seam copies, and a graph whose capture
  failed to start is never destroyed half-registered. The advance op's prepare allocates the
  implicit correction, the integration execution keys are checked once by the options reader, and
  the executor asks the layout whether it is a cubed sphere; diffusion takes the cubed sphere's
  panel from the layout's `panel_of`. Nothing a user runs changes results, so no RED.
- **Register: one owner for booking and prescription. Landed: 83ef60c7, 2a20e4fc, ead839e3,
  2b0914c8, 39bd1d07, abd8670f, c7401fdb, 562303c5.** Every write that
  books an amount through `A_f (after - before)` goes through `overwrite_`, the one write-and-book,
  and through the wall batch; the separate wall-plane apply is folded into it. Every add and scale
  marks its direction for the post-implicit apply, so a source can no longer opt out by a flag.
  The prescribed faces are applied once per active direction after the sources' posts and before
  the divergence, so a source's post on a wall face never reaches du. A wall-batch slot is tied to
  the weights it was registered with, and the face-floor and Riemann event counts have one counting
  path. REDs: a carry without a settling part marks the direction; the wall batch books with
  rebuilt weights; a source post on the walls does not reach du.
- **Register: a prescribed-temperature x1 wall frees its energy row. Landed: fae998e0, e71206ef,
  715d8b39, 9e03e4d3, 5d77f14a.** A wall whose boundary prescribes a temperature has no mass flux but conducts heat, as on main: the
  register frees the energy row on that wall's plane in its one home, `supply_prescribed`, on the
  planes path and on the solid-mask path alike, so with a solid mask the x1 wall rows are freed
  too. Result-changing by design on decks with a prescribed wall temperature, see the table below.
  REDs: a prescribed wall temperature conducts in a full step; the solid-mask prescription frees
  the x1 wall rows.
- **Register: one seam transport. Landed: 1b3add90, a7834bdd, 15c32d63, 610e4509.**
  `Exchange::resolve_seam` and
  `Exchange::seam_exchange` are the one transport for every block-to-block slab: the panel-seam
  reconcile packs, exchanges through them at the executor's cut, and unpacks; `seam_slab` is the
  allocating wrapper; the x1 relays (`x1_rows`, `x1_seam_slab`) are the seam sides 0 and 1 of
  dimension 1 of the same transport, with their message tags unchanged; the board buffer is sized
  through `ensure_like` like every other exchange buffer; `Exchange` takes the panel and the
  cubed-sphere test from the layout. No result changes, so no RED beyond a unit test of the
  two-block column's peers, sides and ghost rows.
- **Register: diffusion and the implicit solve book their transfers with a direction off.
  Landed: 875ec321, 3d898fd7, 28b84f9b, b860b16e.** A disabled flux direction keeps its register total defined
  and zero every stage, so diffusion always books its energy rows in the register and the implicit
  solve always receives the register and the x1 total, also with `disable-flux-x1`. The diffusion
  energy fallback through `add()` and its `posted_` buffer, the solve's null-register branch and
  `X1Momentum`'s rebuild of the rate from the correction buffer are deleted. REDs: a disabled
  direction's register carries the diffusion energy; the implicit solve books its transfers with
  the x1 flux off.
- **`conversion_trips` is an `EfStatus` slot. Landed: d0d83bf6.** The
  conversion-order 4 fallback count is bound to its own `EfStatus` slot like the positivity meters,
  so the capture save list saves and restores it with the other meters; its separate hydro buffer
  and the name lookup in the save list are deleted. The counter's name, type and allocation are
  unchanged, so no RED; the graph-equals-eager meter test pins it.
- **`PrecisionPolicy` holds the wide-dtype and implicit-dtype rules. Landed:
  22901b5f, bf57b973, 9e7f5205, 27a2cc75, 66ad2ca9, c6961197.** `PrecisionPolicy::wide()` is the one rule for the wide dtype on a
  device (FP32 on MPS, which has no FP64; FP64 elsewhere), used by the register, the positivity
  limiter and the stage combine in place of their copies. The implicit solve's dtype and its FP64
  retry's dtype are read from the policy's implicit row, not from rules of their own. The block's
  policy follows the block's device, taken from the caller rather than from a coordinate tensor
  (a cylindrical block holds none), so on MPS the implicit row no longer keeps the CPU's FP64
  storage. No reachable path on CPU or CUDA changes, so no RED; the retry-path dumps are
  byte-identical.
- **One x1 stencil, one key check, a layout option table. Landed: 2465110c, 52d1be7b, d1c47a11,
  8c7dd9ca, baf29561.**
  `CoordinateImpl::x1_face_difference` is the only x1 face-difference stencil; the divergence and
  the gravity work both call it, each with its own face areas, so an FP32 run with FP64 geometry
  (`dynamics.precision: float32`) gives the same bits as before. `ReconstructOptionsImpl::from_node`
  takes the keys its caller allows and does the one key check; an unknown key, and `scale` under
  `scalar/reconstruct`, are still refused. `src/options/layout.yaml` is a struct table like
  `internal_boundary`: it generates the `distribute:` members, their defaults, their deck reader and
  their Python options, with the defaults unchanged. The coord rows and `blocks_per_process` stay
  out of the generation: their deck values differ from the members they feed, and each value
  already has one home; `blocks_per_process` has its one home in the layout. No result changes; REDs only where a reader changed (a test that `scale` is
  refused under `scalar/reconstruct`).
- **Executor: a capture out of device memory stops with a clear error. Landed: 0affeb86, 17c05904,
  9cf0bcb2.** A capture that runs out of device memory on a many-block Mesh now stops with a clear
  error that names `execution: eager`, instead of hanging. RED: a block job that fails on one block
  of a Mesh never hangs it.
- **Executor: one owned-stream helper; two loops call the existing ones. Landed: 0e89dbbc,
  410dc083, 73c45a7c.** The Mesh's worker streams, a block's capture stream and the seam copies'
  stream come from one helper. A stream is lent to one owner at a time; when its owner goes away the
  stream is given back, never destroyed, and the next owner reuses it. The x1 side of the seam
  transport takes its peers from `x1_seam_peers`, and `run_local` runs its segments through
  `run_eager`, instead of loops of their own. No result changes, so no RED; the dump set is
  byte-identical in FP64 and FP32 on CPU and CUDA.

## Result-changing rows and their bounds

Everything not listed here is byte-identical to the base on the dry-run gates (the #349 dump set
in FP64 and FP32, the #348 face decks, nb/pz, x1_golden) on CPU and CUDA. Bounds are relative to
the field (row) maximum unless stated. Evidence: `aam/w3fin/report.md` (d3178f20 vs main deeabfee)
and `docs/v3/closing/dry_run_plan.md`.

| row | what moves | bound (measured) | declared |
|---|---|---|---|
| b316baf4, explicit species sums | the moist decks, FP64: #349 7 files (4 moist sedimentation cases, hydro_u and scalar_s); x1_golden jupiter_crm_fp64; nb/pz cart3d and cart3d_pert (44/49 dumps) | #349 IVX 1.2e-13 CPU / 1.5e-13 CUDA, other fields <= 2e-15; x1_golden 1.2e-12 (one ulp of the state, equal to the 1-ulp u0 mutant); nb/pz <= 4.3e-15 of the dump max | "round-off change in the moist decks, FP64 <= 1.2e-13" |
| e341ffdf, FP32 typed constants | every FP32 run on a weno/LMARS deck: #349 FP32 21/23 files; x1_golden cart_moist_fp32, jupiter_crm_fp32 | FP32-vs-own-FP64 worst unchanged (1.703e-3 vs 1.707e-3, gnomonic_weno5s IVX; per-field ratio 0.71-1.46); x1_golden at the mutant's level (2.4e-4, 1.6e-5) | "FP32 re-baselined against FP64; FP64 bitwise" |
| 4d41d797, the FP32 sweep gauge | CUDA jupiter_crm_fp32 | 1 ulp | "on CUDA jupiter_crm_fp32 moves by 1 ulp" |
| faa65213, boundary velocity reads the stored sine | declared round-off; the audit gates (#349 FP64/FP32, #348 face, nb/pz) were byte-identical on CPU and CUDA at the first audit's sm_75 check | 0 on the gates | "round-off" |
| kintera 2.6.13 (f7034194): Rgas 8.31446 -> 8.31446261815324 (3.1e-7 relative) | every moist result; the seam test's energy flux sum | moist dump fields <= 1.1e-9 (nb/pz 44/49 dumps differ, the 5 dry sph_face dumps identical; measured on d3178f20); seam energy sum 7.6e-7, reference re-recorded 342168.95387051086 -> 342168.69510693091 (momentum sum unchanged to 1.5e-11) | pinned exactly so every leg compares like for like |
| macOS fix 7a70d38f (SNAP_SQR macro) | arm64/AppleClang only: lmars, hllc, roe, WENO and the implicit decomposition fuse multiply-adds as before ff66651a | Linux: #349 byte-identical, ctest by name identical; macOS: test_fp32_typed_constants 4/5 differing -> 0/0 | restores the pre-ff66651a numbers on contracting compilers |
| 2642c0a5 | one new compiler warning (tests/test_riemann.cpp:585, mkstemp) | test-only | noted on #309 |
| the pre-divergence apply and the conducting wall (c7401fdb, e71206ef, 9e03e4d3) | only decks where a source posts on a prescribed wall face: a diffusion or boundary-function temperature source on an x1 wall. The wall's conduction reaches du as on main; a source's post on a wall without a prescribed temperature no longer reaches du. No #349, #348, nb/pz or x1_golden case moves; the register's prescribed amounts move at round-off on the folded path | 0 on the gates; the conducting-column test holds the conduction to 1e-9 relative and the reflecting wall to 1e-12 of the column's energy | "the fixed-temperature wall conducts; other wall posts are removed and booked" |
| CUDA graphs | none in eager mode; graph replay equals eager bitwise | 0 (per-step hashes, FP64 and FP32, dt changing) | bitwise |

Environment-only failures, identical on base and head on every machine used: the Eigen entries
the ctest regex catches (Not Run), the restart/NetCDF tests on a NETCDF=OFF build, test_straka,
test_exchange_ucx_cuda (two GPUs), test_python_import_path (sandbox path), the budget tests without
NetCDF, the six nb/pz sph cell runs (refused since the gravity-work cell was removed).

## kintera pin

- `v3/final` ships pinned to kintera **2.6.13**: CI installs `kintera==2.6.13` on every leg
  (`.github/workflows/ci.yml`), `pyproject.toml` requires `kintera>=2.6.13`, `CMakeLists.txt`
  `find_package(Kintera 2.6.13 REQUIRED)`, release.yml pins `kintera>=2.6.13`. 2.6.13 is on PyPI
  (chengcli/kintera#159, v2.6.13): exact SI Rgas, `nfail_` dropped, third-body and falloff layout
  fixes.
- Why the exact pin: 2.6.12 prepares the moist column's thermo state on MPS (kintera #158), 2.6.11
  asserts in PyTorch's dispatcher at the retry's prepare; 2.6.13's Rgas moves the moist numbers
  (see the table), so every leg must compare like for like.
- The baseline (main deeabfee) does not build against 2.6.13: it reads `pthermo->nfail_`
  (src/mesh/meshblock_redo.cpp:124-126), dropped in 2.6.13. The baseline is not patched. Per
  `docs/v3/closing/dry_run_plan.md`: the correctness and speed legs build **both sides on kintera
  2.6.12**, and one V3-only leg, V3@2.6.12 vs V3@2.6.13, measures the kintera effect by itself.
- Building a V3 head on 2.6.12: heads after f7034194 carry `find_package(Kintera 2.6.13 REQUIRED)`
  and refuse the 2.6.12 package at configure. The dry-run driver (`docs/v3/closing/tools/dryrun.sh`)
  lowers that one floor to 2.6.12 for the k2612 configure only and restores the tree (markers in
  its log); no source changes. The seam test is RED on 2.6.12 by design (its reference is 2.6.13's).

## Gates on the landed head

The gates and their owners are in `docs/v3/closing/dry_run_plan.md`. On the landed head
`v3/final` 73c45a7c:

- Serial ctest, full, compared by name on three machines. CPU-only: 201 passed; the only
  non-passes are the environmental set unchanged from 168de4ef and 4c023985 (12 failed, 3 not run
  for the two-GPU names, the CUDA-only Python tests skipped). arm64 with the MPS tests: 202 of 227
  passed, against 196 of 226 on 168de4ef; test_coordinate and
  test_gravity_work_radial_exact_python went green, test_mps_matches_cpu_python passed all 28
  arms, and BlockPolicyFollowsTheBlockDevice and ImplicitStageRunsOnMps pass. CUDA: the full
  serial ctest compared by name with the 493bf5e6 run.
- Fork CI on 73c45a7c is green: pre-commit, bitwise-cpu, ubuntu and macOS (run 45757541948).
- The #349 dump set against 168de4ef: 23 of 23 dump files sha256-identical on CUDA FP64, CUDA
  FP32, CPU FP64 and CPU FP32.
- A large cubed-sphere Mesh that runs out of device memory now stops in seconds with the error and
  a clean exit, where it used to hang.
- Against main deeabfee, on the rehearsal head: the #349 dump set (FP64 and FP32, CPU and CUDA),
  the #348 face decks, nb/pz and x1_golden are identical except the rows of the table above, each
  within its declared bound.
- Graph equals eager, bitwise, FP64 and FP32, with dt changing every step: on every captured deck,
  including the moist four-block Mesh with every `EfStatus` meter compared, the stretched cubed
  sphere, 2x2 and 4x4 cubed spheres and the 8-, 24- and 40-block Meshes.
- The speed gate: the CUDA default execution runs within the ruled bound of main eager on every
  cell, the CPU eager path within its bound, and the CUDA explicit-eager gap is reported. The table
  is `docs/v3/closing/s4_eager_gap.md` (sections 5.5 and 5.6 for the landed head; the bound as
  ruled in section 1). On a GTX 1080 Ti the default on a single-block deck is 1.04-1.12x main eager
  on both V3 heads (4c023985 and the combined head), so it is not a regression of this batch, while
  the multi-block decks run at about half of main (c3 and S3 at 0.47-0.53 of main).

## Follow-ups (not in this PR)

- The Mesh chemistry path (UCzhangxi/snapy#318): in a Mesh, the blocks' stage skips the chemistry
  op (`defer_chemistry`) and the Mesh runs chemistry itself after the stage, so a deck with both
  chemistry and the closure runs chemistry after the closure. Making the Mesh run chemistry through
  the stage schedule like a bare block waits on a working deck that exercises both and on kintera's
  process-wide `LogSVPFunc::_op`, which running chemistry on the Mesh workers needs changed first.
- The CUDA eager gap: V3 explicit eager is slower than main eager by the amounts recorded in
  `docs/v3/closing/s4_eager_gap.md`; the remaining launches are the per-stage census and register
  work on the eager path, inside a graph they replay for free.
- Capture scope: capture allows the ideal-gas and ideal-moist EOS with lmars or hllc only; other
  EOS and solvers, more than one process and pz > 1 stay refused until each is tested; the FP32
  cubed sphere cannot be built.
- The FP64 retry-column count is kept twice (implicit_hydro.cpp and hydro.cpp, reset differently);
  the meters are not a bypass. Left for the implicit and hydro owners.
- In an FP64 block, swapping `hydro_w` to FP32 after initialize aborts with heap corruption, with
  the fallback off; not the retry's doing and not changed here.
- O3 (deferred by the maintainer): later, as its own PR.
- #369 (terrain, the per-column immersed solid on the x1-wall path): after V3. The column case (a
  Cartesian rest column over a solid, uniform vs x1rat 1.05) is still
  present at b84b8bdb: earth fill redoes at step 1, matched fill leaves a steady 2-cell w; the wall
  twin sits at round-off.
- Known, not changed: the single-block gnomonic decks' sensitivity to the boundary.prescribed item;
  test_python_import_path depends on the sandbox's Python path.
