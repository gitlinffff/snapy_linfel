Lane X: execution (precision, CUDA graphs, mesh, integrator, radiation)
=======================================================================

Owner: Xi Zhang. Round-2 reviewers: as assigned in the lane X thread.
Every ``file:line`` below is at the pins of section 1, except pyharp lines describing the code after P1, which are at P1's head. "Hypothesis" marks a
claim not yet settled by a test; its gate is named next to it.

1. Scope and pins
-----------------

Pins: snapy ``9e5de2d4``, kintera ``3f5ff938``, pyharp ``6bc01d53``,
pydisort ``9e7f0075``.

Lane files, the brief's baseline of 4,274 lines: snapy ``src/mesh/*``
(3,266) and ``src/interface/*`` (1,008). Each directory is counted with::

  git ls-files src/mesh | grep -E '\.(cpp|hpp|h|c|cu|cuh)$' | xargs cat | wc -l

Provider side, outside the percentage, where touched files do not grow:
pyharp ``src/integrator`` (564) and pydisort ``src`` (1,590). The precision
code in ``src/utils`` (``fp32_storage.*``, ``fp64_store.hpp``,
``tensor_cast_cache.hpp``, 313 lines) is designed here and counted in lane O.

Snapy has no CUDA-graph source at the pin; the prototype is a test,
``tests/straka_cuda_graph.py``.

Out of scope here: the hydro pipeline (R), exchanges (S), EOS and limiters
(EF), kintera (K), output and restart format (O). This design states what
the executor needs from each of them (section 11).

2. Concepts and current sites
-----------------------------

``src/mesh`` plus ``src/interface`` moved from 3,989 lines at ``919936da`` to 4,274 at the pin
(``meshblock.cpp`` 1,689 to 1,919; ``mesh.cpp`` 543 to 566; ``.item<``
lines in ``src/mesh`` 23 to 26).

.. csv-table::
   :header: "Concept", "Site at the pin"

   "precision switch", "src/hydro/hydro.hpp:332; option read at src/hydro/hydro_options.cpp:67"
   "precision branches in the lane", "src/mesh/meshblock.cpp:35, src/mesh/meshblock.cpp:41, src/mesh/meshblock.cpp:448, src/mesh/meshblock.cpp:534-538, src/mesh/meshblock.cpp:1798"
   "FP32 storage emulation (env; removed, Cheng)", "src/utils/fp32_storage.hpp:12-26; applied at src/mesh/meshblock.cpp:621-642"
   "RK backup buffers, FP64 at creation", "src/mesh/meshblock.cpp:306-312"
   "backup retyped on first use", "src/mesh/meshblock.cpp:719-721, src/mesh/meshblock.cpp:741-743"
   "state rebinding per stage", "src/mesh/meshblock.cpp:901-902, src/mesh/meshblock.cpp:921-922"
   "RK combine (allocating)", "pyharp/src/integrator/integrator.cpp:105-126"
   "RK weights, exact sum", "pyharp/src/integrator/integrator.cpp:59-64"
   "stage driver", "src/mesh/meshblock.cpp:698-1007"
   "exchange inside stage 0 (implicit radial diffusion)", "src/mesh/meshblock.cpp:714-718"
   "exchange split pack/launch/unpack", "src/mesh/meshblock.cpp:664-685; callers src/hydro/hydro_forward.cpp:1017, src/scalar/scalar.cpp:120-146"
   "cubed-sphere two-round exchange", "src/mesh/meshblock.cpp:651-658"
   "axial closure (host copy, 2 reads, 1 branch)", "src/mesh/meshblock.cpp:1014-1034"
   "gravity-work fixer (host sums, branch, abort; removed, Cheng)", "src/mesh/meshblock.cpp:1036-1109"
   "multi-block global phase", "src/mesh/mesh.cpp:350-407"
   "block worker pool and streams", "src/mesh/mesh.cpp:66-227, src/mesh/mesh.cpp:284-297"
   "global dt", "src/mesh/mesh.cpp:325-348; per block src/mesh/meshblock.cpp:576-595"
   "retry decision", "src/mesh/meshblock.cpp:1506-1601; mesh src/mesh/mesh.cpp:465-507"
   "rollback", "src/mesh/meshblock.cpp:1456-1504"
   "graph prototype", "tests/straka_cuda_graph.py:44-90"
   "radiation provider", "pyharp/src/integrator/radiation_model.cpp:27-34, pydisort/src/disort.cpp:439"

3. Target design
----------------

Types (declared in section 12):

- ``PrecisionPolicy``: built once from ``dynamics.precision``; one row per
  quantity (section 6); ``storage()``, ``compute()``, ``cast()``. Replaces
  every ``fp32() ? a : b``.
- ``StateStore``: owns the fixed buffers ``hydro_u``, ``scalar_s``, ``u0``,
  ``s0`` and ``scalar_r`` (rebound at the pin by ``src/mesh/meshblock.cpp:44-56`` at 631, 923, 1199, 1484 and, on restart, 1743 or 1828; section 7) in the storage dtype, allocated once at ``reset()``. Addresses
  change only on ``reset``, ``to()`` and restart, each of which bumps a
  generation counter.
- ``Segment``: an ordered list of device ops with one ``SegmentKind``
  (compute, pack, host-exchange, unpack, global, eager) and a ``capturable``
  flag set by whoever registers the op.
- ``StageSchedule``: the segments of one RK stage, built once per
  configuration.
- ``StageExecutor`` (owned by ``Mesh``; a bare ``MeshBlock`` owns a one-block one): runs a schedule eagerly or by
  replaying one graph per (block, stage, segment); eager is the reference.
  It holds one chemistry adapter per block, indexed by block.
- ``GraphKey``: device, shapes/strides/dtypes, StateStore generation,
  configuration generation, stage, precision, dt, and the
  (``options_digest``, ``epoch``) pair of each kintera table held and the
  ``kintera::alloc_epoch(m)`` of each kintera module ``m`` in the segment, K's free
  function (K 3.3; skeleton ``K/tables.hpp:59`` at ``ef93e92d``).
- ``MeshBlock`` keeps its API except one type (L-2, accepted by Cheng): ``user_output_callback``
  (``src/mesh/meshblock.hpp:125``) becomes ``std::function<Variables(Variables const&, FluxRegister const&)>``;
  O calls it with the block's register (``src/output/load_user_output_data.cpp:14``, ``src/output/load_user_output_data.cpp:19``),
  and each assigner takes the second argument: ``examples/bryan.cpp:80``, ``examples/jupiter_evap_precip_1d.cpp:66``,
  ``examples/run_hydro.cpp:145``, ``examples/straka.cpp:42``, ``tests/test_user_output.cpp:194``, ``tests/test_user_output.cpp:409``,
  the binding ``python/csrc/pymesh.cpp:173`` (``python/snapy/mesh.pyi:145``) and the docs
  examples (``docs/user_guide/output.rst:488``, ``docs/user_guide/simulation.rst:195-209``, ``docs/api/mesh.rst:106``). The same X PR binds ``FluxRegister`` for Python: a class with R's const views only (R skeleton ``flux_register.hpp:82-84`` at ``97c62312``), which the wrapper at ``python/csrc/pymesh.cpp:170-176`` passes to the Python callable by reference, never owned by Python. The reference and its views are valid only during the call and read-only: the callable clones what it keeps and writes to none; the stub ``python/snapy/mesh.pyi:145`` becomes ``Callable[[Variables, FluxRegister], Variables]``, with a ``FluxRegister`` stub. A one-argument callback must be updated: in C++ it no longer compiles, and a one-argument Python callable fails at its first call. Its body splits into six parts (item 6 of the
  thread): init, advance, exchange, redo, restart, output.

Ownership: ``Mesh`` owns the executor and the global phase (reductions over
blocks and ranks); a block inside a ``Mesh`` uses the Mesh's executor, a bare
``MeshBlock`` owns a one-block executor. Each ``MeshBlock`` owns its ``StateStore`` and contributes
segments. Lanes R, S, EF, K register ops; X schedules them.

Data flow of one step::

  host: dt = min over blocks, ranks           (outside graphs)
  step start: kStepStart list (register step zero, radial post; its host check ends the segment)
  for stage s:
    [pack]  -> host UCX -> [unpack, scalar_r divide]   exchange = segment boundary
    [compute: hydro fwd (R) | seam exchanges (S) split it]
    [compute: scalar, forcings, RK combine out=hydro_u] last stage:
     finish_step, its host check ends the segment
    [compute: every stage: after_combine: limiter (EF), X's scalar slot (920-923), solid (S)]
    [global: device partial sums] -> host allreduce -> [global: apply]
    [boundaries]
  after the last stage: [chemistry slot: the driver's ChemistrySource, eager FP64]
  host: one read of the stacked retry flags -> MAX allreduce
        -> reject: kRollback list on all blocks, rerun from step start
        -> accept: O's observe (O's flush makes the amount SUM)

Without a process group (one rank: ``src/layout/layout.hpp:213-217``,
``src/mesh/meshblock.cpp:986``) there is no allreduce, and the host segment
of the global phase is absent: the device sums feed the device apply in the
same graph.

Hooks (``stage_executor.hpp``, ``meshblock_parts.hpp``). ``step_start(dt)``
runs the ``kStepStart`` entries once per attempt, before the stage-0 resets:
R's ``begin_step()``, then the radial flag's ``zero_``, the radial half step and its
``post_split`` (``src/mesh/meshblock.cpp:714-718``: the split at 715, then the
boundaries and ghost exchange at 716-717, all before the ``u0`` save at 721 and
the resets at ``src/mesh/meshblock.cpp:722-738``); its radial host check
(``src/forcing/diffusion_radial.cpp:97-100``) ends a capture segment, as the
one in ``finish_step`` does (A.rst:550-555 at ``4e0827b8``). ``kStageZero`` takes the ``u0``
backup, then K's ``snapshot_active_set``, then ``reset_failures``, which
replaces the host drain at ``src/mesh/meshblock.cpp:736`` (K 3.2, 3.4), and
the pin's other per-step latches (``src/mesh/meshblock.cpp:722-738``: heat
capacity, VIC clamp and solve, limiter marks, vapor failures; the
gravity-work lines 724-730 go with gw1); stage zero resets per-step slots
only. The positivity census, the run-to-date meters
(``src/hydro/hydro.hpp:263-271``, ``src/scalar/scalar.hpp:109-111``), has two
periods. Before R's PR R-1 the meters stay as at the pin, rejected attempts
included, with no interim ``ef.census`` hook. From R-1 R's monitor counts
accepted steps only: ``begin_step()`` zeroes, ``rollback()`` drops, and O's
``observe`` makes the lifetime sums. ``StepReset`` has no ``commit`` or
``drop`` hook: their one client at the pin is the gravity-work fixer
(``src/mesh/meshblock.cpp:726-730``, ``src/mesh/meshblock.cpp:1477``,
``src/mesh/meshblock.cpp:1502``), which goes with gw1.
``finish_step(dt)`` runs on the last stage after the combine
(``src/mesh/meshblock.cpp:903-905``) and ends that segment; X's ``after_combine`` slot (R's phase) follows every
stage's combine, so the limiter (``src/mesh/meshblock.cpp:907-909``) comes
after ``finish_step`` on the last. A rejected step runs the ``kRollback``
entries in the pin's order (``src/mesh/meshblock.cpp:1476-1504``): ``u0``,
the radial ``restore_step`` (``src/mesh/meshblock.cpp:1479-1480``), K's
``restore_active_set``, the cons2prim, ``s0`` and the scalar primitives
(``src/mesh/meshblock.cpp:1481-1484``), then R's ``rollback()`` with the
limiter marks, after the cons2prim (``src/mesh/meshblock.cpp:1488-1489``);
none erases a step-start post in place, since the retry
reruns ``step_start`` at the new dt, as the pin re-enters stage 0. The retry
stack calls ``consume_failures(flags.narrow(0, 4, 1))``, slot 4
(``src/mesh/meshblock.cpp:1559-1560``).
``observe()`` runs after the accept.

Retry stack: 0-d flags in ``hydro_u``'s dtype (``src/mesh/meshblock.cpp:1527``),
read once as FP64 (``src/mesh/meshblock.cpp:1566``). The slot loop
(``src/mesh/meshblock.cpp:1569``, bound ``out.size()``) keeps six slots: its
bound becomes a literal 6 when ``out`` (1568) widens to seven, and the widened
``out`` is value-initialized, ``std::array<bool, 7> out{};``
(``src/mesh/meshblock.cpp:1568-1569``), so its seventh entry is never read
uninitialized and no spurious retry follows. Slots 0-5 are cause bits
0-5 (floor, clamp, limiter, nan, saturation, vic-solve); slots 6 (vapor) and
7 (heat) fold into bit 2 (``src/mesh/meshblock.cpp:1571-1577``). A seven-slot
loop would read vapor slot 6 as bit 6. Only the cause
arrays gain a seventh entry: ``std::array<bool, 6>`` at 1565, 1568, 1581,
1585 and ``src/mesh/mesh.cpp:476``, the buffer at 1586; the loops over causes
(1588, 1595, ``src/mesh/mesh.cpp:489``) then run to seven. Slot 8,
``diffusion.radial``, maps to cause bit 6 (value 64) by one explicit line
after 1577 and is reserved in V3: named, mapped, zeroed in ``kStepStart``,
written by nobody (Cheng's decision, X thread 13:41 PT). A failed radial-conduction solve stops the run, as at the pin
(``src/forcing/diffusion_radial.cpp:97-100``); it is not a rejected step.
That file, its change and its count are lane A's, and A revision 4 has no
radial cause path (A.rst:148-151, 703-712, 1125-1128, 1168-1171 at
``4e0827b8``). The rollback and retry contract has no radial path.

The stage form is today's Shu-Osher combine (weights
``pyharp/src/integrator/integrator.hpp:20-29``); a two-tableau form is
optional and outside this round (section 10, last subsection).

4. Flux-register contract (executor side)
-----------------------------------------

Names and signatures are lane R's header (``docs/v3/skeleton/R/flux_register.hpp``
at R rev 2.2 ``4c59162a``).

- Storage: ``FluxRegister::reset(coord, policy, nvar, req)``
  (``flux_register.hpp:43-44``) takes no storage and allocates its own,
  fixed at ``reset()`` (R.rst:1395-1396); X's ``BlockInit`` calls
  ``StateStore::register_storage`` on R's tensors after ``reset()``, and a new
  allocation bumps the store generation. ``set_prescribed`` writes the fixed
  masks per solid-mask version (R.rst:1396-1397) with ``copy_`` into the
  buffers allocated at ``reset()``, so addresses hold; ``config_generation``
  has one owner, X's ``StageExecutor::bump_config_generation()``, through
  which every caller goes (``set_prescribed``, option and solid-mask
  changes); ``store_generation`` does not change. R and the save list read
  R's tensors back through ``StateStore::storage(name)``. Never in restart (derived state). Its dtype is a ``PrecisionPolicy`` row;
  ``step_field`` is FP64, FP32 on MPS (R's section 6).
- Posts: device ops inside compute segments (R's pipeline stages).
- Begin stage: the executor calls ``begin_stage(int stage,
  harp::IntegratorWeight const&)`` with that stage's pyharp weight
  (``pyharp/src/integrator/integrator.hpp:20-29``). ``end_stage(dt)`` runs
  R's recursion ``S = w1 S + w2 dt T`` (R's 3.4), so no stage table is
  passed. The weight is a host double fixed for the run, baked into a graph
  keyed by stage; dt is a host double.
- Reads: the register's exchange is an exchange phase (pack, host, unpack);
  it ends the posting segment.
- Corrects: per stage, as R settled: a device op in the segment after the
  unpack, before the RK combine (now ``src/mesh/meshblock.cpp:901``); X
  enters its ``after_combine`` slot, R's phase, after each stage's combine (the limiter, now
  ``src/mesh/meshblock.cpp:907-909``, and the fixers), on the last stage
  after ``finish_step`` (R.rst:691-701; ``flux_register.hpp:22``). R rev 4.3
  (``97c62312``, ``hydro_stage.hpp:67-70``) removes ``HydroStage::after_combine``:
  ``StageExecutor::after_combine(stage, dt)`` is the one runner.
- Reset: R's ``begin_step()`` (``flux_register.hpp:48``; R.rst:31-32,
  70-73) is X's first ``kStepStart`` entry, before the radial
  ``post_split``; ``begin_stage(0)`` zeroes nothing. ``rollback()`` is the
  last ``kRollback`` entry, with the limiter marks (now ``src/mesh/meshblock.cpp:1489``).
- Monitor: R's ``RegisterMonitor``. Its ``retry_flags()`` join the step-end
  MAX read (``src/mesh/meshblock.cpp:1559-1568``,
  ``src/mesh/meshblock.cpp:1586-1591``, ``src/mesh/mesh.cpp:485``): each
  int32 slot (``flux_register.hpp:102``) is cast to the stack dtype as one
  ``RetryFlag`` whose cause bit X maps (R.rst:1256-1258): EF's
  ``ef.floor_patch``, ``ef.vapor_fail`` and ``ef.heat_capacity`` to bit 2,
  ``ef.nan`` to bit 3 (EF.rst:675-685, 2095-2109 at ``f5c8e9a7``; pin
  ``src/mesh/meshblock.cpp:1559-1577``), I's
  ``implicit.column`` to bit 5.
  Until R-1, EF's status stays in X's stack (R.rst:588-589). Its FP64
  amounts belong to O: after the accept X calls ``Consumer::observe``, which
  accumulates them on the device every accepted step, and ``flush`` makes the
  collective and the host read at its cadence (O.rst:307-314, 369-372,
  712-716 at ``c12f78bb``). X keeps only the MAX retry read at step end.
- Fixers: R's ``Fixer`` is the one declaration; ``GlobalCorrection`` holds
  a list of them and declares no ``apply``. The global phase runs every
  ``partial_sums()``, one FP64 collective per stage, then
  ``Fixer::apply(u, sums, dt, stage_weight, reg)`` with ``stage_weight``
  that stage's ``w2`` (now ``src/mesh/meshblock.cpp:1027``), not R's step
  coefficient. A's ``AxialClosureFixer`` is the first client.
- Ordering: post before pack; unpack before correct; correct before the
  stage-end exchange (``src/mesh/mesh.cpp:406``).
- No added full-field copy: the register holds face data only.

The ten register requirements, by number:

.. csv-table::
   :header: "Req.", "Requirement", "X's answer"

   "1", "one owner of every face flux", "not this lane (R)"
   "2", "named increments in a declared order", "not this lane (R); posts stay inside one compute segment, in program order"
   "3", "stage weights into the step transfer", "``begin_stage(int stage, harp::IntegratorWeight const&)``, one host weight per stage; R's recursion needs no table; a graph keyed by stage bakes it in"
   "4", "divergence reads the register", "not this lane (R, C)"
   "5", "two entries per shared face", "not this lane (R, S); X schedules the register's exchange as an exchange phase"
   "6", "one device monitor, no host read in a segment", "``RegisterMonitor::retry_flags()`` joins the step-end MAX read (``src/mesh/meshblock.cpp:1566``); amounts go to O's ``observe`` after the accept; section 7 bans host reads in segments"
   "7", "every fixer and floor registers its addition", "``Fixer`` is R's; X keeps the registry and its collective; fixers post ``cell_amount``"
   "8", "stated dtype, fixed storage", "R's ``reset()`` allocates it, ``StateStore`` registers it (fixed addresses); dtype is a ``PrecisionPolicy`` row, the step accumulator FP64 (MPS: FP32)"
   "9", "no added full-field copy per stage", "face data only; ``out=`` removes today's copies (section 8)"
   "10", "rebuilt, never in restart", "derived state: ``reset()`` at set-up, ``rollback()`` a ``kRollback`` entry"

5. Invariants and gates
-----------------------

.. csv-table::
   :header: "Identity", "Test", "Comparison"

   "eager result after X1, X2, X5 equals the pin", "N-step golden of hydro_u and scalar_s, Cartesian and cubed sphere, FP64 and FP32", "bitwise"
   "out= RK combine equals the allocating form", "pyharp test: each scheme and stage, FP32 and FP64, out aliasing u1 and distinct; a wrong-shaped (empty) out is resized to u0's shape", "bitwise (x86 CPU: NaN payloads may differ)"
   "state addresses fixed", "data_ptr of hydro_u, scalar_s, scalar_r, u0, s0 unchanged over 10 steps with a retry", "exact"
   "no host sync inside a segment", "replay under torch.cuda.set_sync_debug_mode('error')", "exact"
   "graph replay equals eager", "Straka, both precisions, dt changes, a retry", "bitwise (hypothesis); else the tolerance stated by the test"
   "flags survive replay", "a limiter repair and a vapor failure injected on replay N >= 2 reach the retry flags; warm-up, capture and one replay leave the save list (section 7) equal to eager", "exact; save list bitwise"
   "radial failure: no change in V3, abort kept (slot 8, bit 6 reserved)", "none: no writer in V3", "-"
   "rollback restores the step", "inject a NaN; after retry, state equals the pre-step state, registers zero", "bitwise"
   "dtype per quantity", "policy test: each row of section 6 in both modes", "exact"
   "global corrections unchanged", "closure on 1 and 4 blocks, 2 ranks", "bitwise if block partials keep their order (hypothesis)"
   "chemistry slot before Mesh::forward's final exchange", "owner X, unrun; the no-source baseline lands with X4 (global phase for 1 and N blocks), the source run with S-K4, after X4 and EF-8 (K rev 3.1 5a714053, K.rst:650-652). One deck, tests/test_parentless_cloud_nb1.yaml, one block against two (the split of tests/test_parentless_cloud_nb1.cpp:33-45) through both branches of Mesh::forward (src/mesh/mesh.cpp:354-360, 365-406). Initial state: the deck as loaded, then the EOS limiter set false on the block options before the Mesh is built, no restart, filled as tests/test_parentless_cloud_nb1.cpp:57-66 with deficit 0 (rain 0.001 in global cells 8-15, 0 below: non-negative, with a jump at the seam of the two 8-cell blocks). Limiter off, closure and gravity-work fixer off: the whole-column limiter is a cross-block gather with its own gate (that test), so off leaves the slot as the one new seam effect. 10 steps, first without a source, then with one from Mesh::set_chemistry, asserting the caller's uref_R unchanged after the call; compared: every owned cell of hydro_u and scalar_s after each step. N >= 2 catches a stale seam ghost (read at the next step's first stage); 10 let it spread across each block. A mutant with the slot after the final exchange must fail. Adds to the pin's harness (tests/test_radiating_boundary.cpp:472-517: one step, 3 ghost cells per side, allclose 1e-12): owned cells, N steps, species and the slot", "bitwise (hypothesis). The no-source baseline is strictly bitwise: a failure blocks X4. Fallback, for the source run only, triggered when its bitwise compare fails: per field and step, max |two - one| with the source at most A times the no-source baseline's, with a floor of A ulp, the ulp of each compared value in its storage dtype (FP32 or FP64); a cell whose value is 0 must match exactly. A is measured first, not argued: the one-block run against itself with one ulp added to one cell of one field (hydro_u's vapor row ICY in global cell 8; the deck has nx2 = nx3 = 1), chemistry on, 10 steps; A is the largest max difference over fields and steps, in ulp of each value (the slot is cell-local, so it adds no seam effect, but its implicit step and threshold branches can amplify round-off without a fixed bound). The mutant must exceed the tolerance by at least 10 times; if it cannot, the fallback is not a gate"

6. FP32 contract
----------------

.. csv-table::
   :header: "Quantity", "Storage", "Compute", "Why FP64 where FP64"

   "hydro_u (anomaly form)", "FP32", "FP32", ""
   "u0, s0 backups", "= storage", "copy", "bitwise rollback"
   "scalar_s", "FP32", "FP32", ""
   "fluxes, du", "FP32", "FP32", ""
   "reference and geometry masters", "FP64, cached casts", "per use", "anomalies cancel against them"
   "hydro_w totals", "FP64", "FP64", "EOS on totals (src/mesh/meshblock.cpp:448)"
   "gauges, total face pressure", "FP64", "FP64", "cancellation"
   "dt, budget and closure sums", "FP64 (MPS: FP32)", "FP64", "sums over all cells; MPS has no FP64 (src/hydro/hydro_forward.cpp:373)"
   "vertical implicit correction (``kImplicit``)", "FP32 on Cartesian ideal-moist only, elsewhere refused (``src/hydro/hydro.cpp:502-507``)", "FP32, FP64 retry of the failed columns only (Cheng, 2026-11-17; I section 6): with ``implicit-fallback-fp64`` on, a predicated second FP64 launch, always run (no conditional nodes), over the columns whose ``col_status`` marks a failed, not yet retried solve (the pin's other conditions, ``src/hydro/hydro.cpp:529-530``, join that predicate, I.rst:763 at ``fa3ad5e6``), from ``rhs64``/``wi64``, ``kImplicit`` forced FP64, posts in ``kFlux``, cause bit 5 through ``retry_flags()``, with no host read; ``implicit-fallback-fp64`` is read per call in eager, as at the pin, so ``tests/test_fp32_implicit.py:19, 62-67`` is unchanged; under ``cuda-graph`` it is read at capture, and a change after capture takes effect only through ``bump_config_generation()``, a behavior change listed in section 10. Before the change, the pin retried the whole block (``src/hydro/hydro.cpp:527-535``)", "a predicated second FP64 launch (I section 6)"
   "kintera totals, saturation", "FP64", "FP64", "phase equilibrium"
   "time, dt, stage weights", "host double", "host double", "rk3 weights sum to 1 exactly (pyharp/src/integrator/integrator.cpp:59-64, base 6bc01d53)"
   "flux register step accumulator", "FP64 (MPS: FP32)", "FP64", "sum over stages; settled by R (R's section 6)"
   "source accumulator", "FP64, one stage-local field", "FP64", "per-source budgets; today FP32 runs only (src/hydro/hydro_forward.cpp:1364-1374); A uses it in both precisions"
   "diagnostics", "int64/bool counters, FP64 sums (MPS: FP32)", "same", "replaces the ad hoc dtype at src/hydro/hydro_forward.cpp:373"

``store()`` is the one storage rounding, except EF's anomaly floor, which
rounds on purpose (``src/eos/equation_of_state.cpp:37-46``, used at
``src/eos/equation_of_state.cpp:247``), and section 7's in-place writes. MPS has no FP64: there every FP64
row holds FP32 (``src/hydro/hydro_forward.cpp:373``), anomaly storage is
refused (``src/utils/fp64_store.hpp:49``), and runs are eager.

Mechanism: ``PrecisionPolicy`` replaces the 29 callers of ``fp32()`` in ``src``
(hydro, EOS and mesh files) with ``policy.cast(Quantity, x)``. The lane's own five
calls go in X1; the others go with their lanes' PRs. ``u0``/``s0`` are
created in the storage dtype, so ``src/mesh/meshblock.cpp:719-720`` and
``src/mesh/meshblock.cpp:741-742`` go.

7. CUDA-graph contract
----------------------

Host reads, allocations and data-dependent branches in a step, lane X files
(other lanes' sites: ``.item<`` lines per directory at the pin are bc 11,
coord 7, eos 5, forcing 14, hydro 11, implicit 7, output 13, recon 2,
utils 7; each owner rules on its own).

.. csv-table::
   :header: "Site", "Kind", "Fate", "Gate"

   "src/mesh/mesh.cpp:344, src/mesh/meshblock.cpp:588", "dt read", "stays: step start, outside segments", "G1"
   "src/mesh/meshblock.cpp:714-718", "exchange in stage 0", "segment boundary", "G1"
   "src/mesh/meshblock.cpp:719-720, src/mesh/meshblock.cpp:741-742", "allocation on dtype change", "removed (policy)", "G1"
   "src/mesh/meshblock.cpp:736", "host drain of kintera failures", "device counter (K)", "G2"
   "src/mesh/meshblock.cpp:874", "user TorchScript forcing", "eager segment unless declared capturable", "G1"
   "src/mesh/meshblock.cpp:901-902, src/mesh/meshblock.cpp:921-922", "allocation + rebinding", "pyharp out= into fixed storage", "G1"
   "src/mesh/meshblock.cpp:44-56; called at src/mesh/meshblock.cpp:631, src/mesh/meshblock.cpp:923, src/mesh/meshblock.cpp:1199, src/mesh/meshblock.cpp:1484, src/mesh/meshblock.cpp:1743, src/mesh/meshblock.cpp:1827-1829", "allocation + rebinding of ``scalar_r``", "divided with ``out=`` into its ``StateStore`` buffer: 923 after the EF limiter (907-909, writes density), before the solid fill (936-938); 1199 X's op after each ``exchange_ghost_zones`` (its corner refresh, 1194-1195); 1484 a ``kRollback`` entry after ``s0``; 1828 the restart rebuild, or 1743 its stored copy when not rebuilt (1729-1730), outside a step (``StateStore::assign``); 631 goes with the emulation (fp1)", "G1"
   "src/mesh/meshblock.cpp:950-981", "saturation adjustment", "eager segment until kintera capture", "G2"
   "src/mesh/meshblock.cpp:1017-1018, src/mesh/meshblock.cpp:1023", "host copy, 2 reads, 1 branch", "``.item()`` reads removed: device sums and mask; allreduce is a boundary", "G1/G4"
   "src/mesh/meshblock.cpp:1061, src/mesh/meshblock.cpp:1075, src/mesh/meshblock.cpp:1080", "copy to host, branch", "removed with the gravity-work fixer (Cheng)", "G1/G4"
   "src/mesh/meshblock.cpp:1097", "abort on bound", "removed with the gravity-work fixer (Cheng)", "G1"
   "src/implicit/implicit_hydro.cpp:476-487", "finiteness screen: ``item<double>()`` every stage with the implicit correction", "segment end until its owner turns it into a device flag in the retry read", "G3"
   "src/mesh/meshblock.cpp:1517", "full-field clone in the retry test", "``StateStore`` scratch (answered, EF note 7, section 11)", "G1"
   "src/mesh/meshblock.cpp:1566, src/mesh/meshblock.cpp:1591", "one read, MAX allreduce", "stays: step end", "G1/G4"
   "src/mesh/meshblock.cpp:1288-1295, src/mesh/meshblock.cpp:1319-1337", "diagnostic reads", "stay: output cadence, outside", "G7"
   "src/mesh/meshblock.cpp:1426-1443", "legacy reads", "removed with the helpers", "G1"
   "src/mesh/meshblock.cpp:1628-1712", "restart reads", "stay: not in a step", "-"

Segment list of one RK stage, Cartesian: exchange (pack | host | unpack,
``scalar_r`` divide);
compute, split at every within-stage exchange below; scalar, forcings, RK
combine, limiter, scalar combine, ``scalar_r`` divide, solid fill; last stage, with reactions
(``src/mesh/meshblock.cpp:950-951``): the limiter again (``src/mesh/meshblock.cpp:952``),
after the solid fill, as the first op of the saturation adjustment's segment (eager until G2);
global phase (the axial closure when on); boundaries. On the last stage the
radial ``finish_step`` host check ends the combine's segment (section 3).
After the last stage and before the step-end retry read, with no reorder of
any pin op; the driver calls nothing extra. The three pin callers step
chemistry after ``block->forward`` returns, so the slot is the last op of the
last stage's ``MeshBlock::forward``, after its exchange and stage
(``src/mesh/meshblock.cpp:616-618``; 618 goes with fp1); the next call's
exchange-first (616) syncs the ghosts, as at the pin. ``Mesh::forward``
calls ``advance_local`` (``src/mesh/mesh.cpp:355, 366``), never
``MeshBlock::forward``, and exchanges last (359, 406); nothing refills the
conserved cell ghosts before the next stage's reconstruction except the
radial split's exchange (``src/mesh/meshblock.cpp:716-717``, implicit radial
diffusion only). With conversion order 4 the primitive ghosts are refilled
(``src/hydro/hydro_forward.cpp:490-491``); the x1 seam rows (249) and the theta
sync (``src/hydro/flux_positivity.cpp:184``) carry face data and limiter
factors, not cell ghosts. Its pin callers (``examples/bryan.cpp:420``,
``cubed_sphere_convection.cpp:131``, ``planar_convection.cpp:151``,
``shallow_splash.cpp:118``, ``shallow_xy.cpp:115``,
``spherical_convection.cpp:291``, ``straka.cpp:148``, and the
``mesh->forward``/``mesh.forward`` sites in ``tests/``) step no chemistry today: ``git grep`` of ``evolve_implicit`` and
``forward_with_rate`` finds only the three callers above. So none changes;
one that registers a source gets the slot after the per-block stage, the
global phase and the storage write, just before ``Mesh::forward``'s final
exchange, a new order without a pin golden, gated by one block against two
(section 5). Two paths, both S-K4's with their Python bindings (K rev 3.1
``5a714053``, K.rst:789), take the reaction options by value
(``kintera::KineticsOptionsImpl``) and each block's thermo options, as
``examples/run_hydro.cpp:64`` takes them. A bare ``MeshBlock`` (no ``Mesh``:
the three pin callers that build a ``Kinetics``, ``examples/run_hydro.cpp:37``,
``examples/jupiter_evap_precip_1d.cpp:200``, ``tests/test_fp32_jupiter.py:66``)
owns a one-block executor; ``MeshBlock::set_chemistry`` builds its one adapter
into slot 0. Its ``step_start`` and retry scope are that block, exactly as a
one-block ``Mesh``: the same ``kStepStart`` ops, ``max_redo`` and cause bits. A
block built by a ``Mesh`` holds the Mesh's executor, and ``MeshBlock::set_chemistry``
on it throws, naming ``Mesh::set_chemistry``, so no registration misses it. ``Mesh::set_chemistry``, called once, builds one adapter per block
(local index, ``Mesh::blocks`` order) into the ``Mesh`` executor's vector,
``StageExecutor::set_chemistry(block, source)``; each block's slot runs its own.
The copy is member-wise: values (thermo vectors, ``Tref``, ``Pref``, flags) are
copied; the eight reaction sub-options (kintera ``kinetics/kinetics.hpp:64-72``)
are ``shared_ptr``s shared with the caller. The first ``Kinetics`` build writes
only the copy's own members: it appends to the per-species vectors (kintera
``species.cpp:386-436``), fills a null sub-option slot with a new object
(``kinetics/kinetics.cpp:34-48``), shifts ``uref_R`` and ``sref_R`` and sets
``offset_zero`` (60-79); the sub-option modules only read theirs (e.g.
``kinetics/arrhenius.cpp:175-302``). So the caller's object is untouched with no
deeper copy; the caller must not edit the sub-options afterwards. ThermoX
writes its thermo options once at its build too, safe since builds are serial. A load rewrites the process-wide species list (kintera
``species.cpp:58``), so ``Mesh`` builds the adapters one after another on the
calling thread, outside the block pool (``src/mesh/mesh.cpp:284-297``, blocks run
concurrently); afterwards the copy is read-only and shared, and each block has its own ``Kinetics`` and
``ThermoX`` on its device (K rev 3.1 ``5a714053``, K.rst 382-405, "Blocks"); a race is a hypothesis, untested. It is snapy's adapter over kintera's
``Kinetics`` (S-K4; K.rst:372-380 at ``f84a5c58``), as an eager FP64
segment (K 3.6) when its stoichiometry is nonempty. Nothing else registers
one: snapy's ``src`` never steps the module it builds
(``src/scalar/scalar.cpp:54-55``), so a driver that registers no source has
no chemistry, as at the pin; no other driver gains it. Three pin callers step
their own ``Kinetics`` after all stages and before ``check_redo``:
``examples/run_hydro.cpp:171-198`` (gate at 177, ``check_redo`` 200),
``examples/jupiter_evap_precip_1d.cpp:242-261`` (built 227-228,
``check_redo`` 263, no gate) and ``tests/test_fp32_jupiter.py:116-129``
(built 67, again 108 after a restart; ``check_redo`` 130, no gate). S-K4
deletes each caller's own step in the commit that registers the source (the
test registers again after its rebuild, 102-113), so a step runs chemistry
once, and a rejected step reruns it from ``u0`` as the pin's loop does. The
FP32 storage emulation is removed (Cheng's decision; section 9, y1, y2,
fp1-fp3), yet with FP32 storage every in-place write into stored rows rounds: EF's
limiter repairs (``src/mesh/meshblock.cpp:907-909``, every stage with species
or scalars, else the last; 952; the slot's cons2prim, ``U'->W`` at
``src/eos/ideal_moist.cpp:185`` with a reference, else ``limit_totals_`` at 211
in ``U->W``: an EF site, its acknowledgement asked of EF), K's saturation write (972), A's closure and
the ``+=`` into stored species, as in all three callers. The
slot computes temperature and pressure with the EOS, FP64, from that
cons2prim's ``hydro_w`` totals (``W->T``, ``w[IPR]``, as
``examples/run_hydro.cpp:180-182``) and passes them to
``ChemistrySource::apply`` (K ``chemistry_source.hpp:35-37``). "No behaviour change"
is a hypothesis with one golden per caller: K rev 2.8 widens S-K4's bitwise
row (K.rst:748, run_hydro only) to the other two, the test at FP32 with its
restart (its anomaly deck's ``U'->W`` is the same call as ``peos->forward``,
``src/eos/equation_of_state.cpp:209-210``); a double step fails them. One change is named: for an
empty stoichiometry the gate skips the jupiter driver's step, including its
``peos->forward`` (line 245), whose cons2prim calls the conserved limiter on the
stored ``hydro_u`` (``src/eos/ideal_moist.cpp:209-211``), a repair kept in place, with
retry marks for the step-end read (``src/mesh/meshblock.cpp:1535``); ``check_redo``
refreshes ``hydro_w`` itself, from a clone (``src/mesh/meshblock.cpp:1511-1517``) (the test is also ungated, but its deck
is never empty); that repair is one of EF's FP32 writes (above). Cubed sphere adds
the second exchange round (``src/mesh/meshblock.cpp:651-658``) and S's corner
fill.

Within-stage exchanges at the pin. Each is a boundary (pack | host |
unpack) only when its condition holds; the single-block Cartesian default
crosses none. Lane S's exchange table (``docs/v3/design/S.rst`` section 3.3,
``jihenghu/snapy`` branch ``v3-design/S`` at ``4b96f4ea``, S.rst:324-742: rows 0-29, 16r, 16i, 17p and 16l) owns their
order, packing and rounds; X cuts one boundary per blocking round, keyed by
S's rows.

.. csv-table:: within-stage boundaries
   :header: "Site", "When", "What", "S rows"

   "src/hydro/hydro_forward.cpp:1017, src/hydro/hydro_forward.cpp:1029", "cubed sphere", "x2, x3 face states (C's B4)", "11-12"
   "src/scalar/scalar.cpp:120-146", "cubed sphere", "scalar face states", "20-21"
   "src/hydro/hydro_forward.cpp:489-503", "conversion-order 4", "primitive ghosts, then boundary functions (C's B1)", "3-4"
   "src/hydro/hydro.cpp:636, src/hydro/hydro.cpp:712", "x1 split", "reference anchor relay, serial down the column: a host segment (C's B2)", "6-7"
   "src/hydro/hydro.cpp:727-776", "x1 split", "reference ghost rows (C's B3)", "8"
   "src/hydro/hydro_forward.cpp:666, src/hydro/hydro_forward.cpp:953, src/hydro/hydro_forward.cpp:1059-1065, :1096-1097, :1131-1132", "x1 split, with the centroid switch or covariance", "x1 seam rows (C's B5)", "5, 10, 13-16"
   "src/hydro/hydro_forward.cpp:245-253, src/hydro/hydro_forward.cpp:1490", "x1 split, with scalar covariance or the seam work term", "x1 seam rows; not in C's list", "19, 22-25"
   "src/hydro/hydro_forward.cpp:1227", "positivity limiter", "hydro theta (EF); after #370 also hspec and, cubed sphere, donor velocities", "17, 17p"
   "src/scalar/scalar.cpp:205, src/scalar/scalar.cpp:239", "scalar limiter; 205 only with an upper bound", "scalar theta (EF), through src/scalar/scalar.cpp:179", "26-27"
   "src/eos/equation_of_state.cpp:340-347, from src/mesh/meshblock.cpp:909, src/mesh/meshblock.cpp:952", "pz > 1, whole-column vapor and cloud repair", "x1 column gather, a collective: splits the limiter's segment (EF); 1-2 rounds per call, and the last stage's second call (952, with reactions) adds 2 (1 with M8)", "28"
   "src/forcing/diffusion_cubed_sphere.cpp:132-143", "cubed sphere, diffusion", "flux seam, four streamed rounds inside the hydro forcing loop (src/hydro/hydro_forward.cpp:1378-1384): splits that segment", "18"
   "none yet (S.rst:521-534)", "cubed sphere, panel seams", "R's replace of the owner's x2/x3 face slab (proposed)", "16r"
   "none yet (S.rst:535-548)", "fallback, only if G-R2 fails", "the same replace on same-panel x2/x3 edges, at 16r's position: +1 round, 0 on the cubed sphere if packed into 16r's (hypothesis)", "16i"
   "none yet (S.rst:572-588)", "fallback, only if G-S-K7 fails", "a second replace after the carry, at the end of ``limit``: +1 round", "16l"
   "src/mesh/meshblock.cpp:983-990", "axial closure with a process group", "allreduce: the global phase (section 3)", "29"

Reconciled with S's table: rows 18, 16r and 29 were missing here and are
added; row 0 is the stage-0 exchange, rows 1-2 the stage-entry exchange and
row 9 the register's exchange phase (section 4). Row 30 goes with the
gravity-work fixer (its call in d, its body in gw1), and with it merge M9. X accepts S's merges as
proposals with S's labels: M1, M2, M4-M7 and M10 under the section 5 golden;
M8 only once that golden passes; M3 changes the order of the anchor scan, is
not bitwise and is Cheng's call (S.rst:1897-1901). S's sum on the cubed sphere with the
most options is 32 rounds, 17 with M1-M8, 18 with row 16r (S.rst:796-812, 899-902;
hypothesis); the last stage with reactions adds row 28's second gather (+2, +1 with M8),
and rows 16i and 16l add one each only if their gates fail. Row 16r cannot share row
17: theta reads the reconciled flux.

Every x1 seam-row call goes through ``HydroImpl::_x1_ghost_rows``
(``src/hydro/hydro.cpp:344``), which posts and then waits on a
receive (a local board for a same-process neighbour); with the column unsplit it returns at once. A tensor
that crosses one of these boundaries lives in fixed storage (lane C's rule,
adopted). The vapor check is deferred inside every segment, set once by
``BlockAdvance``: the host flag ``DeferVaporCheck``
(``src/mesh/meshblock.cpp:689-701``, ``src/mesh/meshblock.cpp:1516``)
would be baked into a capture. The species clip of user forcings
(``src/mesh/meshblock.cpp:879-884``) is EF's op, which X registers in the
block's compute segment; it posts the mass it withholds. The clones at
``src/mesh/meshblock.cpp:857`` (dry row of ``fut_hydro_du``) and ``:859``
(``fut_scalar_ds``) and ``:869`` (the clip's species base) become ``copy_``
into ``StateStore`` scratch; ``carry_dry_source`` at ``:887`` (outside w,
in no row) takes the dry difference and the scalar base unchanged. They lie
in w's 819-884 and are rewritten, not deleted: no count changes. C's B6, the x1 seam flux average, is the register's exchange phase
(section 4).

dt (amended for V3 D, B3 and C4; the lines in this paragraph are at C4
``a26cb4c4``, on jihenghu/snapy ``v3-closing/graph-ab-d`` ``9b5c7a49``, not at
the pins): a captured segment reads
the block's device dt, not the host dt. The stage's dt is a ``StepDt``
(``src/utils/step_dt.hpp``): the host double in an eager stage, and in a
captured one the executor's persistent 0-dim FP64 dt of that block, filled
in stream order before each replay (``StageExecutor::device_dt``). The
products ``w2 dt`` and ``dt_corr`` are formed from it on the device, bitwise
the host products. dt is therefore not in the ``GraphKey`` of a captured
segment, and a changed dt replays the same graphs. The original reason for
the refusal, a 0-d scalar written before replay leaving host-formed products
stale, no longer applies: no captured op forms one on the host.

The refusal now applies to reading the host dt inside a captured segment.
``StepDt::value()`` throws on a device dt (``src/utils/step_dt.cpp:14-19``),
so an op that still reads the host dt cannot run in a captured segment
unnoticed. Such an op declares ``Op::reads_host_dt``
(``src/mesh/segment.hpp:55``). Its segment captures with the host dt and
keeps dt in its key (``GraphKey::dt``, ``src/mesh/stage_executor.cpp:605``),
so a new dt recaptures it. Every other segment keys no dt.

C4: the vertical implicit solve reads the device dt. ``dt_corr`` is
``dt.scaled(w2)`` (``src/hydro/hydro_forward.cpp:1128-1131``), and the VIC
kernels read it through ``VicParams::dt_device``
(``src/implicit/vic_kernels.h:266-286``). The ``I/dt`` rows of the
radial-exact coupling use ``StepDt::divide``
(``src/implicit/implicit_hydro.cpp:402-408``), which on CUDA takes torch's
path for a host-scalar divisor, ``x * (1/dt)``, with ``1/dt`` formed in FP64
and rounded to x's dtype. The posts' ``A_f dt`` is ``StepDt``'s product. With
the x1 flux off, the x1 mass-flux w term's ``mc / dt``
(``src/forcing/x1_momentum.cpp:63-71``) forms ``1/dt`` on the device and adds
it as add's alpha (``axpy_``); B3 made the cubed sphere's x1 mass source
read the device dt. The
refusal "implicit-scheme is not 0" is dropped from ``capture_reasons``: a
case with an implicit scheme captures, and its advance segment keys no dt
(gate: ``tests/test_graph_implicit_dt.py``, graph equal to eager bitwise
with dt changing every step, FP64 and FP32, implicit-scheme 1 and 9, and
the six-panel cubed sphere in FP64 with the x1 flux on and off).

After C4, the host dt is read only:

- inside the user stage forcing op, which is eager by design and the only op
  with ``reads_host_dt``;
- in cases that run eager as a whole:

  - the chemistry step (``a chemistry step is registered``);
  - the radial-conduction split's ``begin_split``/``finish_split``, the
    Coriolis, diffusion, body/layer heating, bottom-relaxation and sponge
    forcings (``forcing is not const-gravity alone``).

The x1 mass-flux w terms read the device dt since B3 and C4; a Coriolis
spherical-polar block, their other user, runs eager anyway (geometry and
forcing).

Invalidation: any change in ``GraphKey``. Kintera tables (K rev 2.6
``ef93e92d``, K.rst:183-187, 266-285): X first calls ``prepare()`` on each
kintera module of the segment (``Op::kintera``) with the segment's device,
dtype and shape; EF's ``reset()`` does not, since it runs at construction,
before the cell shape and before ``block->to(device)`` (pin
``examples/jupiter_evap_precip_1d.cpp:209``), and eager calls build a missing
table themselves. X requires an idempotent ``prepare()`` (no reallocation when
shape, dtype and device are unchanged), so the stage-zero snapshot and failure
counts survive each mid-step warm-up. K rev 2.8 states it (K.rst:280-283 at
``f84a5c58``) and gates it (K-G10, K.rst:528: same addresses, epoch, snapshot
and counts after a repeat); a hypothesis until K-G10 runs. X then opens a ``kintera::NoTableBuild`` scope
around the segment's warm-up and capture, fills ``GraphKey::tables`` and
``CapturedGraph::tables`` from the warm-up's ``used()``, and throws if the
capture's ``used()`` differs. The modules of a segment are those its ops
name when their owners register them (``Op::kintera``), since ``used()``
returns tables, not modules. Before each replay X recomputes the options
digest on the host and compares it, the ``epoch`` of each table that
``tables_for`` returns now against the held ``ThermoTablesPtr``'s in the
key, and each module's ``alloc_epoch(m)``. This check runs inside a
``NoTableBuild`` scope that X opens, as around warm-up and capture, because
the executor owns capture (K.rst:274-275 at ``ef93e92d``), so it
compares only and never builds a table: a missing one throws. A new table or a reallocation
(``prepare`` or ``reset``) is a new key, so the graph is recaptured, as any
``GraphKey`` change is, and ``BlockInit`` registers that module's tensors
in ``StateStore`` again; a digest that differs without a new ``prepare``
throws, as K rev 2.4 words it (K.rst:255-257). Each held
graph keeps its ``ThermoTablesPtr`` copies (``CapturedGraph``), so a table
outlives every graph that references it. K-F2's conditional node is
declined for this round: X captures segments whole with
``at::cuda::CUDAGraph`` and V3 adds no node-level graph API, so ``fixed``
stays the baseline; it is revisited with parameterized graphs (X2,
section 11). The prototype restores only the
state tensors around warm-up and capture (``tests/straka_cuda_graph.py:59-60``,
``tests/straka_cuda_graph.py:69-72``). X's save list is assembled from the
registries: ``StateStore``'s state, scratch and registered storage (R's
register, kintera's failure counter and active set, ``restore_active_set``
at K.rst:301, 307), every ``RetryFlag``, the stage-zero latches and
``hydro_w`` (section 3), and the run-to-date meters of
section 3, which warm-up and capture would advance; untested (gate "flags
survive replay": this list against eager, bitwise). Owners declare scratch
names at set-up (``declare_scratch``); ``reset()`` allocates them, never a step, where
``scratch()`` only looks a name up. Warm-up and capture run with the step's dt, never dt = 0, so an
op that divides by dt stays finite. With fixed storage the per-replay input
copy (``tests/straka_cuda_graph.py:86-88``) never fires and goes.
``segment_ms()`` gives per-segment time in eager mode only: CUDA events per
segment, never per op or inside a graph (margin).

8. Cross-repository interface changes and merge order
-----------------------------------------------------

- pyharp: ``forward(int stage, Tensor u0, Tensor u1, Tensor u2, std::optional<Tensor> out = std::nullopt)``
  (``pyharp/src/integrator/integrator.hpp:86-88``). An absent ``out`` keeps
  the allocating result. Rules: ``out`` may alias ``u1``, never ``u0`` or
  ``u2``; all four share dtype and device, else it throws. Order
  (``pyharp/src/integrator/integrator.cpp:82-83``): ``add_out(out, w0*u0, w1*u1)``,
  then ``out.add_(w2*u2)`` (never ``add_(x, alpha)``, which may fuse), so the
  result is bitwise equal to the allocating ``w0*u0 + w1*u1 + w2*u2``, NaN
  sign bits included. A wrong-shaped ``out`` is resized per torch's ``out=``
  convention (torch warns unless it is empty), with no extra shape check. The dead
  ``call_average3`` path (base 6bc01d53: ``pyharp/src/integrator/integrator.cpp:6``,
  ``pyharp/src/integrator/integrator.cpp:113-122``,
  ``integrator_dispatch.{cpp,cu,hpp}``) is deleted. The binding
  (``pyharp/python/csrc/pyintegrator.cpp:50-52``) gains
  ``py::arg("out") = py::none()`` (no overload). Callers with ``out=``: the RK
  combine (``src/mesh/meshblock.cpp:901-902``,
  ``src/mesh/meshblock.cpp:921-922``) and ``carry_dry_source``
  (``src/mesh/meshblock.cpp:756``, ``src/mesh/meshblock.cpp:762``).
- kintera (K): a consume-and-reset failure counter on the device and an
  active-set snapshot/restore (section 11).
- pydisort: no interface change; section 9 items D1 and D2 rewrite two flag
  functions as tables. Its fast path reads the host
  (``pydisort/src/disort.cpp:439``, ``pydisort/src/disort.cpp:451``), so it
  is not capturable.
- UCX/commux: no change; exchanges stay host-managed.

Merge order (provider first): pyharp out=; kintera (K); snapy X1 to X7
(section 10).

Radiation: snapy calls no radiation on its step path at the pin; its only
pyharp type is ``harp::Integrator`` (``src/mesh/meshblock.cpp:257``). When
added, radiation is an eager FP64 segment on CPU or CUDA whose heating is a
registered source. Unsupported, explicitly: radiation inside a graph, and
FP32 radiation (pyharp keeps FP64 buffers,
``pyharp/src/integrator/radiation_model.cpp:27-34``).

Observation at base 6bc01d53: pyharp's ``from_yaml`` does not read ``max_redo``
(``pyharp/src/integrator/integrator.cpp:20-31``), so the default 5
(``pyharp/src/integrator/integrator.hpp:58``) always holds from a deck.

9. Line budget
--------------

Counted: the brief's baseline, snapy ``src/mesh`` and ``src/interface`` at
the pin, 4,274 lines (20% leaves at most 3,419). New files count as added
lines; moved lines count in neither lane, except the 159 that leave ``src/mesh``
for O's files (o3, o4), which X counts as deleted (RULING V3-1). ``src/utils``, the FP32 storage
code included, counts in lane O. Rows joint with S are listed apart
and not counted; pyharp and pydisort are listed apart, where touched files
do not grow.

.. csv-table:: lane X
   :header: "File", "Lines now", "Lines after", "Deletions named"

   "src/mesh/meshblock.cpp", 1919, 1217, "items below (-702)"
   "src/mesh/meshblock.hpp", 332, 250, "items below (-82)"
   "src/mesh/mesh.cpp", 566, 352, "items below (-214)"
   "src/mesh/mesh.hpp", 81, 75, "items below (-6)"
   "src/mesh/meshblock_options.cpp", 296, 190, "items below (-106)"
   "src/mesh/mesh_functions.cpp", 36, 31, "items below (-5)"
   "src/mesh/mesh_functions.hpp", 20, 17, "items below (-3)"
   "src/mesh/cycle_diagnostics.hpp", 16, 0, "o4, to O's budget.cpp (-16)"
   "src/mesh/stage_executor.hpp (new)", 0, 70, "segments, schedule, graph cache"
   "src/mesh/stage_executor.cpp (new)", 0, 150, "eager run, capture, replay, global phase"
   "src/mesh/precision_policy.hpp (new)", 0, 40, "storage, compute and cast dtypes"
   "src/interface/athena_arrays.hpp", 669, 0, "F1, after O's writer change"
   "src/interface/init_from_tensor.hpp", 199, 0, "F2, with F1"
   "src/interface/stride_iterator.hpp", 140, 0, "F3"
   "total", 4274, 2392, "cut 1882 (44.0%)"

Items. Each line: net lines, file, ranges at the pin, the replacement.
Gross and added counts are by hand; the bitwise gate is section 5's golden
run on 1 and N blocks, before and after each PR.

a. -59 (meshblock.cpp): host stage timers 711, 774-781, 808-815, 889-898,
   911-918, 924-931, 939-946, 973-980; the executor times segments.
b. -36 (meshblock.cpp): verbose prints in ``reset()`` 188-198, 250-253,
   258-261, 265-268, 272-275, 293-296, 314-318.
c. -18 (meshblock.cpp): six face blocks 161-186 become one loop (+8).
d. -21 (meshblock.cpp): the closure call 983-991 goes to the global phase,
   the fixer call 993-1004 goes with the fixer (the comment 992 stays) (hypothesis: bitwise for 1 block only if 1846-1919 is idempotent).
e. -24 (meshblock.cpp): ``initialize`` 412-433 repeats
   ``initialize_under_mesh`` 472-490 and calls it instead; 398-400 is a
   commented-out barrier; the verbose print 415-417 goes (+1).
f. -29 (meshblock.cpp): legacy helpers 1417-1445; the test at
   ``tests/test_forcing.cpp:754`` calls ``limiter_patch_hit()`` and is
   edited to read the limiter marks.
g. -10 (meshblock.cpp): precision branches 534-538, 719-720, 741-742,
   1798-1800 via the policy (+2).
k. -27 (meshblock.cpp): ``part()`` 337-377, three copied dimension blocks
   become one loop (+14).
l1. -11 (meshblock.cpp): ``max_time_step`` 576-595 calls one global-dt
    helper (MIN allreduce, cfl times 2^-redo) (+9).
l2. -10 (mesh.cpp): ``max_time_step`` 335-347 calls the same helper (+3).
m1. -19 (meshblock.cpp): one retry decision for 1 and N blocks, 1565-1601:
    stack flags, amax, one host read, MAX allreduce (+18); the slot loop
    1569 keeps a literal bound of 6, slot 8 maps to bit 6 explicitly, and only the
    cause arrays (1565, 1568, 1581, 1585-1586; ``mesh.cpp:476``) gain a seventh entry;
    ``out`` at 1568 is value-initialized, ``std::array<bool, 7> out{};``.
m2. -13 (mesh.cpp): 476-492 use it (+4).
m3. -7 (meshblock.hpp): 272-280 (+2).
n1. -36 (meshblock.cpp): ``finalize`` 1349-1371, 1399-1414 calls the
    N-block body of ``Mesh::finalize`` (+3).
n2. -3 (mesh.cpp): 519-521, folded into that body.
n3. -27 (meshblock.cpp): ``clock()`` summary 1373-1397, 569, 1658-1660 and
    ``<ctime>`` at 4; executor wall timing replaces it (+3).
n4. -3 (meshblock.hpp): 317-319.
o1. -20 (meshblock.cpp): verbose prints 504-507, 520-522, 542-544, 571-573,
    1202-1204, 1220-1223.
o2. -3 (meshblock_options.cpp): verbose print 27-29.
o3. -143 (meshblock.cpp): restart checks 1623-1653 (31) to O's ``restart.cpp``,
    cycle diagnostics 1226-1339 less gw1's 1336-1337 (112) to O's
    ``budget.cpp``; RULING V3-1: X counts the deletion, O its new files.
o4. -16 (cycle_diagnostics.hpp): the whole file, with o3, to O's ``budget.cpp``.
q. -5 (meshblock.cpp): one ``thermo()`` lookup for 964-965, 1449-1452,
   1540-1543 (+5).
r. -6 (meshblock.cpp): both ``to()`` overrides 33-42 read the policy (+4).
s1. -4 (meshblock.cpp): commented-out code 242-244, unused ``peos`` 302.
s2. -5 (mesh_functions.cpp): ``UniformMesh`` 8-10 (no caller), comments
    30-31.
s3. -3 (mesh_functions.hpp): its declaration 10-12.
t. -7 (meshblock.cpp): ``bfuncs`` size checks 96-106 become a loop (+4).
u. -14 (meshblock.cpp): one state-exchange helper for 475-487 and 1116-1128
   (+12).
w. -15 (meshblock.cpp): ``apply_extra_forcing`` 819-849 loops over the two
   keys; error texts kept (+16). X counts all of 819-884: the user forcings
   stay in the block's compute segment, after the scalar step.
y1. -7 (meshblock.cpp): ``round_storage_after_stage`` 636-642 (Cheng: the
    FP32 storage emulation is removed).
y2. -2 (meshblock.hpp): 214-215.
fp1. -18 (meshblock.cpp): the rest of the emulation, decided by Cheng:
     include 22, the call 618, ``round_storage`` 620-634, blank 635.
fp2. -7 (meshblock.hpp): 207-213.
fp3. -3 (mesh.cpp): the multi-block call 403-405 (the single-block call
     358 is in M).
gw1. -86 (meshblock.cpp): the global gravity-work fixer, removed by Cheng:
     the step-start reset 724-730, ``gravity_work_fixer_sums`` and
     ``apply_gravity_work_fixer`` 1036-1109, the redo and accept calls 1477
     and 1502, the blank line 1110 and the cycle print 1336-1337, which O's
     printer leaves out. Its call 993-1004 is in d, its ``mesh.cpp`` caller in M.
gw2. -4 (meshblock.hpp): 217-220 (the flag 221-222 is in hp).
aa1. -27 (meshblock.cpp): ``axial_closure_sums`` and ``apply_axial_closure``
     1009-1035.
aa2. -6 (meshblock.hpp): their declarations 223-228.
    The axial closure stays, as a fixer inside ``exact`` (Cheng), and moves
    to lane A's ``AxialClosureFixer`` (``src/forcing/axial_momentum.cpp``).
    One rule for both fixers: every pin line of either in lane X's files is
    X's deletion, counted once; R books none. Gravity work 124 = gw1 86 +
    d 12 (993-1004) + gw2 4 + hp 2 (221-222) + M 20 (``mesh.cpp`` 363,
    384-402); axial closure 61 = aa1 27 + aa2 6 + d 9 (983-991) + hp 1 (229)
    + M 18 (``mesh.cpp`` 364, 367-383). A books the closure's new home as an
    addition inside its +170 and drops the 51 mesh lines from its base.
bb. -5 (meshblock.cpp): the lock 9, 31, 134-135, 137; blocks are built one
    after another (``src/mesh/mesh.cpp:259-272``; hypothesis).
hp. -16 (meshblock.hpp): defer flags 221-222, 229; legacy helpers 282-294.
x. -37 (meshblock.hpp): ``\param`` blocks restating signatures, 162-167,
   171-175, 186-190, 194-199, 244-249, 253-258, 265-269, 304-309 (+8).
M. -46 (mesh.cpp): 354-361, 363-364, 367-402 into the global phase.
h1. -126 (mesh.cpp): ``BlockWorkerPool`` 6-8, 13-14, 66-227; per-block jobs
    on ATen's inter-op pool with executor-owned streams (+41; hypothesis,
    gate: CPU multi-block wall time within 5%, no nested-launch deadlock).
h2. -6 (mesh.hpp): 69-74.
v. -4 (mesh.cpp): redundant single-block branch 423-426.
z. -9 (mesh.cpp): ``check_same_variables`` 47-62 as one key-set comparison
   (+7).
i. -81 (meshblock_options.cpp): six face-parsing blocks 104-212 become one
   loop; keeps x1 periodic for "cubed" only and the null padding (+28).
j. -22 (meshblock_options.cpp): ``check_slice`` 65-86 repeats the bounds
   and order checks of ``validate_slice`` (``src/mesh/meshblock.cpp:203-230``).
sv. -28 (meshblock.cpp): ``validate_slice`` 203-230, once O's generated
    output option checks hold one copy of each check (O rev 2,
    ``22cd9512``). Its sum check repeats
    ``src/output/output_type.cpp:90-98``; its bounds and order checks repeat
    65-86 of ``meshblock_options.cpp``, so j and sv never both delete before
    then.
F1. -669 (src/interface/athena_arrays.hpp): goes once O's writer holds
    tensors (``src/output/output_type.hpp:99``). Its built includers are
    ``src/output/output_type.hpp:22`` and ``src/output/output_utils.hpp:10``,
    both rewritten by O; O ports the test ``tests/test_eos_io_core.cpp:7``. ``src/turbulence/turbulence_model.hpp:9`` names a header of another
    code base, not this one, and is not built.
F2. -199 (src/interface/init_from_tensor.hpp): included only at
    ``src/interface/athena_arrays.hpp:669``.
F3. -140 (src/interface/stride_iterator.hpp): no built includer; its four
    includers are in ``src/diagnostics``, which the build does not glob
    (``src/CMakeLists.txt:35-53``) and O deletes.
I1. -6 (pyharp/src/integrator/integrator.cpp): the dead ``call_average3``
    include 6 and comment 113-122; the ``out=`` path of section 8 adds 5.
I2. -43 (pyharp/src/integrator/integrator_dispatch.cpp): dead; its only
    caller is the comment at ``pyharp/src/integrator/integrator.cpp:121``.
I3. -30 (pyharp/src/integrator/integrator_dispatch.cu): dead.
I4. -14 (pyharp/src/integrator/integrator_dispatch.hpp): dead. The build
    globs ``integrator/*.cpp`` and ``*.cu`` (``pyharp/src/CMakeLists.txt:38``,
    ``pyharp/src/CMakeLists.txt:67``), so no build edit.
I5. -29 (pyharp/src/integrator/integrator.cpp): the weight chain 35-87
    becomes one static table from type to {w0, w1, w2} per stage, with
    "euler" an alias of rk1, the same error text and one comment on the
    exact ``1. - 2./3.`` (+21, and +3 includes); the same constants.
I6. -7 (pyharp/src/integrator/integrator.cpp): ``stop()`` 93-103 as one
    boolean return (+4).
I7. -8 (pyharp/src/integrator/integrator.hpp): the ``\param`` block 66-74
    restating the signature (+1).
D1. -51 (pydisort/src/disort.cpp): ``print_ds_flags`` 544-612, eleven
    if/else pairs, becomes a {label, flag} table and one loop (+18); the
    printed text is byte-identical.
D2. -17 (pydisort/src/disort.cpp): ``set_flags`` 49-93 becomes a map from
    token to ``int*`` into ``ds().flag`` (``prnt[0..4]`` kept) and one loop
    (+28); the same 17 tokens and the same error text.

Added counts for I5, D1 and D2 come from clang-format (Google style) on a
draft. Without h1-h2 the cut is 1750 (40.9%). If O's options schema replaces
``from_yaml``, i goes to O and j stays X's, as O books it:
with i out the cut is 1801 (42.1%), and with i, h1 and h2 out 1669 (39.1%), the lower bound.
Before O's writer change the cut is 986 (23.1%): F1 and F2 wait for it, and
sv waits for O's option checks; F3 does not. o3 and o4 (159) wait for O's
``restart.cpp`` and ``budget.cpp``; without them 827 (19.3%), 28 short of 20%. Of those 827, 614 (14.4%)
is the floor before O's changes: i (81) goes to O if O's schema replaces
``from_yaml``, and h1 (126) and h2 (6) wait on their wall-time gate. d (21) and bb (5)
also rest on hypotheses; without them 588 (13.8%). Cheng's decisions are in the base: the FP32 storage emulation
goes (y1, y2, fp1-fp3; ``fp32_storage.*`` counts in O) and the global
gravity-work fixer goes (gw1, gw2). Their Python bindings
(``python/csrc/pymesh.cpp:239-241``, ``python/csrc/pymesh.cpp:332-343``,
``python/snapy/mesh.pyi:175``), the fixer API in ``src/hydro`` and their
tests go with them, outside the lane and uncounted. ``stage_executor.cpp``
keeps its +150 as an upper bound.

.. csv-table:: joint, not counted
   :header: "File", "Lines now", "Lines after", "Owner and request"

   "src/mesh/meshblock.cpp", 1919, 1919, "S: exchange 644-685, 1135-1196 (O's 1623-1653 and 1226-1339 less 1336-1337 are X's o3)"

O moves ``src/utils/signal_handler.*`` (213 lines) into ``src/mesh``: a move,
counted in neither lane. ``src/mesh`` is its home: it takes ``MeshBlockImpl``
(``src/utils/signal_handler.hpp:45``), and only the mesh includes it
(``src/mesh/mesh.cpp:28``, ``src/mesh/meshblock.cpp:24``). With it,
``src/mesh`` and ``src/interface`` go from 4,487 lines (4,274 and the 213
moved in) to 2,605, a cut of 41.9%.

**Dependency on O**: F1 and F2 (868 lines) land after O's writer drops
``AthenaArray``, and sv (28) after O's option checks; section 10 orders them. The split was agreed with O (rev 1,
``a39e0e8d``): all of ``src/interface`` counts in X, ``src/utils`` in O.

Not counted, because each removes a capability or a public API:
pyharp ``RadiationModel`` (``radiation_model.*``, 202 lines; no binding,
test, deck or caller found, but a public class); the ``IntegratorWeight``
formatter (``pyharp/src/integrator/integrator_formatter.hpp:12-24``; no user
found).

Provider side: pyharp ``src/integrator`` 564 to 427 (I1-I7), pydisort
``src`` 1,590 to 1,522 (D1, D2); no touched file grows.

.. csv-table:: touched outside the lane
   :header: "File", "Lines now", "Lines after", "Change"

   "pyharp/src/integrator/integrator.cpp", 136, 94, "items below (-42)"
   "pyharp/src/integrator/integrator.hpp", 102, 94, "items below (-8); out= fits line 88"
   "pyharp/src/integrator/integrator_dispatch.cpp", 43, 0, "items below (-43)"
   "pyharp/src/integrator/integrator_dispatch.cu", 30, 0, "items below (-30)"
   "pyharp/src/integrator/integrator_dispatch.hpp", 14, 0, "items below (-14)"
   "pydisort/src/disort.cpp", 614, 546, "items below (-68)"
   "pyharp/python/csrc/pyintegrator.cpp", 57, 57, "py::arg out (+1); include 3 goes (integrator.hpp has it)"

10. Migration
-------------

.. csv-table::
   :header: "PR", "Content", "Against main"

   "P1 pyharp", "forward(..., out); delete dispatch; weight table, stop(), doc (I1-I7)", "bitwise"
   "P2 pydisort", "flag parsing and printing as tables (D1, D2)", "bitwise; same printed text"
   "K", "device failure counter, active-set snapshot", "K's call"
   "X1", "PrecisionPolicy; lane's fp32() branches; u0/s0 in storage dtype", "bitwise"
   "X2", "StateStore; out= replaces set_ at 901-902, 921-922", "bitwise (hypothesis; gate: section 5 golden)"
   "X3", "StageExecutor from tests/straka_cuda_graph.py; deck selection; eager default", "eager bitwise; graph parity"
   "X4", "device closure; global phase for 1 and N blocks; the no-source baseline of the section 5 chemistry-slot gate (strictly bitwise, a failure blocks X4; its source run lands with S-K4, K rev 3.1)", "bitwise for accepted steps if partial-sum order is kept (hypothesis), else round-off"
   "X5", "meshblock split into six parts; section 9 src/mesh items except d, M, h1-h2, sv", "bitwise (hypothesis; gate: section 5 golden)"
   "X6", "h1-h2: block jobs on the inter-op pool", "bitwise; CPU multi-block wall time within 5%"
   "X7", "capability matrix, docs, benchmarks (G7)", "no numerics"
   "X8", "delete src/interface (F1-F3) and sv; F1-F2 after O's writer holds tensors (cut 986 of 4,274 before, 827 without o3-o4, 1,882 after), sv after O's option checks, F3 any time; first O ports ``tests/test_user_output.cpp:39-100`` (O's note 8), and one grep for the array class, ``InitFromTensor``, ``CopyFromTensor``, ``OutputData`` and ``pfirst_data_`` over src, tests and python finds only unbuilt files", "no numerics; build and tests pass"

Scope of this round (Cheng): design and declarations only. Nothing here was
built or run; section 5's gates are specified, not run, and a claim that
needs a run is marked hypothesis. Each contract has a milestone: the stage
executor (section 7) is G1 (``V3_ROADMAP.md:155``), its multi-block global
phase G4 (``V3_ROADMAP.md:235``); pyharp's output-buffer RK interface
(section 8) is G1 (``V3_ROADMAP.md:162``); the capability matrix is listed
in G0 (``V3_ROADMAP.md:144``) and documented in G7
(``V3_ROADMAP.md:296``). G0's baselines and capture probes
(``V3_ROADMAP.md:146-149``) are implementation, not part of this design.

Removed: host stage timers (replaced by segment timing), the ``defer_*``
flags, the four legacy helpers, the dead pyharp dispatch. No physics option
and no default changes. Behavior changes: under ``cuda-graph``, a change to
``implicit-fallback-fp64`` after capture takes effect only through
``bump_config_generation()`` (section 6, ``kImplicit`` row); and the FP64
retry (Cheng, 2026-11-17) re-solves the failed columns only. The behaviour
before the change re-solved the whole block in FP64 (``src/hydro/hydro.cpp:527-535``);
now the block's other columns keep their FP32 solution, a change beyond
round-off of the size of the FP32-FP64 solve gap, the ``fp64_fallbacks``
meter counts columns, and a step with no failed column is bitwise (I section 10 (c)).

Deck (G7; names are Cheng's call, ``V3_ROADMAP.md:301``)::

  integration:
    execution: eager        # eager | cuda-graph
    execution-strict: false # true: reject a case with any eager segment

Capability matrix (numerical support is the eager columns; graph support is
separate):

.. csv-table::
   :header: "Case", "eager FP64", "eager FP32", "graph FP64", "graph FP32"

   "Cartesian dry, 1 block", "yes", "yes", "G1", "G1"
   "Cartesian, N blocks, UCX ranks", "yes", "yes", "G4", "G4"
   "kintera moist", "yes", "M-series", "G2", "G2"
   "vertical implicit correction", "yes", "Cartesian ideal-moist, FP64 retry of the failed columns only (Cheng)", "G3", "G3"
   "cubed sphere", "yes", "G5", "G6", "G6"
   "user TorchScript forcing", "yes", "yes", "eager segment", "eager segment"
   "radiation (pyharp/pydisort)", "not on the step path", "no", "no", "no"

Every row keeps its schemes (Cheng): all six Riemann solvers, all six
reconstructions, ANEOS and PnetCDF stay. CPU and MPS run eager only:
``cuda-graph`` falls back to eager there, and strict mode rejects it.

After V3 (note): two-tableau stage form (IMEX-ARK), not in V3 (Cheng)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No other lane writes against this subsection; the contract, skeleton and
budget above stand without it. Today a stage is the Shu-Osher combine
``w0 u0 + w1 u1 + w2 dt L(u1)``, and the vertical implicit correction runs
inside it with dt scaled by the stage's ``w2``
(``src/hydro/hydro_forward.cpp:1568-1572``, stage published at
``src/mesh/meshblock.cpp:703-705``): one linearized solve per stage, not an
additive Runge-Kutta method, so its order in the stiff part is not
established. An ARK pair would add, in prose only (not in the skeleton):

- ``harp::StageTableau`` {``aex``, ``aim``, ``bex``, ``bim``, ``c``,
  ``order``}: ``aex`` strictly lower, ``aim`` lower with diagonal
  ``gamma_i``, host doubles fixed for the run;
- ``StateStore::ark_stages()``: ``kex[i]``, ``kim[i]``, 2s state buffers
  allocated at ``reset()`` in the storage dtype (FP64 if the order gate
  fails in FP32);
- stage i: ``rhs = u0 + dt sum_{j<i} (aex_ij kex_j + aim_ij kim_j)``; the
  implicit owner solves ``u_i = rhs + gamma_i dt F_I(u_i)``
  (``dt_corr = gamma_i dt``); then ``kex_i = F_E(u_i)`` and
  ``kim_i = (u_i - rhs) / (gamma_i dt)``; step end
  ``u = u0 + dt sum_i (bex_i kex_i + bim_i kim_i)``;
- candidates: ARS(2,2,2), ARS(4,4,3) (Ascher, Ruuth and Spiteri 1997),
  ARK3(2)4L[2]SA (Kennedy and Carpenter 2003); with one linearized solve per
  stage the method is linearly implicit, so keeping the design order needs
  Newton iterations or a tableau meeting the W-method conditions;
- gates: rk3 as a tableau with ``aim = 0`` equals Shu-Osher to 1e-14
  relative in FP64; the observed order on a one-column acoustic-gravity
  wave (dt to dt/4 against dt/16, implicit Courant 10-100) is at least
  p - 0.2;
- no milestone in V3; after V3 it would follow G1, which it needs for the
  executor and fixed storage.

Three points stand in its way at the pin. The pinned correction is not
split into explicit and implicit operators. R's register carries one weight
per stage (``begin_stage(int stage, harp::IntegratorWeight const&)``), not
separate explicit and implicit weights. Positivity relies on the SSP
structure of the integrators (``src/hydro/hydro_forward.cpp:1163-1164``),
which an ARK pair need not have.

Cheng ruled (2026-11-17, relayed by the coordinator): not in V3. The implicit
operator is not split, there is no Newton option, SSP positivity stays, and
stages stay Shu-Osher with pyharp's weights.

11. Open questions to other lanes
---------------------------------

R (Linfeng):

- Answered by R rev 2.2 (``4c59162a``): FP64 step field (FP32 on MPS); one
  ``IntegratorWeight`` per ``begin_stage``; corrections per stage, plus
  ``after_combine``; ``rollback()`` in the reset list; the monitor's flags
  join the read at ``src/mesh/meshblock.cpp:1566``; the step is zeroed once,
  in ``begin_step()`` at X's step start (R.rst:31-32).
- Answered by R rev 4.3 (``97c62312``): ``HydroStage::after_combine`` is gone;
  X's ``StageExecutor::after_combine(stage, dt)`` is the one runner (Round-18 notes).

S (Jiheng):

- Closed by S rev 2.3 (``bee1b276``): every exchange is pack, host wait, unpack
  per round, corners filled in the last round's unpack (S item 1); the
  boundary-function refresh ``src/mesh/meshblock.cpp:1135-1196`` is S's,
  under BlockExchange (S rev 2.1 item 2).
- Answers to S's three questions (S.rst:1934-1941 at ``4b96f4ea``): (1) yes,
  cuts at S's rows, merges as labelled (section 7); (2) the options YAML
  parsing (S's move A) is not X's: X keeps no options reader, and the
  generated option schema is O's (section 9, j and sv); X asks O; (3) yes,
  X owns the CI build for S's H18 (``--fmad=false``, ``-ffp-contract=off``)
  as a capability-matrix row (G0, ``V3_ROADMAP.md:144``). S's open point: if
  their gates fail, 16i shares 16r's cut when packed into its round, else
  has its own; 16l is always its own cut, as it reads the post-carry flux.

EF (Zoey):

- Limiter marks and vapor failures as device counters reset by the step-reset
  list (``src/mesh/meshblock.cpp:735-738``)? Answered: EF note 6 below.
- Can the retry test avoid the full-field clone at
  ``src/mesh/meshblock.cpp:1517``? Answered: EF note 7 below.

K:

- A consume-and-reset device counter to replace the host drain
  (``src/mesh/meshblock.cpp:1544-1546``, ``src/mesh/meshblock.cpp:1453``)?
- An active-set snapshot/restore so a retry repeats the warm start
  (``src/mesh/meshblock.cpp:969-970``)?
- Output buffers for saturation adjustment, to make G2 capturable?

O (Jiachen), answers to O rev 1 (``7dac5f4a``, section 14, X1-X3):

- X1, yes: the dt choice and the diagnostic reads sit outside captured
  segments (sections 3 and 7); output, flushes and restart writes run in the
  host phase after the global accept, as ``V3_ROADMAP.md:101`` requires.
- X2, not in the first release: dt and the stage weights stay host doubles
  (section 6), dt is chosen at step start, and a changed dt recaptures (section 7, ``V3_ROADMAP.md:124-129``).
  Post-step consumers run after the accept, outside graphs, so no dt is
  baked in. A device dt scalar comes with parameterized graphs; X will share
  that buffer with O's ``observe`` then. (Amended: captured segments now read
  a per-block device dt, V3 D and C4, section 7; the post-step consumers still
  take the host dt.)
- X3, X's executor: after ``read_restart`` the Mesh calls
  ``StageExecutor::invalidate()`` (skeleton ``stage_executor.hpp``), which
  drops graphs and captured workspaces; restart never serializes them
  (``V3_ROADMAP.md:80``). MeshBlock only loads into the fixed buffers.

Requests to O, and where the two drafts differ:

- Joint rows, agreed with O (``a39e0e8d``): all of ``src/interface``
  (1,008) counts in X, and X deletes it after O's writer drops
  ``AthenaArray`` (``src/output/output_type.hpp:99``); ``src/utils``, the FP32
  storage code included, counts in O. Moved lines (``signal_handler.*`` into
  ``src/mesh``) count in neither lane; restart and diagnostics out of
  ``src/mesh`` count as X's deletion (RULING V3-1, below).
- Take restart and cycle diagnostics, with section 9's ranges: the restart
  checks ``src/mesh/meshblock.cpp:1623-1653`` (the rest of 1603-1833,
  n3's 1658-1660 and g's 1798-1800 included, stays in ``src/mesh``),
  and diagnostics ``src/mesh/meshblock.cpp:1226-1339`` less
  gw1's 1336-1337, with ``src/mesh/cycle_diagnostics.hpp``. X asks that
  restart load into the fixed buffers in the storage dtype and that the
  diagnostics' host reads stay at step end.
- O rev 2 (``22cd9512``): the restart checks 1623-1653 (a move into O's
  restart module) and the cycle diagnostics 1226-1339 with
  ``src/mesh/cycle_diagnostics.hpp`` (O's printer over budget totals) are
  inside the joint rows of section 9, so X counts neither, except the
  fixer's print 1336-1337 (gw1). Superseded (RULING V3-1, agreed by O at
  ``36b61c40``): the 159 lines leave ``src/mesh`` for O's ``restart.cpp`` and
  ``budget.cpp``; X counts the deletion (o3, o4), O only its new files.
- A device dt at ``src/mesh/meshblock.cpp:1210``: not in the first release.
  dt stays a host double and a graph is keyed by it (section 7); output runs
  in the host phase after the accept, outside graphs, so ``AccumulateStats``
  takes the host dt. A device dt comes with parameterized graphs (X2 above;
  lane A's note 14 agrees).
- Jiachen accepted these rows, the totals of section 9 and the refusal of
  the device dt as final; O drops its device-dt request.
- Slice validation has three sites: ``src/output/output_type.cpp:90-98``
  (slice and sum), ``src/mesh/meshblock.cpp:203-230`` (all checks) and
  ``src/mesh/meshblock_options.cpp:65-86`` (bounds and order). O asks the
  owner of ``meshblock.cpp``, which is X, to drop 203-230 once O's generated
  option checks hold them; X deletes it as item sv (O rev 2). Item j
  drops the third now, and j is X's, as O books it.

A (Jiachen), on A's notes 14-17:

- 14: on one rank the closure's host segment is absent (section 3).
- 15: the closure split is in items aa1 and aa2: X deletes 1009-1035 from
  ``src/mesh`` and A books the addition; please drop the 51 mesh lines from
  A's base.
- 16: the user forcings stay in the block's compute segment, where X
  schedules them after the scalar step; X counts
  ``src/mesh/meshblock.cpp:819-884``, A counts none of it.
- 17: the stage-local FP64 source accumulator is a row of section 6.

C:

- Rev 2 is to specify the register, FP32 and graph contracts: please
  reference X's precision and graph contracts (sections 6 and 7) rather
  than define a second one.

Lane I, the six questions of I.rst:1277-1284 (``7604ada0``):

1. A predicated second FP64 launch, always run, not CUDA conditional nodes
   (X's position): the launch count stays fixed and eager, the reference,
   runs the same ops as the replay (section 3); section 6's row.
2. ``kImplicit`` is already a row (section 6; ``precision_policy.hpp:21``);
   the FP64 retry is that row forced FP64. Posts stay ``kFlux``, the stage
   total, whose dtype a retry must not change.
3. Amended (C4, section 7): the solve reads the block's device dt.
   ``dt_corr`` is formed on the device and the VIC kernels read it, so dt is
   not in the ``GraphKey`` of its segment and a dt change replays. The
   original answer was no: dt was a host double keyed in ``GraphKey``, a
   change recaptured, and a device dt was refused.
4. Yes, with no read of its own: the OR of ``col_status`` goes through R's
   int32 ``retry_flags()`` slot into X's one stacked step-end read
   (``src/mesh/meshblock.cpp:1566``) as cause bit 5 (section 4); G3 also
   removes the screen at ``src/implicit/implicit_hydro.cpp:476-487`` (section 7).
5. X's position: the solve has no boundary of its own. It runs in the hydro
   forward's compute segment, which starts after the stage's unpack and ends
   at the next exchange (S's seams, the register's), and until G3 at the
   finiteness screen (section 7); a mask or grid change bumps
   ``config_generation`` and recaptures.
6. Yes in V3 (X's position): the pin refuses FP32 elsewhere
   (``src/hydro/hydro.cpp:502-507``); lifting it waits for G5 (FP32 cubed
   sphere, section 10) and a lane I FP32 gate.

Round-2 notes
~~~~~~~~~~~~~

EF (Zoey), two questions on rev 1:

1. Can the positivity limiter's theta syncs be one exchange, at one segment
   boundary per stage? Yes when the scalar has no upper bound, after a
   reorder; two exchanges with one. At the pin there are up to three syncs,
   in order: the hydro theta (``src/hydro/hydro_forward.cpp:1227``) inside
   the hydro forward (``src/mesh/meshblock.cpp:773``), then, inside the
   scalar forward (``src/mesh/meshblock.cpp:785``), the complement theta
   when the upper bound is set (``src/scalar/scalar.cpp:205``) and the
   scalar theta (``src/scalar/scalar.cpp:239``), both through
   ``src/scalar/scalar.cpp:179``. The hydro and scalar thetas do not depend
   on each other: the hydro scale touches only the species rows
   (``src/hydro/hydro_forward.cpp:1167-1169``), and the scalar reads the
   hydro's dry-mass row (``src/scalar/scalar.cpp:99``,
   ``src/scalar/scalar.cpp:165-170``). So if the scalar computes its fluxes
   and thetas before the hydro sync, one exchange carries both (and the
   complement theta). The scalar theta at 239 reads the fluxes that the
   complement scale changed (``src/scalar/scalar.cpp:210-212``), so with an
   upper bound it needs a second exchange. The reorder is EF's and S's
   (exchange table); X schedules whichever they declare.
2. Where do #370's extra syncs go? Into the hydro theta boundary. #370
   (upstream ``730b3500``, merged after the pin) syncs the species enthalpy
   with the same call as theta and, on the cubed sphere only, exchanges the
   donor velocities as face states (cross-panel only, no interpolation),
   all between the theta and the carry. Both need only ``w``, so they post
   with theta. The face-state temporaries are allocated per call
   (``zeros_like(w)``), and anything that crosses a boundary lives in fixed
   storage, so the executor's workspace holds them.

EF (Zoey), notes 1-16 on rev 1 (``77c65d5a``):

1. Agree; no change.
2. Agree. Section 9 is recounted with Cheng's deletions: fp1-fp3, y1-y2,
   gw1-gw2 (fp1 and gw1 with the blank lines 635 and 1110, as Jiachen
   counts), aa1-aa2, d at -21, and sv (O rev 2). The cut is 1,723 of 4,274
   (40.3%), or 827 (19.3%) before O's writer change, 614 (14.4%) of it before O's changes; the dependency is stated there and in section 10. The
   FP32 variant is gone; the bindings are named as uncounted.
3. Agree; all four are rows of the section 7 table, the vapor repair's
   x1 gather included.
4. Agree; the hydro theta row carries #370's two syncs (and the paragraph
   after the notes).
5. Agree: ``StepReset`` resets in place only, its tensors are allocated at
   setup (skeleton ``meshblock_parts.hpp``); its ``commit`` and ``drop``
   hooks are dropped in rev 3.4 (section 3).
6. Agree; the gate is a row of section 5 ("flags survive replay").
7. Agree: ``StateStore::scratch()``, filled with ``copy_`` (skeleton
   ``state_store.hpp``).
8. Agree: ``RetryFlag`` {name, owner, cause, 0-d device tensor}, stacked in
   one fixed dtype (skeleton ``meshblock_parts.hpp``).
9. Agree; sections 2, 5, 7, 9 and 10 now have the closure only, and
   each ``Fixer::apply`` posts a named correction.
10. Agree: the anomaly-floor exception and a diagnostics row are in
    section 6 and ``precision_policy.hpp``.
11. Agree in substance: section 6 states the MPS dtypes, and a line under
    the capability matrix says CPU and MPS run eager, with strict mode
    rejecting ``cuda-graph``. A column would repeat "eager" in every row.
12. Agree: deferred mode is the only mode inside segments (section 7).
13. Agree: the species clip is EF's op, registered by X in its segment; the limiter at 952 is in the
    segment list, ``carry_dry_source`` takes ``out=`` (section 8), and the
    three clones become fixed scratch (section 7).
14. Agree; item f edits that test.
15. Agree; all three are fixed.
16. Agree; no change.

The pin ``9e5de2d4`` predates #370; ``730b3500`` is its direct child. #370
changes no file of lane X. On a rebase past it,
``src/hydro/hydro_forward.cpp`` lines after 1238 move down 49
(``src/hydro/hydro_forward.cpp:1227`` becomes 1229, the scale at 1237
becomes 1286); the face-state
exchanges X cites at 1017 and 1029 do not move. Nothing in section 9
changes.

Counts. Rev 2 takes Jiachen's final joint rows: +28 for the FP32 emulation
(with the blank line 635), +83 for the gravity-work fixer and +29 for the
closure (aa1, aa2). The fixer's 83 includes the cycle print 1336-1337, which
Zoey's +81 leaves in O's printer; the comment at 1476 stays and is edited.
Item d is -21, not rev 1's -22: the comment at 992 belongs to the boundary
call after 1005 and stays. So X cuts 1,695 before sv, one line below
Jiachen's 1,696, and 1,723 (40.3%) with sv. Lane R books none of the fixer
lines: X deletes the 124 gravity-work lines (98 of ``meshblock.cpp``, 6 of
``meshblock.hpp``, 20 of ``mesh.cpp``; gw1, gw2, d, hp, M) and the 61
axial-closure lines once each; R counts its ``src/hydro`` lines only.

Round-3 notes (rev-2 review)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The reviewer's notes 1-7 on rev 2 (``034ef405``):

1. Resolved; no change.
2. Split: ``kStepStart`` runs before the stage-0 resets, ``kRollback``
   erases no step-start post, the retry reruns step start, and
   ``finish_step`` is named (sections 3 and 4; ``meshblock_parts.hpp``,
   ``stage_executor.hpp``).
3. Added: the ``after_combine`` and ``observe`` hooks (sections 3 and 4);
   rev 3.1 gives the amount sum to O and has ``StateStore`` register
   R's tensors (round-4 notes).
4. Re-cited to R (now rev 2.2 ``4c59162a``): one ``IntegratorWeight`` per
   ``begin_stage``, no stage table; the accumulator is FP64, FP32 on
   MPS (sections 4 and 6).
5. Narrowed to the prototype's state restore; warm-up and capture use the
   step's dt (section 7).
6. One owner per line: O takes only the restart checks 1623-1653 (O head
   ``c12f78bb``), so n3's 1658-1660 and g's 1798-1800 stay X's (section 9);
   the totals 1,723 and 827 do not change.
7. Moved to the optional subsection at the end of section 10, with these
   three points.

The coordinator's items 1-6:

1. As note 2 above.
2. As note 3 above.
3. ``GlobalCorrection`` holds R's ``Fixer`` list only; ``apply`` gets that
   stage's ``w2`` (section 4).
4. ``diffusion.radial`` is cause bit 6, value 64, reserved in V3
   (Round-8 notes; section 3; ``meshblock_parts.hpp``).
5. Section 9's ranges in the joint row and the O block, and j is X's
   (sections 9 and 11); X registers the species clip, EF owns the op
   (section 7).
6. X cites R rev 2.2 ``4c59162a`` (section 4).

Round-4 notes
~~~~~~~~~~~~~

K rev 2 (``fa290bb2``), its three asks to X (K.rst:621-628):

1. Done: ``GraphKey::tables`` holds the (``options_digest``, ``epoch``)
   pairs and ``CapturedGraph`` the ``ThermoTablesPtr`` copies (section 7).
2. Done in K's order (sections 3 and 4; ``meshblock_parts.hpp``).
3. K-F2's conditional node: declined for this round (section 7).

R rev 2.2, its requests to X (R.rst:1612-1614): ``begin_step`` is X's
name (section 4); ``after_combine`` follows each stage's combine (sections
3 and 4); the amount SUM owner is O (section 4); time is per segment, not
per op (section 7); X counts the meshblock fixer
lines: item d deletes the call, gw1 the body (section 9, rows d and gw1).

The rev-3 review (``100bc9b3``), conflicts 1-4: (1) ``after_combine``
after every combine, the limiter after ``finish_step`` (sections 3 and 4);
(2) cited to ``4c59162a``, ``begin_step`` (sections 4 and 11); (3) R's
``reset`` allocates, ``StateStore`` registers (section 4,
``state_store.hpp``); (4) O owns the amount SUM (section 4). Conflict 5,
the clone sites at ``src/mesh/meshblock.cpp:857`` and ``:859``, and the
``diffusion.radial`` details are taken in the next revision.

Round-5 notes
~~~~~~~~~~~~~

The coordinator's items 1-5 on rev 3.1 (``bac201d1``):

1. Clone sites ``src/mesh/meshblock.cpp:857`` and ``:859``: ``StateStore``
   scratch, inside w's range; totals unchanged (section 7).
2. ``diffusion.radial``: host read, slot 8 versus bit 6, ``max_redo``,
   zeroed in ``kStepStart`` (section 3; reserved in V3, Round-8 notes).
3. Rollback order, stack dtype and cause map (section
   3); save list and its gate (sections 5 and 7).
4. 827 (19.3%) stated, 614 (14.4%) before O's changes, 588 (13.8%) without
   the hypothesis items d and bb (section 9); skeleton O ranges at ``c12f78bb`` (``meshblock_parts.hpp``).
5. Note 4 of round 3 no longer names the removed type; R is cited at
   ``4c59162a`` only (section 4).

R's asks (R.rst:1612-1614): (a) per-segment time, eager only (section 7); (b) X
counts the meshblock fixer lines: gravity work 124 and axial closure 61, on
one rule (section 9, after aa2).

Lane S's note-11 edge table is answered in the S thread, not here.

Round-6 notes
~~~~~~~~~~~~~

K rev 2.3 (``42d81bb3``), its asks to X (K.rst:728-733):
(a) ``GraphKey::alloc_epochs`` holds each kintera module's
``alloc_epoch(m)`` next to the (digest, epoch) pairs (section 3,
``stage_executor.hpp``); (b) both ``tables`` fields come from
``NoTableBuild::used()`` (section 7); (c) the chemistry step is an eager
FP64 segment after the saturation adjustment (section 7).

Round-7 notes
~~~~~~~~~~~~~

The reviewer's points 1-5 on rev 3.2 (``66f797a2``):

1. The slot loop at ``src/mesh/meshblock.cpp:1569`` keeps a literal bound 6; slot 8
   maps to bit 6 by an explicit line; only the cause arrays, with
   ``src/mesh/mesh.cpp:476``, gain a seventh entry (section 3; item m1).
2. O at ``c12f78bb`` claims 1226-1339 less 1336-1337 (gw1) only; 1207-1224 is X's, and o1 cuts
   1220-1223 (``meshblock_parts.hpp``).
3. (Radial part superseded by Round-8 note 2: lane A owns the file and
   abort stays the default.) Scratch is
   allocated at ``reset()`` and the save list comes from the registries
   (section 7).
4. Chemistry and K's ``NoTableBuild`` and ``alloc_epoch`` requests are in
   section 7 and the Round-6 notes; ``BlockInit`` calls
   ``register_storage``, ``set_prescribed`` bumps ``config_generation``, the
   int32 flags map to cause bits (section 4); the limiter at 952 opens the
   saturation segment (section 7); stage zero resets 722-738's latches,
   never the run-to-date meters (section 3).
5. Time is per segment, ``StageExecutor::segment_ms()`` (section 7,
   ``stage_executor.hpp``).

K items (``42d81bb3``): (B1) K.rst:255-256 throws on any mismatch; X
recaptures on a ``GraphKey`` change and asks K to word it so (section 7,
a conflict); (B2) ``restore_active_set`` is K.rst:301, 307; (B3) a segment's
modules come from its ops' ``Op::kintera``, and each epoch is read from
``tables_for`` at replay against the held pointers (section 7).

Round-8 notes
~~~~~~~~~~~~~

The review of rev 3.3 (``0fe47c6f``), points 1-5:

1. The widened ``out`` is value-initialized, ``std::array<bool, 7> out{};``
   (``src/mesh/meshblock.cpp:1568-1569``; section 3, item m1, ``meshblock_parts.hpp``).
2. Lane A owns ``src/forcing/diffusion_radial.cpp`` and keeps abort; X drops
   its booking (section 3). No count changes: X never counted that file.
3. EF's names ``ef.floor_patch``, ``ef.nan``, ``ef.vapor_fail``,
   ``ef.heat_capacity`` (section 4); the meters as coordinator note 1 below.
4. ``StateStore::reset()``, ``declare_scratch``, ``storage(name)`` and the
   kintera re-registration on ``alloc_epoch`` (``state_store.hpp``; sections 4
   and 7); the save list adds the stage-zero latches and ``hydro_w`` (section 7).
5. The chemistry segment runs with the kinetics module (placement and
   gate corrected in Round-9 note 2); the pre-replay check never builds a
   table; ``StageExecutor::bump_config_generation()`` is the one owner
   (sections 4 and 7, ``stage_executor.hpp``).

The coordinator's points 1-4 on rev 3.4:

1. Ruling V3-2: the meters as at the pin before R-1, accepted steps only
   in R's monitor from R-1; ``StepReset`` loses ``commit``/``drop``, whose
   one pin client, the gravity-work fixer, goes with gw1 (section 3).
2. The radial file, change and count are lane A's; X keeps the reserved
   slot and a "no change in V3" row, with no radial retry path or gate
   (sections 3 and 5).
3. EF's four names at bits 2, 3, 2, 2 (section 4). EF's order F1, solid
   fill, F1b, K, fixers (Q-X4) is the pin's last stage: limiter
   ``src/mesh/meshblock.cpp:907-909``, solid fill 936-938, limiter 952,
   saturation 953-981, fixers 983-1004 (section 7); confirmed.
4. One rule for both fixers, 124 and 61 lines, each deleted once in X
   (section 9); the cut stays 1,723.

The maintainer's decision (X thread, 13:41 PT): keep the abort for V3;
``diffusion.radial`` (bit 6) is a reserved slot, written by nobody in V3
(section 3). EF drops the interim census hook (EF.rst:671 at ``f5c8e9a7``), and A at
``76904ce8`` reserves the cause and does not call ``flag_retry`` for it in V3.

Round-9 notes
~~~~~~~~~~~~~

The review of rev 3.4 (``513aeebb``), the coordinator and EF:

1. The radial host check at step start and in ``finish_step`` each ends a
   capture segment (A.rst:550-555 at ``4e0827b8``; sections 3 and 7).
2. Chemistry keeps the pin's order: after the last stage's boundaries,
   before the step-end read, as in both drivers that step it,
   ``examples/run_hydro.cpp`` (gate at 177; ``src/run_hydro.cpp`` does not
   exist) and ``examples/jupiter_evap_precip_1d.cpp`` (no gate). X gates on
   nonempty stoichiometry, which differs from the second driver for an empty
   stoichiometry alone; no behaviour change for real decks (section 7). This agrees with K.rst:342-356, 454-457 at
   ``d83150a8``; K.rst:458-459, "K in the last ``after_combine``", conflicts
   unless it means the saturation adjustment only.
3. ``scalar_r`` is fixed storage written in place each stage (section 3; section 7, X.rst:318 at ``8bc22415``).
4. R books none of the fixer lines; X deletes 124 and 61 once (section 11).
5. EF's two questions are marked answered by notes 6 and 7; EF is cited at
   ``5e30b31b`` (EF.rst:672-678, 2065-2079; section 4).
6. K rev 2.5 (``d83150a8``): ``alloc_epoch(m)`` is K's free function, not a
   member, and X opens the ``NoTableBuild`` scope (sections 3 and 7).

Round-10 notes
~~~~~~~~~~~~~~

The review of rev 3.5 (``8bc22415``), points 1-3, and K rev 2.6 (4):

1. ``scalar_r``: its rebinds at 631, 923, 1199, 1484, 1743 and 1828 are rows of the
   section 7 table and its address is in the section 5 gate; 923 divides
   after the EF limiter (907-909) and before the solid fill (936-938), and
   631 goes with the emulation (fp1). No count changes.
2. Chemistry: the driver registers snapy's ``ChemistrySource`` (S-K4) in the
   executor's slot, ``set_chemistry``; S-K4 removes each of the three
   callers' own step in the same commit; drivers that register none gain
   nothing; no emulation rounding is left to order. "No behaviour change"
   rests on one golden per caller; the empty-stoichiometry ``hydro_w``
   change is named (section 7).
3. A revision 4 (``4e0827b8``) keeps the abort: no radial cause path (section 3).
4. K rev 2.6 (``ef93e92d``) withdraws ``prepare()`` from EF's ``reset()``: X
   calls it per segment module before warm-up (section 7; skeleton
   ``run_stage``). X cites K at ``ef93e92d``; K.rst:477-480 now states the pin
   order (Round-9 note 2); K's skeleton comment (``chemistry_source.hpp:14``,
   "last after_combine") is K's to align.

Round-11 notes
~~~~~~~~~~~~~~

The review of rev 3.6 (``fc3b002e``), points 1-6, and R-8 (7):

1. X owns an executor slot, ``StageExecutor::after_combine``: EF's limiter
   (907-909), X's scalar slot (the scalar combine and the ``scalar_r`` divide,
   ``out=``, 920-923), S's solid fill (936-938); A's closure is the global phase. R's order (R.rst:691-701 at
   ``2b4c5775``) lists 920-923 there; only its comment, ``hydro_stage.hpp:67-70``,
   omits it, so R needs no new declaration.
2. The jupiter driver's ``peos->forward`` calls the limiter (section 7).
3. The test's ``U'->W`` is ``peos->forward``'s call; the check stays (section 7).
4. ``MeshBlock::set_chemistry``, S-K4's forwarder, to ``StageExecutor::set_chemistry``.
5. K does not say whether a repeated ``prepare()`` keeps storage; X requires it
   idempotent, a hypothesis until tested; K rev 2.8 to state it (section 7).
6. ``StateStore::assign`` takes ``scalar_r``: ``initialize`` copies the given
   one (seeded ``scalar_s``, pin 530); a restart its stored copy (1743) or divides (1828).
7. R-8 (lead's addition): the radial ``post_split`` (715) precedes the boundaries and
   exchange (716-717), which precede the ``u0`` save (721); section 3 now says so.

Round-12 notes
~~~~~~~~~~~~~~

The review of rev 3.7 (``b4d18131``), points 1-5, and EF's check (6):

1. The ``+=`` is not the only FP32 rounding: every in-place write into FP32
   storage rounds, among them EF's limiter repairs (``src/mesh/meshblock.cpp:907-909``,
   952, the slot's cons2prim) and K's saturation write (972). The chemistry slot computes temperature and pressure with the
   EOS, FP64, from ``hydro_w`` totals, as the pin's callers do
   (``examples/run_hydro.cpp:180-182``), for ``ChemistrySource::apply``
   (K ``chemistry_source.hpp:35-37`` at ``143fedeb``); section 7.
2. The last stage's forward call runs the slot (Round-13 note 1 gives where),
   so a driver calls nothing extra
   before ``check_redo``; K rev 2.8 widens S-K4 to all three pin callers.
3. One caller: ``StageExecutor::after_combine(stage, dt)``, after the combine
   (last stage: after X's ``finish_step``), runs per block the
   ``RegisterPhase::after_combine`` ops in R's order (R.rst:691-701): EF's
   limiter (907-909), X's scalar slot (920-923), S's solid fill (936-938), on
   the last stage with reactions EF's 952 and K's 954-972; then A's closure
   (983-990) as the global phase after every block, R's several-block order.
   R's ``HydroStage::after_combine(u, reg)`` (``hydro_stage.hpp:67-70`` at
   ``2b4c5775``) has no stage or scalars, so X does not call it as the runner;
   its signature and comment (``finish_step`` inside, no 920-923) are R's to align.
4. The segment list names the scalar combine (921-922) between the limiter
   and the ``scalar_r`` divide (section 7).
5. X owns the divide at ``src/mesh/meshblock.cpp:1199``: the scalar slot's
   op, as at 923, run after each ``exchange_ghost_zones`` (corner refresh 1194-1195); S's exchange part
   carries no divide (section 7; ``meshblock_parts.hpp``).
6. EF rev 3.7 (``f5c8e9a7``) has no conflict with X 3.7; X cites EF there
   (EF.rst:671, 675-685, 2095-2109; section 4).

Round-13 notes
~~~~~~~~~~~~~~

The review of rev 3.8 (``7109d465``), points 1-4:

1. No pin op is reordered. The pin's callers step chemistry after
   ``block->forward`` (``MeshBlock::forward``, exchange first) returns, so the
   slot ends that call. No pin chemistry caller uses ``Mesh::forward`` (exchange last);
   after its final exchange the next stage would reconstruct from stale
   ghosts, so there the slot runs just before it: new, checked one block
   against two (section 5).
2. K rev 2.8 states the idempotent ``prepare()`` (K.rst:280-283 at ``f84a5c58``), gate K-G10.
3. K.rst:756 names ``MeshBlock::set_chemistry`` but not its declarer; X asks
   K to state that S-K4 declares it and its Python binding.
4. The limiter in the slot's cons2prim (``src/eos/ideal_moist.cpp:185``, ``U'->W`` with a reference; 211, ``U->W`` and ``U'->W`` without one, 182) is EF's site: the call reuses EF's ``apply_conserved_limiter_`` (``EF.rst:187-192`` at ``f5c8e9a7``), and EF agreed to both sites in the lane X thread.

Round-14 notes
~~~~~~~~~~~~~~

The review of rev 3.9 (``00dee910``), points 1-4:

1. X asks K rev 3.0 to state that S-K4 also declares ``Mesh::set_chemistry``,
   forwarding to every block's ``MeshBlock::set_chemistry``, with its binding; a
   multi-block driver calls it once (section 7). K.rst:762's "X.rst lines
   351-355" moved in this revision (section 7).
2. ``Mesh::forward`` has 7 example call sites and more in ``tests/`` at the pin; none steps
   chemistry, so none changes (section 7; skeleton comment fixed).
3. Section 5 gates the slot one block against two: a hydro-only baseline on
   the pin's Mesh::forward harness, then the reaction deck; owner X, unrun.
4. K rev 2.9 (``c587e4d3``, K.rst:762) states that S-K4 declares ``MeshBlock::set_chemistry`` and its binding.

Round-15 notes
~~~~~~~~~~~~~~

The review of rev 3.10 (``6d39357b``), points 1-3:

1. The gate compares one deck, one block against two, over every owned cell
   for 10 steps (2 catch a stale ghost), bitwise, with a mutant that must fail; it adds
   owned cells, N steps, species and the slot to the pin's harness (one step,
   ghost cells, 1e-12); unrun (section 5).
2. One adapter per block from the shared, read-only reaction set, since the
   block pool runs blocks concurrently (section 7); built one after another
   outside the pool, as kintera writes its options and species list on build;
   a shared-source race is a hypothesis.
3. K rev 3.0 (``112113f6``) states ``Mesh::set_chemistry`` (K.rst:778) and the per-block split (K.rst:382-394).

Round-16 notes
~~~~~~~~~~~~~~

The review of rev 3.11 (``af403e68``), points 1-2:

1. The executor holds one adapter per block, ``set_chemistry(block, source)``;
   ``MeshImpl::set_chemistry`` takes the reaction options by value, so ``Mesh``
   owns the copy that the first build normalizes (sections 3 and 7).
2. The gate states its fallback tolerance (measured amplification, a
   hypothesis until measured), initial state (deck as loaded, limiter then off,
   no restart) and commits (baseline with X4; source after X4 and EF-8) (section 5).

Round-17 notes
~~~~~~~~~~~~~~

The lead's items 1-4 on rev 3.12 (``6405a34a``):

1. A bare ``MeshBlock`` owns a one-block executor and ``MeshBlock::set_chemistry``
   builds its adapter into slot 0; ``Mesh::set_chemistry`` builds one per block
   into the ``Mesh`` executor (section 7; ``stage_executor.hpp``).
2. X4 carries the gate's no-source baseline; the source run lands with S-K4 (K rev 3.1 ``5a714053``, K.rst:650-652).
3. The baseline is strictly bitwise; the fallback serves the source run only (section 5).
4. The copy shares the eight sub-option pointers, but the first build writes only the
   copy's own members, so no deep copy is needed (section 7).
5. K rev 3.1 (``5a714053``) agrees: S-K4 after X4 and EF-8 with the source run, the
   by-value signature, slot 0, the shallow copy; the gate asserts the caller's ``uref_R`` unchanged.

Round-18 notes
~~~~~~~~~~~~~~

The coordinator's items on rev 3.13 (``54ca5356``), the reviewer's notes and Cheng's ruling:

1. L-2: ``user_output_callback`` gains ``FluxRegister const&`` (section 3;
   ``stage_executor.hpp``), with its assigners and binding named.
2. R rev 4.3 (``97c62312``) removes ``HydroStage::after_combine``;
   ``StageExecutor::after_combine(stage, dt)`` is the one runner (sections 4, 11).
3. Lane I's six questions are answered in section 11; device dt is refused (section 7).
4. IMEX-ARK is an after-V3 note (Cheng, section 10).
5. A bare ``MeshBlock`` owns a one-block executor, a block of a ``Mesh`` uses
   the Mesh's, and ``MeshBlock::set_chemistry`` on it throws (sections 3 and 7).
6. The bare executor's step start and retry scope are its one block, as a
   one-block ``Mesh`` (section 7).
7. Cheng accepted the chemistry slot (section 5) and L-2; neither carries a
   ``FOR CHENG`` marker.

Round-19 notes
~~~~~~~~~~~~~~

1. Cheng's decision (2026-11-17, relayed by the coordinator): lane I's
   per-column FP64 retry is accepted in place of the whole-block retry. It is
   now the design of record (sections 6 and 10, the capability matrix,
   ``precision_policy.hpp:21``); the whole-block retry is the behaviour before
   the change.
2. The 159 ``src/mesh`` lines that move to O's ``restart.cpp`` and
   ``budget.cpp`` count as X's deletion (RULING V3-1; section 9).
3. Cites: S's table is S.rst:324-742 at ``4b96f4ea``; EF's C4 row is
   EF.rst:187-192 at ``f5c8e9a7``.

12. Skeleton index
------------------

``docs/v3/skeleton/X/``, declarations only, in no build:

- ``precision_policy.hpp``: ``Quantity``, ``PrecisionPolicy``.
- ``state_store.hpp``: ``StateStore``.
- ``stage_executor.hpp``: ``SegmentKind``, ``Op``, ``Segment``,
  ``StageSchedule``, ``GraphKey``, ``ExecutionOptions``, ``Capability``,
  ``StageExecutor``.
- ``meshblock_parts.hpp``: the six parts of ``MeshBlock``, the reset list
  in three phases (step start, stage 0, rollback), ``RetryFlag`` and the
  fixer registry.
- ``pyharp_integrator.hpp``: the ``out=`` signature and its alias rules.
