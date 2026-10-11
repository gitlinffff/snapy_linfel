// C/C++
#include <chrono>

// snap
#include <snap/snap.h>

#include <snap/mesh/meshblock.hpp>
#include <snap/utils/log.hpp>

#include "flux_positivity.hpp"
#include "hydro.hpp"
#include "../utils/phase_scope.hpp"

namespace snap {

torch::Tensor HydroImpl::forward(double dt, torch::Tensor u,
                                 Variables const& other) {
  enum { DIM1 = 3, DIM2 = 2, DIM3 = 1 };
  bool has_solid = other.count("solid");
  auto start = std::chrono::high_resolution_clock::now();

  auto playout = pmb->get_layout();

  PhaseScope _ps("snap::hydro.eos");
  //// ------------ (1) Calculate Primitives ------------ ////
  auto const& w = other.at("hydro_w");

  peos->forward(u, w);
  if (options->verbose()) {
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    SINFO(Hydro) << "EOS time (s): " << elapsed.count() << "\n";
    start = std::chrono::high_resolution_clock::now();
  }

  if (has_solid) {
    pmb->pib->mark_prim_solid_(w, other.at("solid"));
  }

  // hydrostatic pressure correction
  torch::Tensor rho_grav = torch::zeros_like(w[IDN]);
  int ny = u.size(0) - ICY;
  // the settling part of the x1 species flux, for the positivity carry
  torch::Tensor fsed1;
  // gravity-work: cell -- the Riemann (background) x1 mass flux F^R; the mass
  // that sedimentation and the positivity limiter add (F - F^R) keeps its
  // face-form gravity work
  torch::Tensor bflux1;
  bool gw_cell = options->grav() && options->grav()->grav1() != 0. &&
                 options->grav()->gravity_work() == "cell";

  //// ------------ (2) Calculate dimension 1 flux ------------ ////
  if (u.size(DIM1) > 1) {
    // Hydrostatic wall revision is a PHYSICAL-boundary operation: it rebuilds
    // the x1 boundary in isentropic balance. On a domain decomposed in x1
    // (cubed nb1>1), a block's local il/iu at an internal seam is NOT a
    // physical boundary — the neighbor's data has already been exchanged
    // there — so applying the wall extrapolation would clobber the true
    // neighbor state. Gate each face on whether it is actually physical.
    // (Slab / single-rank: every block owns the whole x1 column, so both
    // faces are physical and behavior is unchanged — bit-identical.)
    bool grav1 = options->grav() && (options->grav()->grav1() != 0);
    bool phys_x1inner = pmb->options->is_physical_boundary(0, 0, -1);
    bool phys_x1outer = pmb->options->is_physical_boundary(0, 0, 1);

    /*if (grav1) {
      if (phys_x1inner) _revise_x1inner_ghost(w);
      if (phys_x1outer) _revise_x1outer_ghost(w);
    }*/

    // Well-balanced x1 reconstruction: decompose pressure and density into a
    // discretely hydrostatic reference + perturbation, reconstruct only the
    // perturbations, restore the reference at the faces. The restored faces
    // satisfy psf_lo(i) - psf_hi(i) = g*rho(i)*dx1f(i) identically, so a
    // resting stratification generates zero flux residual regardless of the
    // reconstruction. Engaged whenever gravity is on and the scheme is
    // defined: the state carries a pressure row, and the block owns the full
    // x1 column, OR it spans a vertical (nb1>1) decomposition -- in which case
    // _hydro_ref_x1 makes the reference continuous across the x1 block seams
    // with a distributed scan, so WB now engages under x1 decomposition too.
    _ps.start("snap::x1.wbref");
    bool wb_x1 =
        grav1 && w.size(0) > IPR && options->eos()->type() != "shallow-water";

    torch::Tensor wtmp;
    if (wb_x1) {
      auto [psf_lo, pref, dsf, dref] = _hydro_ref_x1(w);
      auto pressure = w[IPR].clone();
      auto density = w[IDN].clone();

      w[IPR] -= pref;
      w[IDN] -= dref;

      // Even-parity ghost perturbations at the walls: p'(is-m) =
      // p'(is+m-1), rho' likewise. Only apply this at physical x1 walls.
      int ng = pmb->pcoord->options->nghost();
      int is = pmb->pcoord->il();
      int iu = pmb->pcoord->iu();
      for (int c : {(int)IPR, (int)IDN}) {
        if (phys_x1inner && !is_outflow(pmb->options->bfuncs()[0])) {
          w[c].narrow(-1, is - ng, ng).copy_(w[c].narrow(-1, is, ng).flip(-1));
        }

        if (phys_x1outer && !is_outflow(pmb->options->bfuncs()[1])) {
          w[c].narrow(-1, iu + 1, ng)
              .copy_(w[c].narrow(-1, iu + 1 - ng, ng).flip(-1));
        }
      }

      // floor=false: reconstruction-stage floors would clamp legitimately
      // negative perturbations.
      _ps.start("snap::x1.recon");
      wtmp = precon1->forward(w, DIM1, /*floor=*/false);

      w[IPR].copy_(pressure);
      w[IDN].copy_(density);
      // Restore full face pressure/density; floor any nonlinear-WENO overshoot
      // that would go non-positive (the references are tiny near the top) back
      // to the reference. At rest the perturbation is ~0 so the floor never
      // fires and well-balancing is preserved.
      auto pl = wtmp[ILT][IPR] + psf_lo;
      auto pr = wtmp[IRT][IPR] + psf_lo;
      wtmp[ILT][IPR].copy_(torch::where(pl > 0., pl, psf_lo));
      wtmp[IRT][IPR].copy_(torch::where(pr > 0., pr, psf_lo));

      auto dl = wtmp[ILT][IDN] + dsf;
      auto dr = wtmp[IRT][IDN] + dsf;
      // Positivity fallback = the adjacent cell's density, not dsf: a
      // reference that overestimates density aloft (a bottom-anchored
      // isentrope did, by orders of magnitude) turns every floor event at a
      // reflecting top wall into a large spurious wall impedance.
      // Shift by one along x1 with EDGE REPLICATION, not torch::roll: roll is
      // circular, so at index 0 it would substitute the density from the TOP
      // of the column. That word is not consumed today (physical faces run
      // il..iu+1 with il = nghost), but a wraparound inside a positivity
      // fallback becomes live the moment a caller changes the range.
      auto n1 = density.size(-1);
      auto rho_below = torch::cat(
          {density.narrow(-1, 0, 1), density.narrow(-1, 0, n1 - 1)}, -1);
      wtmp[ILT][IDN].copy_(torch::where(dl > 0., dl, rho_below));
      wtmp[IRT][IDN].copy_(torch::where(dr > 0., dr, density));
    } else {
      _ps.start("snap::x1.recon");
      wtmp = precon1->forward(w, DIM1);
      if (grav1) {
        if (phys_x1inner) _revise_x1inner_lr(wtmp[ILT], wtmp[IRT]);
        if (phys_x1outer) _revise_x1outer_lr(wtmp[ILT], wtmp[IRT]);
      }
    }

    auto wlr1 =
        has_solid ? pmb->pib->forward(wtmp, DIM1, other.at("solid")) : wtmp;

    // Compute hydrostatic pressure correction
    if (options->grav() && (options->grav()->grav1() != 0) &&
        (options->grav()->non_hydrostatic() < 1.)) {
      int is = pmb->pcoord->il();
      int ie = pmb->pcoord->iu() + 1;
      rho_grav.slice(2, is, ie) = (wlr1[ILT][IPR].slice(2, is + 1, ie + 1) -
                                   wlr1[IRT][IPR].slice(2, is, ie)) /
                                  pmb->pcoord->dx1f.slice(0, is, ie);
    }

    _ps.start("snap::x1.riemann");
    // riemann solver
    if (!options->disable_flux_x1()) {
      auto face_pressure1 = options->eos()->type() == "shallow-water"
                                ? torch::Tensor()
                                : _face_pressure1;
      priemann->forward(wlr1[ILT], wlr1[IRT], DIM1, _flux1, face_pressure1);
      if (gw_cell) {
        bflux1 = _flux1[IDN].clone();
        if (ny > 0) bflux1 += _flux1.narrow(0, ICY, ny).sum(0);
      }
      if (options->verbose()) {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        SINFO(Hydro) << "Flux-x1 time (s): " << elapsed.count() << "\n";
        start = std::chrono::high_resolution_clock::now();
      }
    }

    _ps.start("snap::x1.sedimentation");
    // sedimentation flux; skipped when x1 flux is off (_flux1 not rewritten)
    if (psed && !options->disable_flux_x1()) {
      bool carry = options->eos()->limiter() && ny > 0;
      auto fadv = carry ? _flux1.narrow(0, ICY, ny).clone() : torch::Tensor();
      psed->forward(w, _flux1);
      if (carry) fsed1 = _flux1.narrow(0, ICY, ny) - fadv;
    }

    _ps.start("snap::x1.seam");
    // Make internal x1 seam fluxes single-valued. The two ranks sharing an
    // internal x1 face each compute the face flux from their own
    // reconstruction. When the two are not bit-identical (e.g. a reference
    // rebuilt per block, as the well-balanced reconstruction does once it
    // spans seams), the shared face carries two different flux values --
    // a spurious source/sink of mass, momentum, and energy at the seam.
    // Exchange the seam-face slab with the x1 neighbors and set both sides
    // to the average: averaging is symmetric, so both ranks hold
    // bit-identical values and the conserved sums telescope exactly across
    // the seam, independent of reconstruction details (same principle as
    // flux correction at mesh-refinement boundaries). The scalar advective
    // flux upwinds by this mass flux afterwards and inherits the property.
    // No exchange, no behavior change at nb1 = 1. Across process seams only,
    // as with one block per process; a same-process seam is not averaged.
    if (!options->disable_flux_x1() && playout &&
        playout->has_process_group() && playout->options->pz() > 1) {
      auto iloc = playout->loc_of(playout->options->rank());
      int above = playout->neighbor_rank(iloc, {0, 0, 1});
      int below = playout->neighbor_rank(iloc, {0, 0, -1});
      if (above >= 0 && playout->is_local_block(above)) above = -1;
      if (below >= 0 && playout->is_local_block(below)) below = -1;
      if (above >= 0 || below >= 0) {
        constexpr int kSeamFluxUpTag = 0x7720;  // payload travels upward
        constexpr int kSeamFluxDnTag = 0x7721;  // payload travels downward
        int il = pmb->pcoord->il();
        int iu = pmb->pcoord->iu();
        int nv = _flux1.size(0);
        bool has_fp = _face_pressure1.defined() && _face_pressure1.numel() > 0;

        // Pack the flux slab and the face pressure into ONE contiguous tensor
        // per direction: some backends restrict send/recv to a single tensor,
        // and one message per seam is cheaper anyway. torch::cat allocates
        // fresh storage, so the payload cannot alias the flux buffers while a
        // send is in flight.
        bool has_bf = bflux1.defined();  // F^R is averaged with the flux
        auto pack = [&](int face) {
          std::vector<torch::Tensor> rows = {_flux1.select(-1, face)};
          if (has_fp)
            rows.push_back(_face_pressure1.select(-1, face).unsqueeze(0));
          if (has_bf) rows.push_back(bflux1.select(-1, face).unsqueeze(0));
          if (rows.size() == 1) return rows[0].clone();
          return torch::cat(rows, 0);
        };
        auto unpack = [&](int face, torch::Tensor const& avg) {
          _flux1.select(-1, face).copy_(avg.narrow(0, 0, nv));
          int row = nv;
          if (has_fp) _face_pressure1.select(-1, face).copy_(avg[row++]);
          if (has_bf) bflux1.select(-1, face).copy_(avg[row]);
        };

        std::vector<CommWorkPtr> seam_sends;
        std::vector<torch::Tensor> up_mine, dn_mine;
        if (above >= 0) {
          up_mine = {pack(iu + 1)};
          seam_sends.push_back(
              playout->send_to_block(up_mine, above, kSeamFluxUpTag));
        }
        if (below >= 0) {
          dn_mine = {pack(il)};
          seam_sends.push_back(
              playout->send_to_block(dn_mine, below, kSeamFluxDnTag));
        }
        if (above >= 0) {
          std::vector<torch::Tensor> theirs = {torch::empty_like(up_mine[0])};
          playout->recv_from_block(theirs, above, kSeamFluxDnTag)->wait();
          unpack(iu + 1, 0.5 * (up_mine[0] + theirs[0]));
        }
        if (below >= 0) {
          std::vector<torch::Tensor> theirs = {torch::empty_like(dn_mine[0])};
          playout->recv_from_block(theirs, below, kSeamFluxUpTag)->wait();
          unpack(il, 0.5 * (dn_mine[0] + theirs[0]));
        }
        for (auto& sw : seam_sends) sw->wait();
      }
    }
  }

  _ps.start("snap::x23.recon");
  //// ------------ (3.A) Calculate dimension 2 LR states ------------ ////
  torch::Tensor wtmp2, wtmp3;
  SyncOptions sync_opts;
  sync_opts.cross_panel_only(true).interpolate(false).type(kPrimitive);
  std::vector<CommWorkPtr> works2, works3;
  Variables send_vars2, send_vars3;

  if (u.size(DIM2) > 1) {
    wtmp2 = precon23->forward(w, DIM2);

    // sync left/right states across faces for cubed sphere layout
    if (playout->options->type() == "cubed-sphere") {
      send_vars2["hydro_wl:+"] = wtmp2[ILT];
      send_vars2["hydro_wr:-"] = wtmp2[IRT];
      pmb->begin_exchange(send_vars2, sync_opts.dim(DIM2));
    }
  }

  //// ------------ (3.B) Calculate dimension 3 LR states ------------ ////
  if (u.size(DIM3) > 1) {
    wtmp3 = precon23->forward(w, DIM3);

    // sync left/right states across faces for cubed sphere layout
    if (playout->options->type() == "cubed-sphere") {
      send_vars3["hydro_wl:+"] = wtmp3[ILT];
      send_vars3["hydro_wr:-"] = wtmp3[IRT];
      pmb->begin_exchange(send_vars3, sync_opts.dim(DIM3));
    }
  }

  if (playout->options->type() == "cubed-sphere") {
    bool exchange_dim2 = u.size(DIM2) > 1;
    bool exchange_dim3 = u.size(DIM3) > 1;
    if (exchange_dim2) {
      pmb->launch_exchange(sync_opts.dim(DIM2), works2);
    }
    if (exchange_dim3) {
      pmb->launch_exchange(sync_opts.dim(DIM3), works3);
    }
    if (exchange_dim2) {
      pmb->finalize_exchange(send_vars2, sync_opts.dim(DIM2), works2);
    }
    if (exchange_dim3) {
      pmb->finalize_exchange(send_vars3, sync_opts.dim(DIM3), works3);
    }
  }

  _ps.start("snap::x23.riemann");
  //// ------------ (4.A) Calculate dimension 2 flux ------------ ////
  if (u.size(DIM2) > 1) {
    auto wlr2 =
        has_solid ? pmb->pib->forward(wtmp2, DIM2, other.at("solid")) : wtmp2;
    if (!options->disable_flux_x2()) {
      priemann->forward(wlr2[ILT], wlr2[IRT], DIM2, _flux2);
      if (options->verbose()) {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        SINFO(Hydro) << "Flux-x2 time (s): " << elapsed.count() << "\n";
        start = std::chrono::high_resolution_clock::now();
      }
    }
  }

  //// ------------ (4.B) Calculate dimension 3 flux ------------ ////
  if (u.size(DIM3) > 1) {
    auto wlr3 =
        has_solid ? pmb->pib->forward(wtmp3, DIM3, other.at("solid")) : wtmp3;
    if (!options->disable_flux_x3()) {
      priemann->forward(wlr3[ILT], wlr3[IRT], DIM3, _flux3);
      if (options->verbose()) {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        SINFO(Hydro) << "Flux-x3 time (s): " << elapsed.count() << "\n";
        start = std::chrono::high_resolution_clock::now();
      }
    }
  }

  _ps.start("snap::positivity.theta");
  //// ------- (4.C) Tracer flux positivity limiter (flux_positivity.hpp) ----
  ///////
  // Scale each face's species flux by the donor cell's theta so no cell can be
  // drained below zero by transport in this stage (covers advected species AND
  // the sedimentation flux added into _flux1 above). Runs after the x1 seam
  // averaging, so seam-face fluxes are already single-valued; theta's ghost
  // layer is then filled exactly like conserved-variable ghosts (exchange at
  // internal seams, boundary functions at physical faces) so the donor factor
  // of every shared face is identical on both ranks and conservation stays
  // exact. Positivity of the full multi-stage update follows from the SSP
  // structure of the integrators (see flux_positivity.hpp).
  if (options->eos()->limiter() && ny > 0) {
    auto uy = u.narrow(0, ICY, ny);
    auto f1 = _flux1.defined() ? _flux1.narrow(0, ICY, ny) : torch::Tensor();
    auto f2 = _flux2.defined() ? _flux2.narrow(0, ICY, ny) : torch::Tensor();
    auto f3 = _flux3.defined() ? _flux3.narrow(0, ICY, ny) : torch::Tensor();

    torch::Tensor drain;
    auto theta = flux_positivity_theta(uy, f1, f2, f3, pmb->pcoord, dt, &drain);
    _ps.start("snap::positivity.census");
    // census of interior (cell, species) entries, before the ghost fill
    auto cells = pmb->part({0, 0, 0}, PartOptions().exterior(false));
    auto ti = theta.index(cells).to(torch::kFloat64);
    _positivity_hits += (ti < 1.).sum();
    // severe: theta < 0.9 AND the withheld mass is above round-off of the
    // cell's gas mass. An empty cell drained by a round-off face flux has
    // theta = 0 at any dt; its withheld mass is ~1e-20 of the cell's (#256).
    auto cell_mass = (u[IDN] + uy.sum(0)) * pmb->pcoord->cell_volume();
    auto withheld = ((1. - theta) * drain).index(cells).to(torch::kFloat64);
    auto above =
        withheld > positivity_roundoff(u.scalar_type()) *
                       cell_mass.unsqueeze(0).index(cells).to(torch::kFloat64);
    _positivity_severe += ((ti < 0.9) & above).sum();
    _positivity_min.copy_(torch::minimum(_positivity_min, ti.min()));

    // Raw copy, never interpolated: the donor of a panel-seam face is the
    // neighbour's edge cell, and an interpolated ghost is not its factor.
    _ps.start("snap::positivity.ghost");
    Variables tvars;
    tvars["hydro_theta"] = theta;
    // the energy each species carries, from w, ghosts included: exchanging it
    // with theta changed no bit, so it is not exchanged (#238)
    _ps.start("snap::positivity.enthalpy");
    auto hspec = peos->species_enthalpy(w);
    _ps.start("snap::positivity.ghost");
    SyncOptions topts;
    topts.interpolate(false).type(kScalar);
    pmb->exchange(tvars, topts);

    BoundaryFuncOptions bops;
    bops.nghost(pmb->pcoord->options->nghost());
    bops.type(kScalar);
    for (int i = 0; i < pmb->options->bfuncs().size(); ++i) {
      if (pmb->options->bfuncs()[i] == nullptr) continue;
      pmb->options->bfuncs()[i](theta, 3 - i / 2, bops);
    }

    _ps.start("snap::positivity.carry");
    // theta's depth is not its consequence: measure the flux it removes
    auto f1_pre = f1.defined() ? f1.abs() : torch::Tensor();

    // the withheld species mass keeps its energy and momentum in the donor
    if (hspec.defined()) {
      flux_positivity_carry_(theta, hspec, u.narrow(0, IVX, 3) / w[IDN], _flux1,
                             _flux2, _flux3, pmb->pcoord, fsed1);
    }
    _ps.start("snap::positivity.scale");
    flux_positivity_scale_(theta, f1, f2, f3, pmb->pcoord);

    _ps.start("snap::positivity.meters");
    if (f1_pre.defined()) {
      // non-negative while every bfunc keeps the ghost theta in [0,1]
      auto cut = f1_pre - f1.abs();
      _lim_flux += f1_pre.index(cells).sum().to(torch::kFloat64);
      _lim_cut += cut.index(cells).sum().to(torch::kFloat64);
    }

    if (options->verbose()) {
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = end - start;
      SINFO(Hydro) << "Positivity time (s): " << elapsed.count() << "\n";
      start = std::chrono::high_resolution_clock::now();
    }
  }

  _ps.start("snap::divergence");
  //// ------------ (5) Calculate flux divergence ------------ ////
  _div.set_(pmb->pcoord->forward(w, _flux1, _flux2, _flux3, _face_pressure1));
  if (options->verbose()) {
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    SINFO(Hydro) << "Divergence time (s): " << elapsed.count() << "\n";
    start = std::chrono::high_resolution_clock::now();
  }

  _ps.start("snap::forcing");
  //// ------------ (6) Calculate external forcing ------------ ////
  auto du = torch::zeros_like(_div);
  auto interior = pmb->part({0, 0, 0}, PartOptions().exterior(false));
  du.index(interior) = -dt * _div.index(interior);

  auto temp = peos->compute("W->T", {w});
  // only a block carrying tracers needs the forcings' dry-density increment
  bool track_dry =
      pmb->pscalar && pmb->pscalar->nvar() > 0 && !forcings.empty();
  auto dry_before = track_dry ? du[IDN].clone() : torch::Tensor();
  for (auto& f : forcings) f.forward(du, w, temp, dt);
  _forcing_dry = track_dry ? du[IDN] - dry_before : torch::Tensor();

  // Preserve the original cell-centred gravity work through the implicit
  // solve: the VIC matrix assumes that energy-momentum coupling is present in
  // its input.  After VIC has completed, replace that work with the value from
  // the same finite-volume face mass flux as continuity.  Use only the x1
  // divergence here: horizontal mass transport does not cross geopotential
  // surfaces.  _flux1 is read after positivity limiting and sedimentation, so
  // it contains every contribution to vertical mass transport.
  _ps.start("snap::gravity_work");
  torch::Tensor gravity_energy_correction;
  // gravity-work-fixer: this stage's E+PE change of the dynamics (J)
  torch::Tensor gwfix_stage;
  if (options->grav() && options->grav()->grav1() != 0. && _flux1.defined() &&
      !options->disable_flux_x1()) {
    auto grav1 = options->grav()->grav1();
    auto non_hydrostatic = options->grav()->non_hydrostatic();
    auto vertical_mass_flux1 = _flux1[IDN].clone();
    if (ny > 0) {
      vertical_mass_flux1 += _flux1.narrow(0, ICY, ny).sum(0);
    }
    auto total_mass_flux1 = vertical_mass_flux1.clone();
    // cell: the cell work rho g v (const-gravity forcing) stays; only F - F^R
    // is booked as face work
    if (gw_cell) {
      vertical_mass_flux1 = bflux1.defined()
                                ? vertical_mass_flux1 - bflux1
                                : torch::zeros_like(vertical_mass_flux1);
    }

    int is = pmb->pcoord->il();
    int ie = pmb->pcoord->iu() + 1;
    auto area1 = pmb->pcoord->face_area1();
    auto volume = pmb->pcoord->cell_volume();
    auto phi_face = -grav1 * pmb->pcoord->x1f;
    auto phi_cell = -grav1 * pmb->pcoord->x1v;
    auto potential_flux1 = vertical_mass_flux1 *
                           phi_face.narrow(0, 0, vertical_mass_flux1.size(-1));
    auto vertical_mass_div =
        (area1.slice(-1, is + 1, ie + 1) *
             vertical_mass_flux1.slice(-1, is + 1, ie + 1) -
         area1.slice(-1, is, ie) * vertical_mass_flux1.slice(-1, is, ie)) /
        volume.slice(-1, is, ie);
    auto potential_flux_div =
        (area1.slice(-1, is + 1, ie + 1) *
             potential_flux1.slice(-1, is + 1, ie + 1) -
         area1.slice(-1, is, ie) * potential_flux1.slice(-1, is, ie)) /
        volume.slice(-1, is, ie);

    auto face_gravity_work =
        dt *
        (phi_cell.slice(0, is, ie) * vertical_mass_div - potential_flux_div);

    // cp3/cp5/weno5 faces: the face average exceeds m = rho*v by
    // dx^2/12 (m'' + rho'v'); remove the m'' part as div H,
    // H = dx/12 (m_i - m_{i-1}), zeroed at every physical x1 boundary
    // (walls, outflow and periodic alike); rho'v' is no divergence
    auto type1 = precon1->pinterp1->options->type();
    if (!gw_cell && (type1 == "cp3" || type1 == "cp5" || type1 == "weno5")) {
      int n = ie - is;
      auto x1v = pmb->pcoord->x1v;
      auto rhov = w[IDN] * w[IVX];
      auto curv_flux1 =
          (x1v.narrow(0, is, n + 1) - x1v.narrow(0, is - 1, n + 1)) / 12. *
          (rhov.narrow(-1, is, n + 1) - rhov.narrow(-1, is - 1, n + 1));
      if (pmb->options->is_physical_boundary(0, 0, -1)) {
        curv_flux1.select(-1, 0).zero_();
      }
      if (pmb->options->is_physical_boundary(0, 0, 1)) {
        curv_flux1.select(-1, n).zero_();
      }
      auto area = area1.narrow(-1, is, n + 1);
      face_gravity_work -=
          dt * grav1 *
          (area.narrow(-1, 1, n) * curv_flux1.narrow(-1, 1, n) -
           area.narrow(-1, 0, n) * curv_flux1.narrow(-1, 0, n)) /
          volume.slice(-1, is, ie);
    }
    auto original_gravity_work = dt * w[IDN].slice(-1, is, ie) *
                                 w[IVX].slice(-1, is, ie) * grav1 *
                                 non_hydrostatic;
    if (non_hydrostatic < 1.) {
      original_gravity_work += dt * w[IVX].slice(-1, is, ie) *
                               rho_grav.slice(-1, is, ie) *
                               (1. - non_hydrostatic);
    }
    if (gw_cell) {
      gravity_energy_correction = face_gravity_work;
      if (gravity_work_fixer()) {
        // the PE change of the mass the x1 fluxes move (x1 wall faces dropped:
        // the fixer runs with sealed walls only) plus the gravity work booked
        // into E; x2/x3 fluxes do not move mass across geopotential surfaces
        auto fw = total_mass_flux1.clone();
        if (is_x1_wall(0)) fw.select(-1, is).zero_();
        if (is_x1_wall(1)) fw.select(-1, ie).zero_();
        auto dm =
            -dt *
            (area1.slice(-1, is + 1, ie + 1) * fw.slice(-1, is + 1, ie + 1) -
             area1.slice(-1, is, ie) * fw.slice(-1, is, ie)) /
            volume.slice(-1, is, ie);
        auto e = (original_gravity_work + face_gravity_work +
                  phi_cell.slice(0, is, ie) * dm) *
                 volume.slice(-1, is, ie);
        int js = pmb->pcoord->jl(), je = pmb->pcoord->ju() + 1;
        int ks = pmb->pcoord->kl(), ke = pmb->pcoord->ku() + 1;
        gwfix_stage =
            e.slice(-2, js, je).slice(-3, ks, ke).sum().to(torch::kFloat64);
        // the fixer drops these faces, so it checks that no mass crossed them
        for (int f : {0, 1}) {
          if (!is_x1_wall(f)) continue;
          int i = f == 0 ? is : ie;
          auto m = area1.select(-1, i) * total_mass_flux1.select(-1, i).abs();
          _gwfix_wall += dt * m.slice(-1, js, je)
                                  .slice(-2, ks, ke)
                                  .sum()
                                  .to(_gwfix_wall.device(), torch::kFloat64);
        }
      }
    } else {
      gravity_energy_correction = face_gravity_work - original_gravity_work;
      // face-wallc: the x1 wall cells keep the cell work
      if (options->grav()->gravity_work() == "face-wallc") {
        if (is_x1_wall(0)) gravity_energy_correction.select(-1, 0).zero_();
        if (is_x1_wall(1))
          gravity_energy_correction.select(-1, ie - is - 1).zero_();
      }
    }
  }

  _ps.start("snap::forcing.hydrostatic");
  // apply hydrostatic correction
  if (options->grav() && (options->grav()->non_hydrostatic() < 1.)) {
    du[IVX] += dt * rho_grav * (1. - options->grav()->non_hydrostatic());
    du[IPR] +=
        dt * w[IVX] * rho_grav * (1. - options->grav()->non_hydrostatic());
  }

  if (options->verbose()) {
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    SINFO(Hydro) << "Forcing time (s): " << elapsed.count() << "\n";
    start = std::chrono::high_resolution_clock::now();
  }

  _ps.start("snap::implicit.pre");
  //// ------------ (7) Perform implicit correction ------------ ////
  if (picorr) {
    // The implicit correction is NONLINEAR in dt, so the RK stage weight must
    // be applied INSIDE the solve, not to the solve's result:
    //   correct form   : delta = (I + b*dt*J')^-1 * (b*dt*L),  u += delta
    //   snapy (before)  : delta = (I +   dt*J')^-1 * (  dt*L),  u += b*delta
    // The two RHS scalings are equivalent because the solve is linear in its
    // RHS; the OPERATOR is not. At stage 1 of rk3, b = 1/4, so the I/dt
    // regularisation was 4x too weak.
    //
    // Scale ONLY the dt handed to the correction. du, the flux divergence and
    // the forcings stay at the full dt, or this becomes a different operator.
    // Applied only for the 3-stage integrator, matching the behaviour
    // already in production; generalising would change rk1/rk2 results.
    double dt_corr = dt;
    if (pmb->pintg->stages.size() == 3) {
      if (rk_stage >= 0 && rk_stage < pmb->pintg->stages.size()) {
        dt_corr *= pmb->pintg->stages[rk_stage].wght2();
      } else {
        // Loud, but not fatal: some callers drive HydroImpl::forward directly
        // without the stage loop (tests/test_forcing.cpp), and those must keep
        // working. A silent fallback here would restore the full-dt operator
        // this commit exists to remove, so say so.
        TORCH_WARN_ONCE(
            "[Hydro] rk_stage was not published before the implicit "
            "correction, so it is running with the FULL dt -- the operator "
            "this fix replaces. MeshBlockImpl::advance_local publishes it; a "
            "caller invoking HydroImpl::forward directly will see this.");
      }
    }
    // gravity-work-fixer: the implicit block's E+PE change (its mass transfer
    // and the cell gravity work it linearises; its other energy flux sums to
    // zero over a sealed column)
    auto epe = [&](torch::Tensor const& d) {
      auto m = d[IDN].clone();
      if (ny > 0) m += d.narrow(0, ICY, ny).sum(0);
      auto phi = -options->grav()->grav1() * pmb->pcoord->x1v;
      auto in3 = pmb->part({0, 0, 0}, PartOptions().exterior(false).ndim(3));
      return ((d[IPR] + phi * m) * pmb->pcoord->cell_volume())
          .index(in3)
          .sum()
          .to(torch::kFloat64);
    };
    torch::Tensor epe0;
    if (gwfix_stage.defined()) epe0 = epe(du);
    // face work in the operator: the solve also sees the explicit face
    // correction, so the energy it inverts is the face-form energy
    if (gravity_energy_correction.defined() && face_work_in_operator()) {
      int is = pmb->pcoord->il();
      int ie = pmb->pcoord->iu() + 1;
      du[IPR].slice(-1, is, ie) += gravity_energy_correction;
      gravity_energy_correction = torch::Tensor();
    }
    _ps.start("snap::implicit.solve");
    _apply_implicit_correction(du, w, dt_corr, other);
    _ps.start("snap::implicit.gwfix");
    if (gwfix_stage.defined()) gwfix_stage = gwfix_stage + epe(du) - epe0;

    if (options->verbose()) {
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = end - start;
      SINFO(Hydro) << "Implicit time (s): " << elapsed.count() << "\n";
      start = std::chrono::high_resolution_clock::now();
    }
  }

  _ps.start("snap::gravity_work.apply");
  if (gravity_energy_correction.defined()) {
    int is = pmb->pcoord->il();
    int ie = pmb->pcoord->iu() + 1;
    du[IPR].slice(-1, is, ie) += gravity_energy_correction;
  }

  if (gwfix_stage.defined()) {
    // weight of this stage's du in the step's update u0 -> u1 (stage s:
    // u <- w0 u0 + w1 u + w2 du): w2_s * prod_{t > s} w1_t (rk3: 1/6 1/6 2/3)
    double cw = 1.;
    auto const& st = pmb->pintg->stages;
    if (rk_stage >= 0 && rk_stage < static_cast<int>(st.size())) {
      cw = st[rk_stage].wght2();
      for (int t = rk_stage + 1; t < static_cast<int>(st.size()); ++t)
        cw *= st[t].wght1();
    }
    _gwfix_d += cw * gwfix_stage.to(_gwfix_d.device());
  }

  _ps.stop();
  return du;
}

}  // namespace snap
