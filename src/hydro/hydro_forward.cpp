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
    // _hydro_ref_x1 makes the reference continuous across the x1 process seams
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
      wtmp[ILT][IDN].copy_(torch::where(dl > 0., dl, dsf));
      wtmp[IRT][IDN].copy_(torch::where(dr > 0., dr, dsf));
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
      if (options->verbose()) {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        SINFO(Hydro) << "Flux-x1 time (s): " << elapsed.count() << "\n";
        start = std::chrono::high_resolution_clock::now();
      }
    }

    // add sedimentation flux
    _ps.start("snap::x1.sedimentation");
    if (psed) psed->forward(w, _flux1);

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
    // No exchange, no behavior change at nb1 = 1.
    if (!options->disable_flux_x1() && playout &&
        playout->has_process_group() && playout->options->pz() > 1) {
      auto iloc = playout->loc_of(playout->options->rank());
      int above = playout->neighbor_rank(iloc, {0, 0, 1});
      int below = playout->neighbor_rank(iloc, {0, 0, -1});
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
        auto pack = [&](int face) {
          auto f = _flux1.select(-1, face);
          if (!has_fp) return f.clone();
          return torch::cat({f, _face_pressure1.select(-1, face).unsqueeze(0)},
                            0);
        };
        auto unpack = [&](int face, torch::Tensor const& avg) {
          _flux1.select(-1, face).copy_(avg.narrow(0, 0, nv));
          if (has_fp) _face_pressure1.select(-1, face).copy_(avg[nv]);
        };

        std::vector<CommWorkPtr> seam_sends;
        std::vector<torch::Tensor> up_mine, dn_mine;
        if (above >= 0) {
          up_mine = {pack(iu + 1)};
          seam_sends.push_back(
              playout->comm->send(up_mine, above, kSeamFluxUpTag));
        }
        if (below >= 0) {
          dn_mine = {pack(il)};
          seam_sends.push_back(
              playout->comm->send(dn_mine, below, kSeamFluxDnTag));
        }
        if (above >= 0) {
          std::vector<torch::Tensor> theirs = {torch::empty_like(up_mine[0])};
          playout->comm->recv(theirs, above, kSeamFluxDnTag)->wait();
          unpack(iu + 1, 0.5 * (up_mine[0] + theirs[0]));
        }
        if (below >= 0) {
          std::vector<torch::Tensor> theirs = {torch::empty_like(dn_mine[0])};
          playout->comm->recv(theirs, below, kSeamFluxUpTag)->wait();
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

    auto theta = flux_positivity_theta(uy, f1, f2, f3, pmb->pcoord, dt);
    _ps.start("snap::positivity.census");
    // census before the ghost fill: ghosts are exactly 1 here, so this counts
    // interior (cell, species) entries only, with no double count across ranks
    _positivity_hits += (theta < 1.).sum();

    _ps.start("snap::positivity.ghost");
    Variables tvars;
    tvars["hydro_theta"] = theta;
    _ps.start("snap::positivity.ghost");
    SyncOptions topts;
    topts.interpolate(true).type(kScalar);
    pmb->exchange(tvars, topts);

    BoundaryFuncOptions bops;
    bops.nghost(pmb->pcoord->options->nghost());
    bops.type(kScalar);
    for (int i = 0; i < pmb->options->bfuncs().size(); ++i) {
      if (pmb->options->bfuncs()[i] == nullptr) continue;
      pmb->options->bfuncs()[i](theta, 3 - i / 2, bops);
    }

    _ps.start("snap::positivity.scale");
    flux_positivity_scale_(theta, f1, f2, f3, pmb->pcoord);

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
  for (auto& f : forcings) f.forward(du, w, temp, dt);

  // Preserve the original cell-centred gravity work through the implicit
  // solve: the VIC matrix assumes that energy-momentum coupling is present in
  // its input.  After VIC has completed, replace that work with the value from
  // the same finite-volume face mass flux as continuity.  Use only the x1
  // divergence here: horizontal mass transport does not cross geopotential
  // surfaces.  _flux1 is read after positivity limiting and sedimentation, so
  // it contains every contribution to vertical mass transport.
  _ps.start("snap::gravity_work");
  torch::Tensor gravity_energy_correction;
  if (options->grav() && options->grav()->grav1() != 0. && _flux1.defined() &&
      !options->disable_flux_x1()) {
    auto grav1 = options->grav()->grav1();
    auto non_hydrostatic = options->grav()->non_hydrostatic();
    auto vertical_mass_flux1 = _flux1[IDN].clone();
    if (ny > 0) {
      vertical_mass_flux1 += _flux1.narrow(0, ICY, ny).sum(0);
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
    auto original_gravity_work = dt * w[IDN].slice(-1, is, ie) *
                                 w[IVX].slice(-1, is, ie) * grav1 *
                                 non_hydrostatic;
    if (non_hydrostatic < 1.) {
      original_gravity_work += dt * w[IVX].slice(-1, is, ie) *
                               rho_grav.slice(-1, is, ie) *
                               (1. - non_hydrostatic);
    }
    gravity_energy_correction = face_gravity_work - original_gravity_work;
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
    _ps.start("snap::implicit.solve");
    _apply_implicit_correction(du, w, dt, other);

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

  _ps.stop();
  return du;
}

}  // namespace snap
