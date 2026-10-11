// yaml
#include <yaml-cpp/yaml.h>

#include <cmath>
#include <limits>
#include <string>
#include <vector>

// snap
#include <snap/snap.h>

#include <snap/coord/coord_utils.hpp>
#include <snap/hydro/gravity_work_radial.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/input/check_keys.hpp>
#include <snap/mesh/meshblock.hpp>
#include <snap/utils/log.hpp>

#include "implicit_dispatch.hpp"
#include "implicit_hydro.hpp"

namespace snap {

ImplicitOptions ImplicitOptionsImpl::from_yaml(const std::string& filename,
                                               bool /*verbose*/) {
  auto config = YAML::LoadFile(filename);
  if (!config["integration"]) return nullptr;
  auto intg = config["integration"];
  // pyharp's integrator reads this block too, so its keys are listed by name
  check_keys(intg, "integration",
             {"implicit-scheme", "implicit-advection-cfl", "shear-cfl"},
             "pyharp",
             {"type", "cfl", "tlim", "nlim", "ncycle_out", "verbose"});
  auto op =
      intg["implicit-scheme"] ? from_yaml(intg["implicit-scheme"]) : nullptr;
  if (!op) {
    TORCH_CHECK(!intg["implicit-advection-cfl"] && !intg["shear-cfl"],
                "integration/implicit-advection-cfl and shear-cfl bound an "
                "implicit direction: set implicit-scheme or remove them");
    return nullptr;
  }
  {
    auto read_finite = [&](char const* key, double fallback) {
      if (!intg[key]) return fallback;
      auto const value = intg[key];
      TORCH_CHECK(value.IsScalar(), "integration/", key,
                  " must be a finite number.");
      double parsed = 0.;
      try {
        parsed = value.as<double>();
      } catch (YAML::Exception const&) {
        TORCH_CHECK(false, "integration/", key,
                    " must be a finite number, got '", value.Scalar(), "'.");
      }
      TORCH_CHECK(std::isfinite(parsed), "integration/", key,
                  " must be a finite number, got ", parsed, ".");
      return parsed;
    };
    op->advection_cfl(read_finite("implicit-advection-cfl", 1.0));
    TORCH_CHECK(op->advection_cfl() > 0.,
                "integration/implicit-advection-cfl must be > 0, got ",
                op->advection_cfl(), " (use a large value to relax the bound)");
    op->shear_cfl(read_finite("shear-cfl", 0.0));
    TORCH_CHECK(op->shear_cfl() >= 0.,
                "integration/shear-cfl must be >= 0, got ", op->shear_cfl(),
                " (0 switches the bound off)");
  }
  return op;
}

ImplicitOptions ImplicitOptionsImpl::from_yaml(const YAML::Node& node) {
  int s = node.as<int>();
  // scheme 0 == "none": an implicit object that does nothing. Treat it as if
  // the `implicit-scheme` key were absent (return nullptr) so `implicit-scheme:
  // 0` is a true explicit spelling that also runs at nb1>1, instead of tripping
  // the nb1 guard on a phantom no-op object. picorr != null now faithfully
  // means "implicit is active".
  if (s == 0) return nullptr;
  auto op = ImplicitOptionsImpl::create();
  op->scheme(s);
  return op;
}

std::string ImplicitOptionsImpl::type() const {
  switch (scheme()) {
    case 0:
      return "none";
      break;
    case 1:
      return "vic-partial";
      break;
    case 9:
      return "vic-full";
      break;
    default:
      TORCH_CHECK(false, "Unsupported implicit scheme");
  }
}

ImplicitHydroImpl::ImplicitHydroImpl(ImplicitOptions const& options_,
                                     torch::nn::Module* p)
    : options(options_) {
  phydro = dynamic_cast<HydroImpl const*>(p);
  reset();
}

void ImplicitHydroImpl::reset() {
  TORCH_CHECK(phydro, "[ImplicitHydro] Parent Hydro is null");
  TORCH_CHECK(options, "[ImplicitHydro] options is null");
  options->type();  // throws on an unsupported scheme
  _a = register_buffer("a", torch::empty({0}, torch::kFloat64));
  _b = register_buffer("b", torch::empty({0}, torch::kFloat64));
  _c = register_buffer("c", torch::empty({0}, torch::kFloat64));
  _delta = register_buffer("delta", torch::empty({0}, torch::kFloat64));
  _du0 = register_buffer("du0", torch::empty({0}, torch::kFloat64));
  _corr = register_buffer("corr", torch::empty({0}, torch::kFloat64));
  _mass_corr = register_buffer("mass_corr", torch::empty({0}, torch::kFloat64));
  _clamp_residual =
      register_buffer("clamp_residual", torch::zeros({1}, torch::kFloat64));
  _dry_clamp_step =
      register_buffer("dry_clamp_step", torch::zeros({1}, torch::kFloat64));
}

void ImplicitHydroImpl::ensure_workspace(torch::Tensor const& w) {
  auto pcoord = phydro->pmb->pcoord;
  int nx1 = pcoord->options->nx1();
  int nx2 = pcoord->options->nx2();
  int nx3 = pcoord->options->nx3();
  int m = options->size();

  auto abc_shape = std::vector<int64_t>{1, nx3, nx2, nx1 * m * m};
  auto delta_shape = std::vector<int64_t>{1, nx3, nx2, nx1 * m};

  auto needs_reset = [&](torch::Tensor const& t,
                         std::vector<int64_t> const& shape) {
    return !t.defined() || t.sizes().vec() != shape ||
           t.scalar_type() != w.scalar_type() || t.device() != w.device();
  };

  auto maybe_resize = [&](torch::Tensor& t, std::vector<int64_t> const& shape) {
    if (needs_reset(t, shape)) {
      t.set_(torch::empty(shape, w.options()));
    }
  };

  maybe_resize(_a, abc_shape);
  maybe_resize(_b, abc_shape);
  maybe_resize(_c, abc_shape);
  maybe_resize(_delta, delta_shape);
  maybe_resize(_du0, w.sizes().vec());
  maybe_resize(_corr, w.sizes().vec());
  maybe_resize(_mass_corr, w.sizes().vec());
}

torch::Tensor ImplicitHydroImpl::forward(torch::Tensor du, torch::Tensor w,
                                         torch::Tensor gamma, double dt) {
  return forward_masked(du, w, gamma, dt, torch::Tensor());
}

torch::Tensor ImplicitHydroImpl::forward_masked(torch::Tensor du,
                                                torch::Tensor w,
                                                torch::Tensor gamma, double dt,
                                                torch::Tensor solid) {
  if (options->scheme() == 0) {  // null operation
    if (_corr.sizes() != du.sizes() ||
        _corr.scalar_type() != du.scalar_type() ||
        _corr.device() != du.device()) {
      _corr.set_(torch::zeros_like(du));
    } else {
      _corr.zero_();
    }
    return _corr;
  }

  // Once a stage fails, later stages cannot turn the step into a success.
  if (_solve_failed) {
    _corr.zero_();
    _mass_corr.zero_();
    return _corr;
  }

  TORCH_CHECK(phydro->options->grav(),
              "[ImplicitHydro] forcing does not have const-gravity");

  auto pcoord = phydro->pmb->pcoord;
  auto interior = phydro->pmb->part({0, 0, 0}, PartOptions().exterior(false));
  auto cos_theta = pcoord->cosine_cell_kj;
  auto sin_theta = torch::sqrt(1.0 - cos_theta * cos_theta);

  /*if (torch::isnan(du.index(interior)).any().item<bool>()) {
    TORCH_CHECK(false, "[ImplicitHydro] NaN encountered before implicit solve");
  }*/

  ensure_workspace(w);
  _du0.copy_(du);
  _mass_corr.zero_();

  auto w0 = w.clone();  // restore exactly if this solve is rejected

  auto reject = [&](torch::Tensor const& bad_columns) {
    _solve_failed = true;
    auto columns = bad_columns.nonzero().cpu();
    auto index = columns.accessor<int64_t, 2>();
    for (int64_t n = 0; n < columns.size(0); ++n)
      std::cerr << "[ImplicitHydro] rank=" << get_rank()
                << " VIC singular/near-singular or nonfinite value: column=("
                << index[n][1] + pcoord->kl() << ","
                << index[n][2] + pcoord->jl()
                << ") step=" << phydro->pmb->cycle + 1
                << " stage=" << phydro->rk_stage
                << " retry=" << phydro->pmb->pintg->current_redo
                << "; check_redo will restore step input." << std::endl;
    du.copy_(_du0);
    w.copy_(w0);
    _corr.zero_();
    _mass_corr.zero_();
    return _corr;
  };
  auto finite_columns = [&](torch::Tensor const& values) {
    return torch::isfinite(values.index(interior)).all(0).all(-1).unsqueeze(0);
  };
  auto bad_inputs = torch::logical_not(finite_columns(du) & finite_columns(w) &
                                       finite_columns(gamma.unsqueeze(0)));
  if (bad_inputs.any().item<bool>()) return reject(bad_inputs);

  /// (1) Project to local orthonormal frame
  w[IVY] += w[IVZ] * cos_theta;
  w[IVZ] *= sin_theta;

  coord_vec_raise_(du.narrow(0, IVX, 3), cos_theta);
  pcoord->prim2local1_(du);

  auto mask = solid.defined() ? solid.to(w.options()).contiguous()
                              : torch::zeros_like(gamma);

  auto area = pcoord->face_area1().contiguous();
  auto volume = pcoord->cell_volume().contiguous();
  int nc1 = w.size(-1);
  auto work_lo = .5 * area.narrow(-1, 0, nc1) *
                 (pcoord->x1v - pcoord->x1f.narrow(0, 0, nc1)) / volume;
  auto work_hi = .5 * area.narrow(-1, 1, nc1) *
                 (pcoord->x1f.narrow(0, 1, nc1) - pcoord->x1v) / volume;

  //// -------- Solve block-tridiagonal matrix --------- ////
  auto iter =
      at::TensorIteratorConfig()
          .resize_outputs(false)
          .check_all_same_dtype(true)
          .declare_static_shape(du.index(interior).sizes(),
                                /*squash_dims=*/{0, 3})
          .add_owned_output(du.index(interior))
          .add_owned_output(_mass_corr.index(interior))
          .add_owned_input(w.index(interior))
          .add_owned_input(gamma.unsqueeze(0).index(interior))
          .add_owned_input(area.unsqueeze(0).index(interior))
          .add_owned_input(volume.unsqueeze(0).index(interior))
          .add_input(_a)
          .add_input(_b)
          .add_input(_c)
          .add_input(_delta)
          .add_owned_input(mask.unsqueeze(0).index(interior))
          .add_owned_input(work_lo.unsqueeze(0).contiguous().index(interior))
          .add_owned_input(work_hi.unsqueeze(0).contiguous().index(interior))
          .build();

  // Linearize the FULL gravity: du always carries it (body force + rho_grav
  // sum to grav1); scaling by non_hydrostatic() drops the gravity coupling
  // and destabilizes the solve at dt >> dt_acoustic whenever nh < 1.
  auto grav1 = phydro->options->grav()->grav1();
  // gravity-work: face books the metric-weighted work in the energy row
  bool face_work = phydro->face_work_in_operator();
  int adir = face_work ? kVicFaceWork : 0;
  if (face_work && pcoord->options->type() == "cartesian")
    adir |= kVicCartesianFaceWork;
  bool diffusive_work =
      grav1 != 0. && phydro->options->grav()->gravity_work() == "cell";
  if (diffusive_work) adir |= kVicDiffusiveCell;

  // SNAP_GRAVITY_WORK_RADIAL_EXACT: the energy row also carries the corrected
  // potential energy's work g1 sigma^2 s[drho] of the mass this solve moves
  // (derivation sec 7, eq. 7). Booked only after the solve, that work is
  // explicit in a fixed fraction of the face work of a grid-scale density
  // change, and a rest column grows from round-off at vertical acoustic
  // Courant ~200 (#296). The matrix holds s as a tridiagonal stencil (the
  // one-sided wall slope's third point lumped onto the neighbour), coupled to
  // the solved minus the explicit density change; the post-solve term books
  // the rest, so the total stays g1 sigma^2 s[du - du0].
  bool radial_exact = phydro->radial_exact_work();
  auto in3 =
      phydro->pmb->part({0, 0, 0}, PartOptions().exterior(false).ndim(3));
  int nx1 = pcoord->options->nx1();
  torch::Tensor rx_lo, rx_mid, rx_hi, rx_fluid, rx_mass0;
  auto rx_tri = [&](torch::Tensor const& q) {  // q: [nx3, nx2, nx1]
    auto s = rx_mid * q;
    s.narrow(-1, 1, nx1 - 1) +=
        rx_lo.narrow(-1, 1, nx1 - 1) * q.narrow(-1, 0, nx1 - 1);
    s.narrow(-1, 0, nx1 - 1) +=
        rx_hi.narrow(-1, 0, nx1 - 1) * q.narrow(-1, 1, nx1 - 1);
    return rx_fluid * s;
  };
  auto couple_radial_exact = [&]() {
    if (!radial_exact || nx1 < 3) return;
    int is = pcoord->il(), ie = pcoord->iu() + 1, m = options->size();
    // S[k][i]: the weight of cell k in the slope of cell i
    auto S = centroid_slope(torch::eye(nx1, w.options()),
                            pcoord->x1v.slice(0, is, ie).to(w.options()));
    auto gv = grav1 * x1_variance(pcoord->x1f.slice(0, is, ie + 1),
                                  pcoord->options->type() == "spherical-polar")
                          .to(w.options());
    rx_mid = gv * S.diagonal();
    rx_lo = torch::zeros_like(rx_mid);
    rx_hi = torch::zeros_like(rx_mid);
    rx_lo.narrow(0, 1, nx1 - 1).copy_(S.diagonal(1));
    rx_hi.narrow(0, 0, nx1 - 1).copy_(S.diagonal(-1));
    rx_hi[0] += S[2][0];
    rx_lo[nx1 - 1] += S[nx1 - 3][nx1 - 1];
    rx_lo *= gv;
    rx_hi *= gv;
    rx_fluid = (mask.index(in3) == 0).to(w.options());
    // row IPR, column 0 (the total mass) of each cell's blocks; Eigen stores
    // them column-major
    int nx2 = pcoord->options->nx2(), nx3 = pcoord->options->nx3();
    auto entry = [&](torch::Tensor const& t) {
      return t.view({nx3, nx2, nx1, m, m}).select(-2, 0).select(-1, m - 1);
    };
    entry(_a).sub_(rx_fluid * rx_mid / dt);
    entry(_b).sub_(rx_fluid * rx_lo / dt);
    entry(_c).sub_(rx_fluid * rx_hi / dt);
    rx_mass0 = _du0[IDN].index(in3).clone();
    if (du.size(0) > ICY)
      rx_mass0 += _du0.narrow(0, ICY, du.size(0) - ICY).sum(0).index(in3);
    du[IPR].index(in3).sub_(rx_tri(rx_mass0));
  };

  if ((options->scheme() >> 3) & 1) {
    at::native::vic_assemble_full(du.device().type(), iter, dt, grav1, adir);
    couple_radial_exact();
    at::native::vic_solve_full(du.device().type(), iter, dt, grav1, 0);

  } else {
    // Match the full-VIC pipeline: assemble coefficients, run the column
    // solve + reductions, then apply the per-cell redistribution map.
    at::native::vic_assemble_partial(du.device().type(), iter, dt, grav1, adir);
    couple_radial_exact();
    at::native::vic_solve_partial(du.device().type(), iter, dt, grav1, 0);
  }

  // Both dispatches share the same rejection sentinel. This also catches
  // nonfinite backward-substitution results, before redistribution sees them.
  auto bad_columns = torch::logical_not(torch::isfinite(_delta).all(-1));
  if (bad_columns.any().item<bool>()) return reject(bad_columns);
  if ((options->scheme() >> 3) & 1)
    at::native::vic_redistribute_full(du.device().type(), iter, dt, grav1, 0);
  else
    at::native::vic_redistribute_partial(du.device().type(), iter, dt, grav1,
                                         0);

  // only the availability clamp breaks sum_ch MASS*VOL == M(i) - M(i+1)
  if (!phydro || phydro->meter_step()) {
    int is = pcoord->il();
    int ie = pcoord->iu() + 1;
    int nyc = du.size(0) - ICY;
    auto cell = _mass_corr[IDN].clone();
    for (int n = 0; n < nyc; ++n) cell += _mass_corr[ICY + n];
    auto M = _mass_corr[IVX];
    auto Ml = M.slice(-1, is, ie);
    auto Mu = M.slice(-1, is + 1, ie + 1);
    auto lhs = (cell * pcoord->cell_volume()).slice(-1, is, ie);
    auto rhs = Ml - Mu;
    // per cell: a column max would hide a binding in a low-flux layer
    // Keep the double floor unchanged; 1e-300 is zero in float32.
    const double floor = Ml.scalar_type() == torch::kFloat32
                             ? std::numeric_limits<float>::min()
                             : 1.e-300;
    auto scale = torch::maximum(Ml.abs(), Mu.abs()).clamp_min(floor);
    _clamp_residual.copy_(torch::maximum(
        _clamp_residual, ((lhs - rhs).abs() / scale).max().detach()));
  }
  _dry_clamp_step.copy_(
      torch::maximum(_dry_clamp_step, _mass_corr[IPR].max().detach()));

  if (face_work || diffusive_work) {
    int is = pcoord->il(), ie = pcoord->iu() + 1;
    auto in3 =
        phydro->pmb->part({0, 0, 0}, PartOptions().exterior(false).ndim(3));
    auto volume = pcoord->cell_volume().index(in3);
    auto requested = _mass_corr[IVX];
    auto moved = _mass_corr[IVZ] - requested;
    auto projected =
        (requested.slice(-1, is, ie) - requested.slice(-1, is + 1, ie + 1))
            .slice(-2, pcoord->jl(), pcoord->ju() + 1)
            .slice(-3, pcoord->kl(), pcoord->ku() + 1) /
        volume;
    auto raw_mass = _delta
                        .view({pcoord->options->nx3(), pcoord->options->nx2(),
                               pcoord->options->nx1(), options->size()})
                        .select(-1, 0);
    auto explicit_mass = _du0[IDN].clone();
    if (w.size(0) > ICY)
      explicit_mass += _du0.narrow(0, ICY, w.size(0) - ICY).sum(0);
    auto phi = -grav1 * pcoord->x1v.slice(0, is, ie);
    auto projection_work =
        phi * (raw_mass - explicit_mass.index(in3) - projected);
    auto z = pcoord->x1v.slice(0, is, ie);
    auto dp_lo = -grav1 * (pcoord->x1f.slice(0, is, ie) - z);
    auto dp_hi = -grav1 * (pcoord->x1f.slice(0, is + 1, ie + 1) - z);
    auto clamp_work = -(dp_hi * moved.slice(-1, is + 1, ie + 1) -
                        dp_lo * moved.slice(-1, is, ie))
                           .slice(-2, pcoord->jl(), pcoord->ju() + 1)
                           .slice(-3, pcoord->kl(), pcoord->ku() + 1) /
                      volume;
    du[IPR].index(in3).add_(
        torch::where(mask.index(in3) == 0, projection_work + clamp_work, 0.));
  }

  /// (3) De-project from local orthonormal frame
  w[IVZ] /= sin_theta;
  w[IVY] -= w[IVZ] * cos_theta;
  pcoord->flux2global1_(du);

  // face-wallc retains its legacy post-solve swap. Face mode already books
  // the metric-weighted work inside the matrix on every coordinate system.
  if (grav1 != 0. && phydro->options->grav()->gravity_work() != "cell" &&
      !face_work) {
    int is = pcoord->il();
    int ie = pcoord->iu() + 1;

    auto face_mass = _mass_corr[IVZ];
    auto x1v = pcoord->x1v.slice(0, is, ie);
    auto dphi_top = -grav1 * (pcoord->x1f.slice(0, is + 1, ie + 1) - x1v);
    auto dphi_bot = -grav1 * (pcoord->x1f.slice(0, is, ie) - x1v);
    auto volume = pcoord->cell_volume();
    auto face_gravity_work = -(dphi_top * face_mass.slice(-1, is + 1, ie + 1) -
                               dphi_bot * face_mass.slice(-1, is, ie)) /
                             volume.slice(-1, is, ie);
    auto matrix_gravity_work = dt * grav1 * du[IVX].slice(-1, is, ie);
    auto swap = face_gravity_work - matrix_gravity_work;
    // face-wallc: the x1 wall cells keep the matrix's cell work
    if (phydro->options->grav()->gravity_work() == "face-wallc") {
      if (phydro->is_x1_wall(0)) swap.select(-1, 0).zero_();
      if (phydro->is_x1_wall(1)) swap.select(-1, ie - is - 1).zero_();
    }
    du[IPR].slice(-1, is, ie) += swap;
  }

  // SNAP_GRAVITY_WORK_RADIAL_EXACT: the matrix books the face form for the
  // mass this solve moved and the tridiagonal part of the corrected potential
  // energy's remainder (derivation sec 7) for its raw density change; add the
  // rest, so the remainder is g1 sigma^2 s[du - du0] and E + P closes over the
  // explicit and implicit parts
  if (radial_exact) {
    int is = pcoord->il(), ie = pcoord->iu() + 1;
    auto moved = du[IDN] - _du0[IDN];
    if (du.size(0) > ICY)
      moved += (du.narrow(0, ICY, du.size(0) - ICY) -
                _du0.narrow(0, ICY, du.size(0) - ICY))
                   .sum(0);
    auto work = corrected_pe_work(moved.slice(-1, is, ie), pcoord->x1f,
                                  pcoord->x1v, is, ie, grav1,
                                  pcoord->options->type() == "spherical-polar");
    if (rx_mass0.defined()) {
      int nx2 = pcoord->options->nx2(), nx3 = pcoord->options->nx3();
      auto raw = _delta.view({nx3, nx2, nx1, options->size()}).select(-1, 0);
      work.slice(-2, pcoord->jl(), pcoord->ju() + 1)
          .slice(-3, pcoord->kl(), pcoord->ku() + 1)
          .sub_(rx_tri(raw - rx_mass0));
    }
    // solid cells get none: their slope stencil reads fluid neighbours
    du[IPR].slice(-1, is, ie) +=
        torch::where(mask.slice(-1, is, ie) == 0, work, 0.);
  }

  auto bad_results =
      torch::logical_not(finite_columns(du) & finite_columns(_mass_corr));
  if (bad_results.any().item<bool>()) return reject(bad_results);

  _corr.copy_(du);
  _corr.sub_(_du0);

  auto bad_correction = torch::logical_not(finite_columns(_corr));
  if (bad_correction.any().item<bool>()) return reject(bad_correction);

  return _corr;
}

std::shared_ptr<ImplicitHydroImpl> ImplicitHydroImpl::create(
    ImplicitOptions const& opts, torch::nn::Module* p,
    std::string const& name) {
  TORCH_CHECK(p != nullptr, "[ImplicitHydro] Parent module is nullptr");
  TORCH_CHECK(opts != nullptr, "[ImplicitHydro] Options pointer is nullptr");

  return p->register_module(name, ImplicitHydro(opts, p));
}

}  // namespace snap
