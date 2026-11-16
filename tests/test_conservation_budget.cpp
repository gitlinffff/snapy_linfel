// Conservation budget on closed decks to C n eps, every step: mass, species,
// E + P (face work; P the corrected PE) or E + rho phi (cell work + fixer), x2
// momentum where periodic. Per stage: x1 walls pass no mass, x2 wrap faces
// agree. Terrain: earth-filled solid under a mountain, the budget over the
// fluid with each column's P in its wall frame, the x1 wall at each column's
// surface.
#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

#include <snap/hydro/gravity_work_radial.hpp>
#include <snap/mesh/mesh.hpp>

#include "cuda_test_gate.hpp"
using namespace snap;
struct Case {  // gw: cell (+ fixer) or face; per: periodic x2/x3; vsed: rain
  char const *name, *deck, *gw;  // qv: vapor, supersaturated above ~0.022
  int scheme;
  bool per, pocket;
  double vsed = 0., qv = 0.01;
  int key = 2;           // dynamics/conversion-order
  bool terrain = false;  // a 3-cell mountain on an x2 (panel) edge
};
double const C = 4., kEps = 2.220446049250313e-16, g = 10.;
bool const kMp = getenv("SNAP_BUDGET_MP") != nullptr;  // E2: three ranks

std::string card(Case const& c) {  // the in-tree deck, closed and small
  auto d = YAML::LoadFile(c.deck);
  d.remove("outputs");
  auto cells = d["geometry"]["cells"];
  if (cells["nx2"].as<int>() == 1) cells["nx2"] = 4, cells["nx3"] = 4;
  for (auto f : {"x2-inner", "x2-outer", "x3-inner", "x3-outer"})
    if (c.per || !d["boundary-condition"]["external"][f])
      d["boundary-condition"]["external"][f] =
          c.per ? "periodic" : "reflecting";
  d["forcing"]["const-gravity"]["grav1"] = -g;
  d["forcing"]["const-gravity"]["gravity-work"] = c.gw;
  d["dynamics"]["conversion-order"] = c.key;
  if (c.terrain) {  // fix_vapor refuses a solid; three fluid cells on a
                    // mountain
    d["dynamics"]["equation-of-state"]["limiter"] = false;
    if (cells["nx1"].as<int>() < 6)
      cells["nx1"] = 8, d["geometry"]["bounds"]["x1max"] = 8.;
  }
  if (c.vsed != 0.)
    d["sedimentation"] =
        YAML::Load("{radius: {}, density: {}, const-vsed: {rain: " +
                   std::to_string(c.vsed) + "}}");
  if (kMp)  // three ranks: at two, both x2 neighbours are the same peer
    d["distribute"] = YAML::Load("{layout: slab, nb2: 3, nb3: 1}"),
    cells["nx2"] = 6;
  d["integration"] = YAML::Load("{type: rk3, cfl: 0.5}");
  d["integration"]["implicit-scheme"] = c.scheme;
  std::string path = std::string("budget_") + c.name +  // one per rank
                     (getenv("RANK") ? getenv("RANK") : "") + ".yaml";
  std::ofstream(path) << d;
  return path;
}
struct Budget : testing::TestWithParam<Case> {};
TEST_P(Budget, closes_to_round_off) {
  auto c = GetParam();
  bool cuda = getenv("SNAP_BUDGET_CUDA");  // the _cuda entry
  if (cuda && !snapy_cuda_test_enabled()) GTEST_SKIP() << "no CUDA";
  if (kMp != (std::string(c.name) == "e2_mp"))
    GTEST_SKIP() << "mp only / not mp";
  if (kMp && cuda) GTEST_SKIP() << "E2 runs on CPU ranks only";
  auto mesh = Mesh(MeshOptionsImpl::from_yaml(card(c)));
  int nb = mesh->blocks.size(),
      ny = mesh->blocks[0]->phydro->peos->nvar() - ICY;
  bool face = std::string(c.gw) == "face";
  MeshVariables vars(nb);
  std::vector<torch::Tensor> vol(nb), phi(nb), dep(nb), fl(nb);
  // terrain: solid cells above il per interior column, 3 exp(-(d / 1.5 dx)^2)
  // around the middle of the x2 edge (lon/lat great-circle d on the sphere)
  auto mountain = [&](int i) {
    auto co = mesh->blocks[i]->pcoord;
    auto lon = [&](int b, char const* n) {
      auto mod =
          dynamic_cast<torch::nn::Module*>(mesh->blocks[b]->pcoord.get());
      return mod->named_buffers()[n].select(-1, 0).cpu();
    };
    int ng = co->il(), k = ng + 4, j = co->ju();
    if (co->options->type() == "cartesian") {
      auto m = torch::meshgrid({co->x3v.cpu(), co->x2v.cpu()}, "ij");
      double x3c = .5 * (co->options->x3min() + co->options->x3max()),
             dx = co->dx2f[ng].item<double>();
      auto d = ((m[1] - co->options->x2max()).square() + (m[0] - x3c).square())
                   .sqrt();
      return (3. * (-(d / (1.5 * dx)).square()).exp()).round().to(torch::kLong);
    }
    auto lo0 = lon(0, "longitude"), la0 = lon(0, "latitude");
    double l0 = .5 * (lo0[k][j] + lo0[k][j + 1]).item<double>(),
           p0 = .5 * (la0[k][j] + la0[k][j + 1]).item<double>(),
           dl = (la0[ng + 1][k] - la0[ng][k]).abs().item<double>();
    auto la = lon(i, "latitude"),
         d = (la.sin() * std::sin(p0) +
              la.cos() * std::cos(p0) * (lon(i, "longitude") - l0).cos())
                 .clamp(-1., 1.)
                 .acos();
    return (3. * (-(d / (1.5 * dl)).square()).exp()).round().to(torch::kLong);
  };
  for (int i = 0; i < nb; ++i) {
    auto b = mesh->blocks[i];
    auto co = b->pcoord;
    auto m = torch::meshgrid({co->x3v, co->x2v, co->x1v}, "ij");
    double x0 = co->options->x1min(), h = co->options->x1max() - x0;
    auto r = m[2], s = (r - x0) / h;
    // isothermal column: scale height h when dry, 300 K air with tracers
    double rho0 = ny ? 1.16 : 1.e5 / (g * h), u0 = std::sqrt(10. / rho0);
    auto w = r.new_zeros({ICY + ny, r.size(0), r.size(1), r.size(2)});
    w[IDN] = rho0 * (-(r - x0) * rho0 * g / 1.e5).exp() *
             (1. + 0.01 * (3. * m[1]).sin() * m[0].cos());
    w[IPR] = w[IDN] / rho0 * 1.e5;
    w[IVX] = u0 * (M_PI * s).sin() * (2. * m[1]).cos();
    w[IVY] = u0 * (M_PI * s).sin();
    for (int n = 0; n < ny; ++n)  // vapor qv, first condensate empty
      w[ICY + n] = (n == 1 ? 0. : n ? 0.01 : c.qv) * (1. + 0.5 * s);
    if (c.pocket) {  // the clamp at the wall and theta < 1 at the x2 edge
      w[ICY].select(-1, co->il()).fill_(1.e-12);
      w[ICY].select(-2, co->jl()).fill_(1.e-12);
    }
    vol[i] = co->cell_volume();
    phi[i] = g * (co->options->type() == "cartesian" ? r - x0 : r);
    if (cuda) b->to(torch::kCUDA);
    for (auto* t : {&vol[i], &phi[i]}) *t = t->to(co->x1v.device());
    vars[i]["hydro_w"] = w.to(co->x1v.device());
    dep[i] = torch::zeros({r.size(0), r.size(1)}, torch::kLong);
    if (c.terrain) {  // interior columns only: ghost columns take the API's
      auto in2 = std::vector<torch::indexing::TensorIndex>{
          torch::indexing::Slice(co->kl(), co->ku() + 1),
          torch::indexing::Slice(co->jl(), co->ju() + 1)};
      dep[i].index_put_(in2, mountain(i).index(in2));
      auto at = torch::arange(r.size(2), torch::kLong);
      auto s = (at >= co->il()) & (at < co->il() + dep[i].unsqueeze(-1));
      vars[i]["solid"] = s.to(co->x1v.device());
      fl[i] = (~s).to(vol[i].dtype()).to(co->x1v.device());
    }
  }
  mesh->initialize(vars);
  auto P = PartOptions().exterior(false);
  auto in = mesh->blocks[0]->part({}, P),
       in3 = mesh->blocks[0]->part({}, P.ndim(3));
  auto totals = [&]() {  // mass, energy, x2 momentum, then each species
    auto t = torch::zeros({3 + ny}, torch::kFloat64);
    for (int i = 0; i < nb; ++i) {
      auto co = mesh->blocks[i]->pcoord;
      int il = co->il(), iu = co->iu();
      auto u = vars[i].at("hydro_u").index(in);
      auto dV = vol[i].index(in3), rho = u[IDN] + u.narrow(0, ICY, ny).sum(0);
      if (c.terrain) dV = dV * fl[i].index(in3);  // the fluid cells
      auto pe = rho * phi[i].index(in3);
      auto f = co->x1f.slice(0, il, iu + 2), v = co->x1v.slice(0, il, iu + 1);
      auto ms = x1_measure(co->options->type());  // P by hand, not through
      auto sl = centroid_slope(rho, v);  // terrain: one-sided at the surface
      auto d = dep[i].index({in3[0], in3[1]}).to(rho.device()).unsqueeze(-1);
      for (int o = 1; c.terrain && o <= d.max().item<int>(); ++o) {
        int n = v.size(0) - o;
        auto so = torch::zeros_like(sl);
        so.narrow(-1, o, n).copy_(
            centroid_slope(rho.narrow(-1, o, n), v.narrow(0, o, n)));
        sl = torch::where(d == o, so, sl);
      }
      if (face)  // corrected_pe_work: P = sum V [rho phi(r_c) - g1 <(x1 -
        pe += g * x1_variance(f, ms != X1Measure::plain) * sl;  // r_c)^2> rho']
      if (face && ms == X1Measure::radial_mid)                  // r_c != x1v
        pe += g * rho * x1_centroid_offset(f, v);
      std::vector<torch::Tensor> x = {rho * dV, (u[IPR] + pe) * dV,
                                      u[IVY] * dV};
      for (int n = 0; n < ny; ++n) x.push_back(u[ICY + n] * dV);
      for (int k = 0; k < 3 + ny; ++k)
        t[k] += x[k].sum().cpu().to(torch::kFloat64);
    }
    if (kMp) {
      std::vector<torch::Tensor> v = {t};
      mesh->blocks[0]->get_layout()->comm->allreduce(v, c10d::ReduceOp::SUM);
    }
    return t;
  };
  auto stage_checks = [&](double dt) {  // the x1 walls, the x2 wrap, the solve
    for (int i = 0; i < nb; ++i) {
      auto b = mesh->blocks[i];
      auto co = b->pcoord;
      int il = co->il(), iu = co->iu(), jl = co->jl(), ju = co->ju() + 1;
      auto kj = [&](torch::Tensor x) {  // interior x3, x2 of a face slice
        return x.slice(-2, co->kl(), co->ku() + 1).slice(-1, jl, ju);
      };
      auto f1 = b->phydro->flux1(), a1 = co->face_area1();
      auto rows = torch::cat({f1.narrow(0, IDN, 1), f1.narrow(0, ICY, ny)});
      auto u = vars[i].at("hydro_u").index(in);
      auto mw = (u[IDN] + u.narrow(0, ICY, ny).sum(0)) * vol[i].index(in3);
      auto sum = [](torch::Tensor x) { return x.abs().sum().item<double>(); };
      auto d = dep[i].to(mw.device()).unsqueeze(-1);  // each column's surface
      auto din = d.slice(0, co->kl(), co->ku() + 1).slice(1, jl, ju);
      double wall = 0.,
             mwall = sum(mw.gather(-1, din)) + sum(mw.select(-1, -1));
      auto at = d + il, a1c = a1.expand({rows.size(1), rows.size(2), -1});
      auto lo = rows.gather(-1, at.expand({rows.size(0), -1, -1, 1})) *
                a1c.gather(-1, at);
      wall += dt * sum(kj(lo.select(-1, 0)));
      wall += dt * sum(kj(rows.select(-1, iu + 1) * a1.select(-1, iu + 1)));
      EXPECT_LE(wall, 10. * kEps * mwall) << "x1 wall flux, stage dt " << dt;
      if (c.per && !kMp) {  // one block: the two copies of the x2 wrap face
        auto f2 = b->phydro->flux2()
                      .slice(1, co->kl(), co->ku() + 1)
                      .slice(3, il, iu + 1);
        EXPECT_TRUE(torch::equal(f2.select(2, jl), f2.select(2, ju)))
            << "periodic x2 face copies differ";
      }
      auto const& lm = b->phydro->peos->limiter_marks();
      EXPECT_TRUE(!lm.defined() || !lm.any().item<bool>()) << "limiter fired";
      if (b->phydro->picorr)  // a failed solve skips the fixer
        EXPECT_EQ(b->phydro->picorr->solve_failed_tensor(torch::kFloat64)
                      .item<double>(),
                  0.);
    }
  };
  auto t0 = totals();
  for (int step = 1; step <= 20; ++step) {
    double dt = mesh->max_time_step(vars);
    for (int st = 0; st < 3; ++st) {
      mesh->forward(vars, dt, st);
      stage_checks(dt);
    }
    auto t = totals();
    printf("budget %s step %2d rel", c.name, step);
    for (int k = 0; k < 3 + ny; ++k) {
      if ((k == 2 && !c.per) || (k > 2 && c.qv > .02)) continue;  // x2 mom.:
      // periodic only; species: the saturating row gates their sum (mass)
      double x0 = t0[k].item<double>(), dx = (t[k] - t0[k]).item<double>();
      double rel = std::abs(dx) / std::abs(x0 ? x0 : t0[0].item<double>());
      printf(" %.2e", rel / (step * kEps));
      EXPECT_LE(rel, C * step * kEps) << "total " << k << " step " << step;
    }
    printf("  (units of n eps)\n");
    double dc = c.qv > .02 ? (t[4] - t0[4]).item<double>() : 1.;  // cloud
    if (c.qv > .02) printf("condensed %.3e kg\n", dc);
    EXPECT_GT(dc, 0.) << "the saturating row made no cloud";
  }
}
char const *A = "test_gravity_energy.yaml", *B = "test_species_face_clamp.yaml",
           *Cs = "test_fix_vapor_volume.yaml",
           *D = "test_cubed_sphere_cell_volume.yaml";
Case const kDecks[] = {{"a_cell_s0", A, "cell", 0},
                       {"a_cell_s1", A, "cell", 1},
                       {"a_cell_s9", A, "cell", 9},
                       {"a_face_s0", A, "face", 0},
                       {"a_face_s1", A, "face", 1, true},
                       {"a_face_s9", A, "face", 9},
                       {"a_per", A, "face", 0, true},
                       {"b_moist_slab", B, "face", 0, true, true},
                       {"b_sat", B, "face", 0, true, false, 0., 0.03},
                       {"b_key4", B, "face", 0, true, false, 0., 0.01, 4},
                       {"c_face_s0", Cs, "face", 0, false, false, -0.5},
                       {"c_face_s9", Cs, "face", 9},
                       {"c_cell_s9", Cs, "cell", 9},
                       {"d_cubed", D, "face", 0},
                       {"d_cubed_s9", D, "face", 9},
                       {"t_cart_s0", A, "face", 0, 0, 0, 0., .01, 2, true},
                       {"t_cart_s9", A, "face", 9, 0, 0, 0., .01, 2, true},
                       {"t_cubed_s0", D, "face", 0, 0, 0, 0., .01, 2, true},
                       {"t_cubed_s9", D, "face", 9, 0, 0, 0., .01, 2, true},
                       {"t_moist_s0", B, "face", 0, 0, 0, 0., .01, 2, true},
                       {"e2_mp", A, "face", 0, true}};
INSTANTIATE_TEST_SUITE_P(Decks, Budget, testing::ValuesIn(kDecks),
                         [](auto const& p) {
                           return std::string(p.param.name);
                         });
