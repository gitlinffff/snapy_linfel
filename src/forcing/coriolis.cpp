// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/snap.h>

#include <snap/coord/coord_utils.hpp>
#include <snap/coord/coordinate.hpp>
#include <snap/coord/cubed_sphere_utils.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/layout/cubed_sphere_layout.hpp>
#include <snap/mesh/meshblock.hpp>

#include "forcing.hpp"

namespace snap {

CoriolisOptions CoriolisOptionsImpl::from_yaml(YAML::Node const &forcing) {
  if (!forcing["coriolis"]) return nullptr;

  auto node = forcing["coriolis"];
  auto op = CoriolisOptionsImpl::create();

  op->type() = node["type"].as<std::string>("xyz");
  TORCH_CHECK(op->type() == "xyz" || op->type() == "123",
              "CoriolisOptions: unsupported type ", op->type());

  op->omega1() = node["omega1"].as<double>(0.);
  op->omega2() = node["omega2"].as<double>(0.);
  op->omega3() = node["omega3"].as<double>(0.);
  op->profile() = node["profile"].as<std::string>("constant");
  op->radius() = node["radius"].as<double>(0.);
  op->planet_omega() = node["planet_omega"].as<double>(0.);
  op->traditional() = node["traditional"].as<bool>(false);

  return op;
}

Coriolis123Impl::Coriolis123Impl(CoriolisOptions const &options_,
                                 torch::nn::Module *p)
    : options(options_) {
  phydro = dynamic_cast<HydroImpl const *>(p);
  reset();
}

void Coriolis123Impl::reset() {
  TORCH_CHECK(phydro, "[Coriolis123] Parent Hydro is null");
}

torch::Tensor Coriolis123Impl::forward(torch::Tensor du, torch::Tensor w,
                                       torch::Tensor temp, double dt) {
  if (options->omega1() != 0.0 || options->omega2() != 0.0 ||
      options->omega3() != 0.0) {
    auto m1 = w[IDN] * w[IVX];
    auto m2 = w[IDN] * w[IVY];
    auto m3 = w[IDN] * w[IVZ];
    du[IVX] += 2. * dt * (options->omega3() * m2 - options->omega2() * m3);
    du[IVY] += 2. * dt * (options->omega1() * m3 - options->omega3() * m1);

    if (w.size(1) > 1) {  // 3d
      du[IVZ] += 2. * dt * (options->omega2() * m1 - options->omega1() * m2);
    }
  }

  return du;
}

CoriolisXYZImpl::CoriolisXYZImpl(CoriolisOptions const &options_,
                                 torch::nn::Module *p)
    : options(options_) {
  phydro = dynamic_cast<HydroImpl const *>(p);
  reset();
}

void CoriolisXYZImpl::reset() {
  TORCH_CHECK(phydro, "[CoriolisXYZ] Parent Hydro is null");
  auto pcoord = phydro->pmb->pcoord;

  cubed_sphere = false;
  shallow_water = false;
  traditional = false;
  face_id = -1;

  auto mesh = torch::meshgrid({pcoord->x3v, pcoord->x2v, pcoord->x1v}, "ij");

  auto omegaz = options->omega1();
  auto omegax = options->omega2();
  auto omegay = options->omega3();

  if (pcoord->options->type() == "cartesian") {
    auto x3 = mesh[0];  // y-direction (meridional)

    if (options->profile() == "cartesian-sine-plane") {
      TORCH_CHECK(options->radius() > 0.0,
                  "CoriolisXYZ: radius must be > 0 for cartesian-sine-plane");

      auto planet_omega = options->planet_omega();
      if (planet_omega == 0.0) {
        planet_omega = omegaz;
      }

      omega1 = planet_omega * torch::sin(x3 / options->radius());
      omega2 = torch::zeros_like(x3);
      omega3 = torch::zeros_like(x3);
    } else {
      omega1 = omegaz * ones_like(mesh[0]);
      omega2 = omegax * ones_like(mesh[0]);
      omega3 = omegay * ones_like(mesh[0]);
    }
  } else if (pcoord->options->type() == "cylindrical") {
    auto theta = mesh[1];

    omega1 = theta.cos() * omegax + theta.sin() * omegay;
    omega2 = -theta.sin() * omegax + theta.cos() * omegay;
    omega3 = omegaz * ones_like(mesh[0]);
  } else if (pcoord->options->type() == "spherical-polar") {
    auto theta = mesh[1];
    auto phi = mesh[0];

    omega1 = theta.sin() * phi.cos() * omegax +
             theta.sin() * phi.sin() * omegay + theta.cos() * omegaz;
    omega2 = theta.cos() * phi.cos() * omegax +
             theta.cos() * phi.sin() * omegay - theta.sin() * omegaz;
    omega3 = -phi.sin() * omegax + phi.cos() * omegay;
  } else if (pcoord->options->type() == "gnomonic-equiangle") {
    int r = phydro->pmb->options->layout()->rank();
    auto layout = phydro->pmb->get_layout();
    auto face = std::get<2>(layout->loc_of(r));

    cubed_sphere = true;
    shallow_water = phydro->peos->options->type() == "shallow-water";
    traditional = options->traditional() || shallow_water;
    face_id = face;
    alpha = register_buffer("alpha", mesh[1]);
    beta = register_buffer("beta", mesh[0]);

    auto omega =
        torch::empty({3, mesh[0].size(0), mesh[0].size(1), mesh[0].size(2)},
                     mesh[0].options());
    omega[VEL1].fill_(omegaz);
    omega[VEL2].fill_(omegax);
    omega[VEL3].fill_(omegay);
    cs_cart_to_contra_(omega, alpha, beta, face_id);
    cs_contra_to_sph_(omega, alpha, beta, face_id);

    omega1 = omega[VEL1];
    omega2 = omega[VEL2];
    omega3 = omega[VEL3];
  } else {
    throw std::runtime_error("CoriolisXYZ: unsupported coordinate system");
  }

  register_buffer("omega1", omega1);
  register_buffer("omega2", omega2);
  register_buffer("omega3", omega3);
}

torch::Tensor CoriolisXYZImpl::forward(torch::Tensor du, torch::Tensor w,
                                       torch::Tensor temp, double dt) {
  if (cubed_sphere) {
    auto force = w.narrow(0, IVX, 3).clone();
    force *= w[IDN].unsqueeze(0);
    cs_contra_to_sph_(force, alpha, beta, face_id);

    auto o1 = omega1;
    auto o2 = omega2;
    auto o3 = omega3;
    if (traditional) {
      o2 = torch::zeros_like(omega2);
      o3 = torch::zeros_like(omega3);
    }

    auto m1 = force[VEL1].clone();
    auto m2 = force[VEL2].clone();
    auto m3 = force[VEL3].clone();
    force[VEL1] = 2. * (o3 * m2 - o2 * m3);
    force[VEL2] = 2. * (o1 * m3 - o3 * m1);
    force[VEL3] = 2. * (o2 * m1 - o1 * m2);

    cs_sph_to_contra_(force, alpha, beta, face_id);
    coord_vec_lower_(force, phydro->pmb->pcoord->cosine_cell_kj);
    if (traditional) {
      du.narrow(0, IVY, 2) += dt * force.narrow(0, VEL2, 2);
    } else {
      du.narrow(0, IVX, 3) += dt * force;
    }
    return du;
  }

  auto m1 = w[IDN] * w[IVX];
  auto m2 = w[IDN] * w[IVY];
  auto m3 = w[IDN] * w[IVZ];

  du[IVX] += 2. * dt * (omega3 * m2 - omega2 * m3);
  du[IVY] += 2. * dt * (omega1 * m3 - omega3 * m1);
  du[IVZ] += 2. * dt * (omega2 * m1 - omega1 * m2);

  return du;
}
}  // namespace snap
