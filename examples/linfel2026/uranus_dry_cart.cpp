// C/C++
#include <cmath>
#include <string>

// yaml
#include <yaml-cpp/yaml.h>

// kintera
#include <kintera/constants.h>

// snap
#include <snap/snap.h>

#include <snap/mesh/mesh.hpp>

using namespace snap;

namespace {

struct RunConfig {
  std::string input_file;
  std::string restart_file;
};

RunConfig ParseArguments(int argc, char** argv,
                         std::string const& default_input) {
  RunConfig cfg{default_input, ""};
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if ((arg == "-r" || arg == "--restart") && i + 1 < argc) {
      cfg.restart_file = argv[++i];
    } else {
      cfg.input_file = arg;
    }
  }
  return cfg;
}

void set_user_output_callback(MeshBlock block, double ps, double rd,
                              double cp) {
  block->user_output_callback = [rd, cp, ps](Variables const& vars) {
    auto const& w = vars.at("hydro_w");
    auto temp = w[IPR] / (w[IDN] * rd);

    Variables out;
    out["temp"] = temp;
    out["theta"] = temp * (ps / w[IPR]).pow(rd / cp);
    if (vars.count("scalar_r")) {
      out["tracer"] = vars.at("scalar_r")[0];
    }
    return out;
  };
}

void initialize_block(MeshBlock block, Variables& vars,
                      YAML::Node const& config, torch::Device const& device) {
  auto T0 = config["problem"]["T0"].as<double>();
  auto P0 = config["problem"]["P0"].as<double>();
  auto z0 = config["problem"]["z0"].as<double>();
  auto Tmin = config["problem"]["Tmin"].as<double>();
  auto perturb = config["problem"]["perturb"].as<double>();
  auto grav = -config["forcing"]["const-gravity"]["grav1"].as<double>();
  auto gamma = config["dynamics"]["equation-of-state"]["gammad"].as<double>();
  auto weight = config["dynamics"]["equation-of-state"]["weight"].as<double>();

  auto pcoord = block->pcoord;
  auto rd = kintera::constants::Rgas / weight;
  auto cp = gamma / (gamma - 1.0) * rd;

  auto grids = torch::meshgrid({pcoord->x3v, pcoord->x2v, pcoord->x1v}, "ij");
  auto x1v = grids[2];

  int nc3 = pcoord->options->nc3();
  int nc2 = pcoord->options->nc2();
  int nc1 = pcoord->options->nc1();

  auto w = torch::zeros(
      {5, nc3, nc2, nc1},
      torch::TensorOptions().dtype(torch::kFloat64).device(device));
  // order: IDN, IVX, IVY, IVZ, IPR
  // w[IDN] is density, w[IVX:IVZ] are velocities, and w[IPR] is pressure.

	auto opts = torch::TensorOptions().dtype(torch::kFloat64).device(device);
  torch::manual_seed(1234 + block->options->layout()->rank());
  auto eps = perturb * torch::randn({nc3, nc2, 1}, opts);

  auto T0_col = T0 * (1.0 + eps);
  auto P0_col = P0 * (1.0 + eps);

  auto temp_ad = T0_col - grav * (x1v - z0) / cp;
  auto temp = torch::maximum(temp_ad, torch::full_like(temp_ad, Tmin));

  torch::Tensor pres;
  if (Tmin < T0) {
    auto z_iso = cp * (T0_col - Tmin) / grav + z0;
    auto pres_iso =
        P0_col * torch::pow(torch::full_like(T0_col, Tmin) / T0_col, cp / rd);
    auto pres_ad =
        P0_col * torch::pow(torch::clamp_min(temp_ad, Tmin) / T0_col, cp / rd);
    pres = torch::where(
        temp_ad > Tmin, pres_ad,
        pres_iso * torch::exp(-grav * (x1v - z_iso) / (rd * Tmin)));
  } else {   // Entire profile is isothermal
    pres = P0_col * torch::exp(-grav * (x1v - z0) / (rd * Tmin));
  }

  w[IPR] = pres;
  w[IDN] = pres / (rd * temp);

  vars["hydro_w"] = w;

  if (block->pscalar->nvar() > 0) {
    auto scalar_r = torch::zeros(
        {block->pscalar->nvar(), nc3, nc2, nc1},
        torch::TensorOptions().dtype(torch::kFloat64).device(device));
    auto x1min = pcoord->x1v[0];
    auto x1max = pcoord->x1v[-1];
    scalar_r[0] = (1.0 - (x1v - x1min) / (x1max - x1min)).clamp(0.0, 1.0);
    vars["scalar_r"] = scalar_r;
  }

  set_user_output_callback(block, P0, rd, cp);
}

}  // namespace

int main(int argc, char** argv) {
  torch::set_num_threads(1);
  torch::set_num_interop_threads(1);

  auto args = ParseArguments(argc, argv, "jupiter_dry.yaml");
  auto config = YAML::LoadFile(args.input_file);

  auto mesh = Mesh(MeshOptionsImpl::from_yaml(args.input_file));
  auto device = torch::Device(mesh->options->device_str());
  if (device.is_cuda()) {
    std::cout << "Running on CUDA" << std::endl;
  }
  mesh->to(device);

  MeshVariables vars(mesh->blocks.size());
  for (size_t i = 0; i < mesh->blocks.size(); ++i) {
    if (args.restart_file.empty()) {
      initialize_block(mesh->blocks[i], vars[i], config, device);
    }
  }

  double current_time = args.restart_file.empty()
                            ? mesh->initialize(vars)
                            : mesh->initialize(vars, args.restart_file.c_str());
  mesh->make_outputs(vars, current_time);

  int cycle = mesh->blocks.front()->cycle;
  while (!mesh->blocks.front()->pintg->stop(cycle, current_time)) {
    ++cycle;
    mesh->set_cycle(cycle);

    auto dt = mesh->max_time_step(vars);
    mesh->print_cycle_info(vars, current_time, dt);

    for (int stage = 0; stage < mesh->blocks.front()->pintg->stages.size();
         ++stage) {
      mesh->forward(vars, dt, stage);
    }

    int redo = mesh->check_redo(vars);
    if (redo > 0) {
      cycle = mesh->blocks.front()->cycle;
      continue;
    }
    if (redo < 0) break;

    current_time += dt;
    mesh->make_outputs(vars, current_time);
  }

  mesh->finalize(vars, current_time);
  return 0;
}
