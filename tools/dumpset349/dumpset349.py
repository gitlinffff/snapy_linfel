#!/usr/bin/env python3
"""RECONSTRUCTION of the chengcli/snapy #349 bitwise dump set.

#349's own harness was never committed; this rebuilds it from the PR text:
20 one-block cases, a deterministic initial state, 20 rk3 steps at a fixed dt,
saved quantity the final hydro_u (and scalar_s). Deck names follow the PR's
table; where the table is ambiguous (which solver goes with which box2d recon
variant) the choice made here is documented in CASES.

No extra cases: plume-roe (edited by C1) cannot run in a hydro deck on base.

  python dumpset349.py --device cuda --dtype float64 --out DIR [--case NAME]
Each case runs in its own child process (kintera's species table is per process).
Writes DIR/<case>.<array>.bin (raw C-order bytes) and DIR/<case>.json (shape, dtype, dt, status).
"""
import argparse
import json
import math
import os
import subprocess
import sys
import tempfile

NSTEP = 20
NG = 3

# name: (kind, recon type, scale, shock, riemann)
BOX2D = {
    "box2d_weno5s_lmars": ("weno5", True, False, "lmars"),
    "box2d_weno5_hllc": ("weno5", False, False, "hllc"),
    "box2d_weno3s_roe": ("weno3", True, False, "roe"),
    "box2d_weno3_lmars": ("weno3", False, False, "lmars"),
    "box2d_cp5_hllc": ("cp5", False, False, "hllc"),
    "box2d_cp3_roe": ("cp3", False, False, "roe"),
    "box2d_plm_lmars": ("plm", False, False, "lmars"),
    "box2d_weno5ss_hllc": ("weno5", True, True, "hllc"),
}
CASES = {}
for k, v in BOX2D.items():
    CASES[k] = ("box2d",) + v
CASES["box3d_weno5s_hllc"] = ("box3d", "weno5", True, False, "hllc")
for rs in ("lmars", "hllc", "roe"):
    CASES[f"gnomonic_weno5s_{rs}"] = ("gnomonic", "weno5", True, False, rs)
CASES["column2048_weno5s_lmars"] = ("column2048", "weno5", True, False, "lmars")
CASES["column2048_weno3s_hllc"] = ("column2048", "weno3", True, False, "hllc")
CASES["column2048_cp5_lmars"] = ("column2048", "cp5", False, False, "lmars")
# moist: (recon, scale, shock) + limiter / sedimentation / scalar flags in the name
CASES["moist_weno5ss_lim_sed_scalar"] = ("moist", "weno5", True, True, "lmars")
CASES["moist_weno5s_lim_sed_scalar"] = ("moist", "weno5", True, False, "lmars")
CASES["moist_weno5s_lim_sed_scalarhole"] = ("moist", "weno5", True, False, "lmars")
CASES["moist_weno5s_nolim_sed"] = ("moist", "weno5", True, False, "lmars")
CASES["shallow_weno5_sroe"] = ("shallow", "weno5", False, False, "shallow-roe")
# plume-roe was tried as an extra case and dropped: on base 730b3500 the hydro
# forward always asks for face pressure and PlumeRoeSolver rejects that at the
# first call, so no hydro deck can reach it.
EXTRA = {}
ALL = dict(CASES, **EXTRA)

R_DRY, GAMMA, T0, P0, G = 287.0, 1.4, 300.0, 1.e5, 10.0


def deck(name):
    kind, rtype, scale, shock, rs = ALL[name]
    recon = {"type": rtype, "scale": scale, "shock": shock}
    dry_species = [{"name": "dry", "composition": {"O": 0.42, "N": 1.56, "Ar": 0.01}, "cv_R": 2.5}]
    cfg = {"reference-state": {"Tref": 0., "Pref": P0}, "species": dry_species,
           "dynamics": {"equation-of-state": {"type": "ideal-gas"},
                        "reconstruct": {"vertical": dict(recon), "horizontal": dict(recon)},
                        "riemann-solver": {"type": rs}},
           "integration": {"type": "rk3", "cfl": 0.4, "implicit-scheme": 0, "nlim": -1, "tlim": 1.e9},
           "forcing": {"const-gravity": {"grav1": -G}}}
    per = {"x1-inner": "reflecting", "x1-outer": "reflecting", "x2-inner": "periodic",
           "x2-outer": "periodic", "x3-inner": "periodic", "x3-outer": "periodic"}
    if kind == "box2d":
        geo = {"type": "cartesian", "bounds": {"x1min": 0., "x1max": 1.e4, "x2min": 0., "x2max": 2.e4,
                                               "x3min": 0., "x3max": 1.e3},
               "cells": {"nx1": 64, "nx2": 128, "nx3": 1, "nghost": NG}}
    elif kind == "box3d":
        geo = {"type": "cartesian", "bounds": {"x1min": 0., "x1max": 1.e4, "x2min": 0., "x2max": 1.e4,
                                               "x3min": 0., "x3max": 1.e4},
               "cells": {"nx1": 32, "nx2": 24, "nx3": 24, "nghost": NG}}
    elif kind == "gnomonic":
        geo = {"type": "gnomonic-equiangle",
               "bounds": {"x1min": 6.4e6, "x1max": 6.41e6, "x2min_pi": -0.25, "x2max_pi": 0.25,
                          "x3min_pi": -0.25, "x3max_pi": 0.25},
               "cells": {"nx1": 32, "nx2": 16, "nx3": 16, "nghost": NG}}
        per = {k: "reflecting" for k in per}
    elif kind == "column2048":
        geo = {"type": "cartesian", "bounds": {"x1min": 0., "x1max": 2.048e4, "x2min": 0., "x2max": 1.,
                                               "x3min": 0., "x3max": 1.},
               "cells": {"nx1": 2048, "nx2": 1, "nx3": 1, "nghost": NG}}
        # no gravity: with const-gravity the x1 well-balanced path made cp5 and
        # weno5s bitwise equal on this column, so the recon variant was not tested
        del cfg["forcing"]
    elif kind == "plume":
        geo = {"type": "cartesian", "bounds": {"x1min": 0., "x1max": 1.e4, "x2min": 0., "x2max": 1.,
                                               "x3min": 0., "x3max": 1.},
               "cells": {"nx1": 256, "nx2": 1, "nx3": 1, "nghost": NG}}
        del cfg["forcing"]
    elif kind == "moist":
        geo = {"type": "cartesian", "bounds": {"x1min": 0., "x1max": 1.e4, "x2min": 0., "x2max": 2.e4,
                                               "x3min": 0., "x3max": 1.e3},
               "cells": {"nx1": 48, "nx2": 64, "nx3": 1, "nghost": NG}}
        cfg["species"] = dry_species + [
            {"name": "vapor", "composition": {"H": 2, "O": 1}, "cv_R": 2.5, "u0_R": 0.},
            {"name": "cloud", "composition": {"H": 2, "O": 1}, "cv_R": 9.0, "u0_R": -3430.}]
        cfg["reactions"] = [{"equation": "vapor <=> cloud", "type": "nucleation",
                             "rate-constant": {"formula": "h2o_ideal"}}]
        cfg["dynamics"]["equation-of-state"] = {"type": "ideal-moist", "limiter": "_lim_" in name}
        cfg["sedimentation"] = {"radius": {}, "density": {}, "const-vsed": {"cloud": -2.0}}
        if "scalar" in name:
            cfg["scalar"] = {"nvar": 1, "names": ["trace"], "reconstruct": {"type": rtype, "shock": shock}}
    elif kind == "shallow":
        geo = {"type": "cartesian", "bounds": {"x1min": -10., "x1max": 10., "x2min": -5., "x2max": 5.},
               "cells": {"nx1": 128, "nx2": 64, "nx3": 1, "nghost": NG}}
        cfg = {"geometry": geo,
               "dynamics": {"equation-of-state": {"type": "shallow-water"},
                            "reconstruct": {"vertical": dict(recon), "horizontal": dict(recon)},
                            "riemann-solver": {"type": "shallow-roe", "dir": "xy"}},
               "boundary-condition": {"external": {k: "periodic" for k in per if not k.startswith("x3")}},
               "integration": {"type": "rk3", "cfl": 0.9, "nlim": -1, "tlim": 1.e9}}
        return cfg
    cfg["geometry"] = geo
    cfg["boundary-condition"] = {"external": per}
    return cfg


def fixed_dt(name, cfg):
    """A fixed dt from the deck alone (not from the build): 0.3 * min dx / (cs + 30 m/s)."""
    kind = ALL[name][0]
    g = cfg["geometry"]
    b, c = g["bounds"], g["cells"]
    if kind == "shallow":
        dx = min((b["x1max"] - b["x1min"]) / c["nx1"], (b["x2max"] - b["x2min"]) / c["nx2"])
        return 0.3 * dx / (math.sqrt(12.) + 2.)
    dxs = [(b["x1max"] - b["x1min"]) / c["nx1"]]
    if kind == "gnomonic":
        dxs.append(b["x1min"] * 0.5 * math.pi / c["nx2"])
    else:
        if c["nx2"] > 1:
            dxs.append((b["x2max"] - b["x2min"]) / c["nx2"])
        if c["nx3"] > 1:
            dxs.append((b["x3max"] - b["x3min"]) / c["nx3"])
    cs = math.sqrt(GAMMA * R_DRY * T0)
    if kind == "plume":
        return 1.e-3
    return 0.3 * min(dxs) / (cs + 30.)


def initial_w(name, block, cfg, torch, snapy):
    kind = ALL[name][0]
    coord = block.module("coord")
    x1 = coord.buffer("x1v").cpu().double()
    x2 = coord.buffer("x2v").cpu().double()
    x3 = coord.buffer("x3v").cpu().double()
    w = dict(block.named_buffers())["hydro.D"].clone().zero_().cpu().double()
    nc3, nc2, nc1 = w.shape[1:]
    Z = x1.view(1, 1, -1).expand(nc3, nc2, nc1)
    X = x2.view(1, -1, 1).expand(nc3, nc2, nc1)
    Y = x3.view(-1, 1, 1).expand(nc3, nc2, nc1)
    if kind == "shallow":
        r2 = Z ** 2 + X ** 2
        w[0] = 10. + 2. * torch.exp(-r2 / 2.)
        w[1] = 0.5 + 0.2 * torch.sin(math.pi * X / 5.)
        w[2] = 0.3 * torch.cos(math.pi * Z / 10.)
        return w
    b = cfg["geometry"]["bounds"]
    z = Z - float(b["x1min"])
    lz = float(b["x1max"]) - float(b["x1min"])
    if kind == "gnomonic":
        xi, eta = (X + 0.25 * math.pi) / (0.5 * math.pi), (Y + 0.25 * math.pi) / (0.5 * math.pi)
    else:
        lx = float(b.get("x2max", 1.)) - float(b.get("x2min", 0.))
        ly = float(b.get("x3max", 1.)) - float(b.get("x3min", 0.))
        xi, eta = (X - float(b.get("x2min", 0.))) / lx, (Y - float(b.get("x3min", 0.))) / ly
    if kind == "column2048":  # acoustic pulse plus a long wave, no gravity
        w[snapy.kIPR] = P0 * (1. + 0.01 * torch.exp(-((z / lz - 0.5) / 0.05) ** 2))
        w[snapy.kIDN] = P0 / (R_DRY * T0) * (1. + 0.02 * torch.exp(-((z / lz - 0.3) / 0.05) ** 2))
        w[snapy.kIV1] = 5. * torch.sin(2 * math.pi * z / lz)
        return w
    if kind == "plume":
        w[snapy.kIDN] = 1. + 0.5 * torch.exp(-((z / lz - 0.5) / 0.1) ** 2)
        w[snapy.kIPR] = 1.e5 * (1. + 0.1 * torch.sin(2 * math.pi * z / lz))
        w[snapy.kIV1] = 0.1 * torch.cos(2 * math.pi * z / lz)
        return w
    H = R_DRY * T0 / G
    # warm bubble on an isothermal hydrostatic profile
    bub = torch.exp(-(((z / lz - 0.3) / 0.12) ** 2 + ((xi - 0.5) / 0.12) ** 2
                      + (((eta - 0.5) / 0.12) ** 2 if nc3 > 1 else 0.)))
    T = T0 * (1. + 0.02 * bub)
    p = P0 * torch.exp(-z / H)
    w[snapy.kIPR] = p
    w[snapy.kIV1] = 2. * torch.sin(2 * math.pi * xi) * torch.sin(math.pi * z / lz)
    if nc2 > 1:
        w[snapy.kIV1 + 1] = 10. * (0.5 + z / lz) + 3. * torch.cos(2 * math.pi * z / lz)
    if nc3 > 1:
        w[snapy.kIV1 + 2] = 5. * (0.25 + z / lz) * torch.cos(2 * math.pi * xi)
    if kind == "moist":
        # vapour well below saturation; a thin cloud layer with gaps (zeros)
        # so sedimentation drains it and the species limiter acts
        qv = 0.002 * torch.exp(-z / (0.5 * lz))
        band = torch.exp(-((z / lz - 0.6) / 0.08) ** 2) * (torch.sin(6 * math.pi * xi) > 0).double()
        qc = 1.e-6 * band
        w[snapy.kICY] = qv
        w[snapy.kICY + 1] = qc
        Rm = R_DRY * (1. - qv - qc) + 461.5 * qv
        w[snapy.kIDN] = p / (Rm * T)
    else:
        w[snapy.kIDN] = p / (R_DRY * T)
    return w


def run_child(name, device, dtype, out):
    import torch
    import yaml
    import snapy
    from snapy import MeshBlock, MeshBlockOptions
    torch.set_default_dtype(torch.float64)
    torch.use_deterministic_algorithms(False)
    cfg = deck(name)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=out) as f:
        yaml.safe_dump(cfg, f)
        tmp = f.name
    try:
        block = MeshBlock(MeshBlockOptions.from_yaml(tmp))
    finally:
        os.unlink(tmp)
    dt_t = getattr(torch, dtype)
    block.to(torch.device(device), dt_t)
    w = initial_w(name, block, cfg, torch, snapy)
    init = {"hydro_w": w.to(device, dt_t)}
    if "scalar" in cfg:
        nc = w.shape[1:]
        Zi = (torch.arange(nc[2], dtype=torch.float64) / nc[2]).view(1, 1, -1).expand(*nc)
        Xi = (torch.arange(nc[1], dtype=torch.float64) / nc[1]).view(1, -1, 1).expand(*nc)
        r = 0.001 * (1. + 0.5 * torch.sin(2 * math.pi * Xi)) * torch.exp(-Zi)
        if name.endswith("scalarhole"):
            hole = ((Zi - 0.5) ** 2 + (Xi - 0.5) ** 2) < 0.15 ** 2
            r = torch.where(hole, torch.zeros_like(r), r)
        init["scalar_r"] = r.unsqueeze(0).clone().to(device, dt_t)
    v, _ = block.initialize(init)
    dt = fixed_dt(name, cfg)
    stages = len(block.module("intg").stages)
    for _ in range(NSTEP):
        for st in range(stages):
            block.forward(v, dt, st)
    if device == "cuda":
        torch.cuda.synchronize()
    meta = {"case": name, "extra": name in EXTRA, "device": device, "dtype": dtype, "dt": dt,
            "nstep": NSTEP, "stages": stages, "arrays": {}}
    bufs = dict(block.named_buffers())
    meta["counters"] = {k: bufs[k].detach().cpu().tolist() for k in bufs
                        if "positivity" in k or "hits" in k or "theta" in k and bufs[k].numel() < 8}
    for key in ("hydro_u", "scalar_s"):
        if key in v and v[key] is not None and v[key].numel() > 0:
            a = v[key].detach().cpu().contiguous()
            with open(os.path.join(out, f"{name}.{key}.bin"), "wb") as fh:
                fh.write(a.numpy().tobytes())
            meta["arrays"][key] = {"shape": list(a.shape), "dtype": str(a.dtype),
                                   "finite": bool(torch.isfinite(a).all())}
    with open(os.path.join(out, f"{name}.json"), "w") as fh:
        json.dump(meta, fh, indent=1, sort_keys=True)
    print(json.dumps(meta))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--out", required=True)
    ap.add_argument("--case", default=None)
    ap.add_argument("--child", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.child:
        run_child(a.case, a.device, a.dtype, a.out)
        return
    names = [a.case] if a.case else list(ALL)
    status = {}
    for n in names:
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", "--case", n,
                            "--device", a.device, "--dtype", a.dtype, "--out", a.out],
                           capture_output=True, text=True)
        status[n] = p.returncode
        with open(os.path.join(a.out, f"{n}.log"), "w") as fh:
            fh.write(p.stdout + "\n---stderr---\n" + p.stderr)
        print(f"{n}: rc={p.returncode}", flush=True)
    with open(os.path.join(a.out, "status.json"), "w") as fh:
        json.dump(status, fh, indent=1)


if __name__ == "__main__":
    main()
