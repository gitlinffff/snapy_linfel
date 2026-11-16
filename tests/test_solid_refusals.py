"""Modules not wired to each column's surface refuse an immersed solid at initialize (full support after 3.0);
the E+P cycle meter (message None) drops pe= instead. Each deck without a solid initializes and steps."""
import ctypes, os, re, sys, tempfile, pytest, torch, yaml
from snapy import MeshBlock, MeshBlockOptions, kIDN, kIPR
torch.set_default_dtype(torch.float64)
SP = lambda n, cv, u0, c={"H": 2, "O": 1}: {"name": n, "composition": c, "cv_R": cv, "u0_R": u0}
MOIST = {"reference-state": {"Tref": 300., "Pref": 1.e5}, "species": [SP("dry", 2.5, 0., {"O": .42, "N": 1.56, "Ar": .01}),
         SP("vapor", 2.5, 0.), SP("cloud", 9., -3430.)], "reactions": [{"equation": "vapor <=> cloud",
         "type": "nucleation", "rate-constant": {"formula": "h2o_ideal"}}]}
ROWS = [("diffusion", dict(forcing={"diffusion": {"nu_iso": 1.e-3, "kappa_iso": 1.e-3}}), "[Diffusion]"),
        ("sedimentation", dict(moist=True, sed=True), "[Sedimentation]"),
        *[(f, dict(forcing={f: c}), "[Forcing]") for f, c in (("bot-heat", {"flux": 1.}), ("relax-bot-velo", {"tau": 1.}),
          ("bot-sponge-lyr", {"tau": 1., "width": .2}), ("relax-bot-temp", {"tau": 1., "btemp": 300.}))],
        ("relax-bot-comp", dict(moist=True, forcing={"relax-bot-comp": {"tau": 1., "species": ["vapor"], "xfrac": [.01]}}), "[Forcing]"),
        ("fix_vapor", dict(moist=True, limiter=True), "[EquationOfState]"), ("E+P meter", dict(ncycle=1), None),
]

def block(moist=False, limiter=False, sed=False, ncycle=0, forcing={}):
    c = {"geometry": {"type": "cartesian", "cells": {"nx1": 8, "nx2": 4, "nx3": 1, "nghost": 3}, "bounds": dict(
             x1min=0., x1max=1., x2min=0., x2max=.25, x3min=0., x3max=1.)}, "integration": dict(type="rk3", ncycle_out=ncycle),
         "dynamics": {"equation-of-state": {"type": "ideal-moist" if moist else "ideal-gas", "limiter": limiter},
                      "riemann-solver": {"type": "lmars"}}, "forcing": {"const-gravity": {"grav1": -1.}} | forcing,
         "boundary-condition": {"external": {f"x{d}-{s}": ("periodic", "reflecting")[d < 3] for d in (1, 2, 3) for s in ("inner", "outer")}}}
    c |= (MOIST if moist else {}) | ({"sedimentation": {"radius": {}, "density": {}, "const-vsed": {"cloud": -2.}}} if sed else {})
    with tempfile.TemporaryDirectory() as d:
        yaml.safe_dump(c, open(os.path.join(d, "c.yaml"), "w"))
        b = MeshBlock(MeshBlockOptions.from_yaml(os.path.join(d, "c.yaml")))
    w, z = b.buffer("hydro.D").clone().zero_(), (torch.arange(14.) - 2.5) / 8
    w[kIDN], w[kIPR], w[kIPR + 1:] = 1.2 * torch.exp(-z), 1.e5 * torch.exp(-z), 1.e-3
    return b, w

@pytest.mark.parametrize("solid", [True, False], ids=["solid", "no-solid"])
@pytest.mark.parametrize("cfg,msg", [r[1:] for r in ROWS], ids=[r[0] for r in ROWS])
def test_solid_refusal(cfg, msg, solid, capfd):
    (b, w), s = block(**cfg), torch.zeros(1, 10, 14, dtype=torch.bool)
    s[..., 3:5], init = solid, {"hydro_w": w} | ({"solid": s} if solid else {})
    w[kIDN][s], w[kIPR][s] = 1.e3, 1.e9
    if solid and msg:  # refused at setup, naming the module
        with pytest.raises(RuntimeError, match=re.escape(msg) + ".*after 3.0"):
            b.initialize(init)
        return
    var, _ = b.initialize(init)
    for st in range(0 if solid else 3):
        b.forward(var, 1.e-5, st)
    assert solid or torch.isfinite(var["hydro_u"]).all()
    if msg is None:  # E+P meter: pe= only without a solid
        capfd.readouterr(), b.print_cycle_info(var, 0., 1.e-3), ctypes.CDLL(None).fflush(None)
        line = "".join(capfd.readouterr())
        assert "cycle=" in line and (" pe=" in line) != solid

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
