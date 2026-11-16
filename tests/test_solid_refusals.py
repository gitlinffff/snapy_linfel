"""Modules not wired to each column's surface refuse an immersed solid at initialize (full support after 3.0);
the E+P cycle meter (message None) drops pe= instead. Each deck without a solid initializes and steps."""
import ctypes, os, re, sys, pytest, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_immersed_solid import block, kIDN, kIPR
SP = lambda n, cv, u0, c={"H": 2, "O": 1}: {"name": n, "composition": c, "cv_R": cv, "u0_R": u0}
MOIST = {"reference-state": {"Tref": 300., "Pref": 1.e5}, "species": [SP("dry", 2.5, 0., {"O": .42, "N": 1.56, "Ar": .01}),
         SP("vapor", 2.5, 0.), SP("cloud", 9., -3430.)], "reactions": [{"equation": "vapor <=> cloud",
         "type": "nucleation", "rate-constant": {"formula": "h2o_ideal"}}]}
ROWS = [("diffusion", dict(forcing={"diffusion": {"nu_iso": 1.e-3, "kappa_iso": 1.e-3}}), "[Diffusion]"),
        ("sedimentation", dict(moist=True, sed=True), "[Sedimentation]"),
        ("bot-heat", dict(forcing={"bot-heat": {"flux": 1.}}), "[Forcing]"),  # one check for all five
        ("fix_vapor", dict(moist=True, limiter=True), "[EquationOfState]"), ("E+P meter", dict(ncycle=1), None),
]

def deck(moist=False, limiter=False, sed=False, ncycle=0, forcing={}):
    """test_immersed_solid's deck builder: 8 x1 cells, 4 in x2; the state exp(-z), species 1e-3"""
    extra = {"forcing": forcing, "integration": {"ncycle_out": ncycle}} | (MOIST if moist else {}) | (
        {"sedimentation": {"radius": {}, "density": {}, "const-vsed": {"cloud": -2.}}} if sed else {})
    b = block(torch.arange(9.) / 8, n2=4, eos={"type": "ideal-moist", "limiter": limiter} if moist else None,
              extra=extra)
    w, z = b.buffer("hydro.D").clone().zero_(), (torch.arange(14.) - 2.5) / 8
    w[kIDN], w[kIPR], w[kIPR + 1:] = 1.2 * torch.exp(-z), 1.e5 * torch.exp(-z), 1.e-3
    return b, w

@pytest.mark.parametrize("solid", [True, False], ids=["solid", "no-solid"])
@pytest.mark.parametrize("cfg,msg", [r[1:] for r in ROWS], ids=[r[0] for r in ROWS])
def test_solid_refusal(cfg, msg, solid, capfd):
    (b, w), s = deck(**cfg), torch.zeros(1, 10, 14, dtype=torch.bool)
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
