#!/usr/bin/env python3
"""OPTIONAL extra case, OUTSIDE the #349 dump set: extra_column256_dc_plumeroe.

A 1-D ideal-gas column (256 cells, no gravity, donor-cell recon) run with the
plume-roe Riemann solver, added because V3 C1 edits src/riemann/plume_roe.cpp
and none of the 20 #349 cases reaches that solver. It is kept in this separate
script so the 20-case gate (dumpset349.py, its compare and checksums) does not
change.

On main 730b3500 this case does NOT run: the hydro forward always asks the
Riemann solver for face pressure and PlumeRoeSolver rejects that at the first
call ("Face-pressure output is not implemented for PlumeRoeSolver"). The script
then exits 2 and writes no dump. If a later step makes plume-roe reachable,
it writes the same raw-byte dump files as dumpset349.py, which compare.py reads.

  python extra_plumeroe.py --device cpu|cuda --out DIR
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dumpset349 as H  # noqa: E402

NAME = "extra_column256_dc_plumeroe"
H.EXTRA[NAME] = ("plume", "dc", False, False, "plume-roe")
H.ALL[NAME] = H.EXTRA[NAME]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    try:
        H.run_child(NAME, a.device, a.dtype, a.out)
        rc = 0
    except RuntimeError as e:
        print(f"{NAME}: not runnable on this build: {str(e).splitlines()[0]}", file=sys.stderr)
        rc = 2
    with open(os.path.join(a.out, f"{NAME}.status.json"), "w") as fh:
        json.dump({NAME: rc}, fh)
    print(f"{NAME}: rc={rc}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
