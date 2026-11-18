#!/usr/bin/env python3
"""Byte-compare two dump directories written by dumpset349.py.
usage: compare.py DIR_A DIR_B   -> prints per-file result and a summary line.
Exit status 0 only if both runs have every case at rc=0 (status.json), the same
dump files, and every file is byte-identical; otherwise it lists the cases that
differ (or failed / are missing) and exits 1."""
import json, os, sys
import numpy as np
a, b = sys.argv[1], sys.argv[2]
fa = sorted(f for f in os.listdir(a) if f.endswith(".bin"))
fb = sorted(f for f in os.listdir(b) if f.endswith(".bin"))
only = sorted(set(fa) ^ set(fb))
same = diff = 0
names = ["IDN", "IVX", "IVY", "IVZ", "IPR/IEN", "ICY0", "ICY1", "ICY2"]
for f in sorted(set(fa) & set(fb)):
    ba, bb = open(os.path.join(a, f), "rb").read(), open(os.path.join(b, f), "rb").read()
    if ba == bb:
        same += 1; print(f"IDENTICAL {f} ({len(ba)} bytes)"); continue
    diff += 1
    case, key = f[:-4].rsplit(".", 1)
    meta = json.load(open(os.path.join(a, case + ".json")))["arrays"][key]
    xa = np.frombuffer(ba, np.float64 if "64" in meta["dtype"] else np.float32).reshape(meta["shape"])
    xb = np.frombuffer(bb, xa.dtype).reshape(meta["shape"])
    ne = np.argwhere(xa != xb)
    first = ne[0]
    var = int(first[0])
    d = np.abs(xa[var] - xb[var]); ref = np.maximum(np.abs(xa[var]), np.abs(xb[var]))
    rel = np.max(np.where(ref > 0, d / np.where(ref > 0, ref, 1), 0))
    dall = np.abs(xa - xb); ra = np.max(dall / np.where(np.abs(xa) > 0, np.abs(xa), 1))
    print(f"DIFFERS {f}: {len(ne)} elements differ; first at index {tuple(int(i) for i in first)} "
          f"(field {var} {names[var] if key=='hydro_u' and var < len(names) else ''}) "
          f"field max|d|={d.max():.3e} max rel={rel:.3e}; whole-array max|d|={dall.max():.3e} max rel={ra:.3e}")
print(f"SUMMARY files_common={len(set(fa)&set(fb))} identical={same} differ={diff} only_one_side={len(only)} {only}")

bad = set()
for d in (a, b):
    st = os.path.join(d, "status.json")
    if not os.path.exists(st):
        print(f"FAIL {d}: no status.json"); bad.add("<status.json>")
        continue
    for case, rc in json.load(open(st)).items():
        if rc != 0:
            print(f"FAIL {d}: case {case} exited rc={rc}"); bad.add(case)
for f in only:
    bad.add(f[:-4].rsplit(".", 1)[0])
for f in sorted(set(fa) & set(fb)):
    if open(os.path.join(a, f), "rb").read() != open(os.path.join(b, f), "rb").read():
        bad.add(f[:-4].rsplit(".", 1)[0])
if not fa or not fb:
    bad.add("<no dumps>")
if bad:
    print(f"RESULT FAIL: {len(bad)} case(s) differ, failed or are missing: {' '.join(sorted(bad))}")
    sys.exit(1)
print("RESULT PASS: every case byte-identical")
