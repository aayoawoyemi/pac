# -*- coding: utf-8 -*-
# _pac_rapm.py — RAPM on team PAC per 100 instead of points per 100.
# Same spec as _pure_rapm_A3000TT.py: same stints, lineup matrix, weights, centering, alpha.
# Target: PAC/100 = 100 * (PTS - TSA*(2*lgTS - s*Lbar)) / poss
#   Lbar = TSA-weighted mean season share of the five on-court players (tsa-share proxy).
import sys; sys.stdout.reconfigure(encoding="utf-8")
import json, os, collections, math
import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import lsqr

be = os.path.dirname(os.path.abspath(__file__))
ANCHOR = sys.argv[1] if len(sys.argv) > 1 else "2526"
SLOPE  = float(sys.argv[2]) if len(sys.argv) > 2 else 0.42
ORDER  = ["2122", "2223", "2324", "2425", "2526"]
ai = ORDER.index(ANCHOR); WIN = ORDER[max(0, ai - 2):ai + 1]
DECAY = {WIN[-1]: 1.0}
if len(WIN) >= 2: DECAY[WIN[-2]] = 0.7
if len(WIN) >= 3: DECAY[WIN[-3]] = 0.49
ALPHA = 3000.0

def poss_of(t): return max(t[1] + 0.44 * t[2] + t[4] - t[3], 0.0)
def tsa_of(t):  return t[1] + 0.44 * t[2]

# season player share + league TS from the PAC season files
SHARE, LGTS = {}, {}
for c in WIN:
    d = json.load(open(os.path.join(be, f"_sv_season_{c}.json"), encoding="utf-8"))
    rows = d if isinstance(d, list) else d.get("rows", list(d.values()))
    tp = ts = 0.0
    for r in rows:
        SHARE[(c, int(r["pid"]))] = float(r["l"])
        tp += float(r["pts"]); ts += float(r["tsa"])
    LGTS[c] = tp / (2 * ts)

R = []; pids = set()
for c in WIN:
    d = json.load(open(os.path.join(be, f"_stintx707v2_{c}.json")))
    dk = DECAY[c]; lg2 = 2 * LGTS[c]
    for g, ss in d.items():
        for st in ss:
            h5 = [int(p) for p in st[0]]; a5 = [int(p) for p in st[1]]
            if float(st[2]) < 15: continue
            for o5, d5, tal in ((h5, a5, st[3]), (a5, h5, st[4])):
                ps = poss_of(tal)
                if ps < 0.3: continue
                sh = [SHARE.get((c, p), 0.0) for p in o5]
                tot = sum(sh)
                lbar = sum(x * x for x in sh) / tot if tot > 0 else 0.0
                pac = tal[5] - tsa_of(tal) * (lg2 - SLOPE * lbar)
                R.append((o5, d5, dk * ps, 100.0 * pac / ps))
                pids.update(o5); pids.update(d5)

pids = sorted(pids); idx = {p: i for i, p in enumerate(pids)}; P = len(pids)
ri = []; ci = []; dat = []; W = []
for r, (o5, d5, w, t) in enumerate(R):
    sw = math.sqrt(w)
    for p in o5: ri.append(r); ci.append(idx[p]); dat.append(sw)
    for p in d5: ri.append(r); ci.append(P + idx[p]); dat.append(sw)
    W.append(sw)
Xw = sp.csr_matrix((dat, (ri, ci)), shape=(len(R), 2 * P)); W = np.array(W)
tg = np.array([r[3] for r in R]); mu = np.average(tg, weights=W ** 2)
Aug = sp.vstack([Xw, sp.identity(2 * P) * np.sqrt(ALPHA)]).tocsr()
sol = lsqr(Aug, np.concatenate([(tg - mu) * W, np.zeros(2 * P)]), iter_lim=4000, atol=1e-12, btol=1e-12)[0]

poss = collections.defaultdict(float)
for c in WIN:
    d = json.load(open(os.path.join(be, f"_stintx707v2_{c}.json")))
    for g, ss in d.items():
        for st in ss:
            for s5, tal in ((st[0], st[3]), (st[1], st[4])):
                for p in s5: poss[int(p)] += poss_of(tal) * DECAY[c]
out = [{"pid": p, "pO": float(sol[idx[p]]), "pD": float(sol[P + idx[p]]),
        "pN": float(sol[idx[p]] - sol[P + idx[p]]), "ps": round(poss[p], 1)} for p in pids]
json.dump({"anchor": ANCHOR, "window": WIN, "alpha": ALPHA, "slope": SLOPE,
           "mu": round(float(mu), 3), "players": out},
          open(os.path.join(be, f"_pac_rapm_{ANCHOR}.json"), "w"))
print(f"PAC RAPM {ANCHOR}: {len(out)} players, mu={mu:.2f}, stints={len(R)}")
