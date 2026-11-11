"""Detector v3: learn what a road's cross-section looks like from the known Chaco roads, then
retry Lowry's Road A.

v2 (road_detector.py) only looks for a 5 m dip. The known-road views showed that some published
roads read as raised lines or have berms, so v3 learns the shape instead of assuming it.

Sample = a 100 m stretch at a given direction. Feature = its averaged cross-section, -15..+15 m
at 1 m, detrended on the outer 10-15 m (cm), plus the squares of those values (so a road can be a
dip or a ridge). Model = L2 logistic regression.

Training (Chaco 0.5 m DTM averaged to 1 m, Fig. 11 roads):
  positives: points on marked roads where the marking is straight within 50 m, at its direction
  negatives: random ground >= 25 m from any marked road at random directions; road points turned
             60-90 deg; road points shifted 20-40 m sideways at the road's direction
Scoring: 5 folds split by road segment (a road is never in its own training fold). AUC and
road-found share at a 5% false-alarm rate, vs v2's fixed dip shape on the same samples.

Lowry: Road A (bearing from roads.json, 0-3 km) and 9 control bearings. Each line's own strip
(±200 m) is scored; a "hit" is a 100 m stretch above the strip's 97.5th percentile. On-line share
= along-line metres with a hit within 10 m of the line; background = the same for every other
21 m wide band in the strip.

Outputs: output/detector_v3/ (model.npz, report.txt, lowry_road_A.json, road_A_view_v3.png)
"""
import datetime as dt
import json
import math
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from scipy import ndimage
from scipy.ndimage import map_coordinates, uniform_filter1d
from scipy.optimize import minimize
from scipy.stats import rankdata
from scipy.spatial import cKDTree

import corridor_test as ct
import corridor_views as cv
import mapped_roads as mr
import road_detector as rd

OUT = Path("output/detector_v3")
HALF_W = 15                                   # cross-section half width, m
ALONG = 100                                   # stretch length, m
OFFS = np.arange(-HALF_W, HALF_W + 1)
OUTER = np.abs(OFFS) >= 10
LAMBDA = 1.0
N_POS_PER_SEG = 300
SEED = 7
LOWRY = (37.58473, -108.91967)
ROAD_KM = 3.0
CONTROLS = (-20, -10, -5, 5, 10, 20, -90, 90, 180)
HIT_PCT = 97.5
NEAR_M = 10
STOP_AT = dt.time(20, 15)


def detrend_matrix():
    X = np.c_[np.ones(OFFS.size), OFFS]
    return np.eye(OFFS.size) - X @ np.linalg.pinv(X[OUTER])  @ np.eye(OFFS.size)[OUTER]


A = detrend_matrix()


def features(prof_m):
    p = prof_m @ A.T * 100                                         # cm
    return np.concatenate([p, p ** 2 / 10], axis=-1)


def v2_dip(prof_m):
    """v2's fixed shape on the same cross-section: shoulders (4-10 m) minus centre (<= 2.5 m)."""
    p = prof_m @ A.T
    sh = (np.abs(OFFS) >= 4) & (np.abs(OFFS) <= 10)
    return p[..., sh].mean(-1) - p[..., np.abs(OFFS) <= 2.5].mean(-1)


def auc(pos, neg):
    s = np.r_[pos, neg]
    r = rankdata(s)
    return (r[:pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size)


def found_at(pos, neg, fpr=0.05):
    return float((pos > np.quantile(neg, 1 - fpr)).mean())


class Model:
    def fit(self, X, y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = np.c_[np.ones(len(X)), (X - self.mu) / self.sd]
        w = np.where(y == 1, 0.5 / y.mean(), 0.5 / (1 - y.mean()))          # balance classes

        def loss(b):
            t = Z @ b
            l = np.sum(w * (np.logaddexp(0, t) - y * t)) + LAMBDA * np.sum(b[1:] ** 2)
            g = Z.T @ (w * (1 / (1 + np.exp(-t)) - y)) + 2 * LAMBDA * np.r_[0, b[1:]]
            return l, g
        self.b = minimize(loss, np.zeros(Z.shape[1]), jac=True, method="L-BFGS-B").x
        return self

    def score(self, X):
        return self.b[0] + ((X - self.mu) / self.sd) @ self.b[1:]


# ---------------------------------------------------------------- Chaco training samples
def chaco_samples(rng):
    dem, valid, _ = rd.load_1m()
    with rasterio.open("output/ground_truth/fig11_roads.tif") as s:
        g = s.read(1)
    h, w = dem.shape
    g = g[:h * 2, :w * 2].reshape(h, 2, w, 2)
    road, area = (g == 2).any(axis=(1, 3)), (g >= 1).any(axis=(1, 3))
    lab, _ = ndimage.label(road, structure=np.ones((3, 3)))
    rr, cc = np.nonzero(road)
    tree = cKDTree(np.c_[rr, cc])

    def prof(r, c, d):                                   # d = unit (drow, dcol) along the road
        n = np.array([d[1], -d[0]])
        t = np.arange(-ALONG / 2, ALONG / 2)
        R = r + t[:, None] * d[0] + OFFS[None, :] * n[0]
        C = c + t[:, None] * d[1] + OFFS[None, :] * n[1]
        if R.min() < 0 or C.min() < 0 or R.max() > h - 1 or C.max() > w - 1:
            return None
        if map_coordinates(valid.astype(float), [R, C], order=0).min() < 1:
            return None
        return map_coordinates(dem, [R, C], order=1).mean(0)

    P, N, Pg, Ng, kinds = [], [], [], [], []
    for seg in range(1, lab.max() + 1):
        idx = np.nonzero(lab[rr, cc] == seg)[0]
        if idx.size < 100:
            continue
        for i in rng.choice(idx, min(idx.size, N_POS_PER_SEG * 3), replace=False):
            if sum(1 for x in Pg if x == seg) >= N_POS_PER_SEG:
                break
            nb = tree.query_ball_point([rr[i], cc[i]], ALONG / 2)
            pts = np.c_[rr[nb], cc[nb]].astype(float)
            ev, vec = np.linalg.eigh(np.cov((pts - pts.mean(0)).T))
            if ev[0] <= 0 or ev[1] / ev[0] < 30:                     # junction or bend: skip
                continue
            d = vec[:, 1]
            p = prof(rr[i], cc[i], d)
            if p is None:
                continue
            P.append(p); Pg.append(seg)
            a = math.radians(rng.uniform(60, 90) * rng.choice([-1, 1]))      # turned
            dt_ = np.array([d[0] * math.cos(a) - d[1] * math.sin(a), d[0] * math.sin(a) + d[1] * math.cos(a)])
            q = prof(rr[i], cc[i], dt_)
            if q is not None:
                N.append(q); Ng.append(seg); kinds.append("turned")
            sh = rng.uniform(20, 40) * rng.choice([-1, 1])                    # shifted sideways
            q = prof(rr[i] + sh * d[1], cc[i] - sh * d[0], d)
            if q is not None and tree.query([rr[i] + sh * d[1], cc[i] - sh * d[0]])[0] >= 12:
                N.append(q); Ng.append(seg); kinds.append("shifted")
    far = ndimage.distance_transform_edt(~road) >= 25
    br, bc = np.nonzero(area & far & valid)
    for i in rng.choice(br.size, 2 * len(P), replace=False):
        a = rng.uniform(0, math.pi)
        q = prof(br[i], bc[i], np.array([math.cos(a), math.sin(a)]))
        if q is not None:
            N.append(q); Ng.append(-1); kinds.append("random")
    return np.array(P), np.array(N), np.array(Pg), np.array(Ng), np.array(kinds)


def cross_validate(P, N, Pg, Ng, kinds, rng):
    segs = np.unique(Pg)
    order = segs[np.argsort([-(Pg == s).sum() for s in segs])]
    fold_of = {s: k % 5 for k, s in enumerate(order)}
    rnd_fold = rng.integers(0, 5, size=len(N))                 # random ground spread over folds
    Pf = np.array([fold_of[s] for s in Pg])
    Nf = np.array([fold_of[s] if s >= 0 else rnd_fold[i] for i, s in enumerate(Ng)])
    sp, sn = np.zeros(len(P)), np.zeros(len(N))
    FP, FN = features(P), features(N)
    for k in range(5):
        X = np.r_[FP[Pf != k], FN[Nf != k]]
        y = np.r_[np.ones((Pf != k).sum()), np.zeros((Nf != k).sum())]
        m = Model().fit(X, y)
        sp[Pf == k], sn[Nf == k] = m.score(FP[Pf == k]), m.score(FN[Nf == k])
    return sp, sn


# ---------------------------------------------------------------- Lowry
def line_scores(model, tiles, p0, bearing, length_m):
    u = np.array([math.sin(math.radians(bearing)), math.cos(math.radians(bearing))])
    v = np.array([u[1], -u[0]])
    offs = np.arange(-ct.BAND_M, ct.BAND_M + 1, 1.0)
    out = []
    pad = ALONG // 2 + 5
    for s0 in np.arange(0, length_m, 1000.0):
        s1 = min(s0 + 1000, length_m)
        ss = np.arange(s0 - pad, s1 + pad, 1.0)
        S, T = np.meshgrid(ss, offs, indexing="ij")
        z = ct.sample(tiles, p0[0] + S * u[0] + T * v[0], p0[1] + S * u[1] + T * v[1])
        ok = ~np.isnan(z)
        z = np.where(ok, z, np.nanmean(z))
        avg = uniform_filter1d(z, ALONG, axis=0)[pad:pad + int(s1 - s0)]
        okw = uniform_filter1d(ok.astype(float), ALONG, axis=0)[pad:pad + int(s1 - s0)] > 0.95
        win = np.lib.stride_tricks.sliding_window_view(avg, OFFS.size, axis=1)
        sc = model.score(features(win).reshape(-1, 2 * OFFS.size)).reshape(win.shape[:2])
        okc = np.lib.stride_tricks.sliding_window_view(okw, OFFS.size, axis=1).all(-1)
        out.append(np.where(okc, sc, np.nan))
    return np.concatenate(out), offs[HALF_W:-HALF_W], u, v


def on_line_share(sc, offs):
    thr = np.nanpercentile(sc, HIT_PCT)
    hit = np.nan_to_num(sc, nan=-np.inf) > thr
    hit = uniform_filter1d(hit.astype(float), ALONG, axis=0) > 0.5          # held for >= ~50 of 100 m
    bands = {}
    for o in range(int(offs[0]) + NEAR_M, int(offs[-1]) - NEAR_M + 1, 2 * NEAR_M + 1):
        m = np.abs(offs - o) <= NEAR_M
        bands[o] = float(hit[:, m].any(1).mean())
    on = float(hit[:, np.abs(offs) <= NEAR_M].any(1).mean())
    others = [b for o, b in bands.items() if abs(o) > 2 * NEAR_M]
    return on, float(np.median(others)), hit


def pieces_from(hit, offs, min_m=100):
    near = hit[:, np.abs(offs) <= NEAR_M]
    rows = near.any(1)
    rows = ndimage.binary_closing(rows, np.ones(21)) | rows
    lab, n = ndimage.label(rows)
    out = []
    for sl in ndimage.find_objects(lab):
        a, b = sl[0].start, sl[0].stop
        if b - a >= min_m:
            cols = np.nonzero(near[a:b].any(0))[0]
            o = offs[np.abs(offs) <= NEAR_M][cols].mean() if cols.size else 0.0
            out.append((float(a), float(b), float(o)))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    log = []

    def say(s):
        print(s, flush=True)
        log.append(s)

    P, N, Pg, Ng, kinds = chaco_samples(rng)
    say(f"Chaco samples: {len(P)} road stretches on {len(np.unique(Pg))} segments; "
        f"{len(N)} non-road ({', '.join(f'{k} {(kinds == k).sum()}' for k in np.unique(kinds))})")
    sp, sn = cross_validate(P, N, Pg, Ng, kinds, rng)
    d2p, d2n = v2_dip(P), v2_dip(N)
    say("Held-out roads (5 folds by segment), AUC / road found at 5% false alarms:")
    for name, a, b in (("v3 learned", sp, sn), ("v2 dip shape", d2p, d2n), ("v2 dip, either sign", np.abs(d2p), np.abs(d2n))):
        say(f"  {name:20s} all non-road {auc(a, b):.3f} / {found_at(a, b):.0%}   "
            + "   ".join(f"{k} {auc(a, b[kinds == k]):.3f}" for k in np.unique(kinds)))
    per = []
    for s in np.unique(Pg):
        m = Pg == s
        per.append((int(s), int(m.sum()), auc(sp[m], sn[kinds == "random"]), auc(d2p[m], d2n[kinds == "random"])))
    say("Per segment AUC vs random ground (v3 | v2): " + ", ".join(f"{s}: {a:.2f}|{b:.2f}" for s, n, a, b in per))
    pr = P @ A.T * 100
    say("Mean road cross-section, cm (offset -15..15 m, every 3 m): "
        + " ".join(f"{x:+.1f}" for x in pr.mean(0)[::3]))
    raised = (pr[:, np.abs(OFFS) <= 2].mean(1) > pr[:, (np.abs(OFFS) >= 4) & (np.abs(OFFS) <= 10)].mean(1))
    say(f"Share of road stretches whose centre is higher than its shoulders (raised): {raised.mean():.0%}")

    model = Model().fit(np.r_[features(P), features(N)], np.r_[np.ones(len(P)), np.zeros(len(N))])
    np.savez(OUT / "model.npz", b=model.b, mu=model.mu, sd=model.sd, offs=OFFS)

    roads = json.loads(Path("output/lowry_roads/roads.json").read_text())
    brg = roads["road_A_bearing"]
    tiles = ct.tile_index()
    p0 = np.array(ct.TO_UTM.transform(np.array([LOWRY[1]]), np.array([LOWRY[0]]))).ravel()
    res = {}
    for d in (0,) + CONTROLS:
        if dt.datetime.now().time() >= STOP_AT:
            say("stop time reached")
            break
        b = (brg + d) % 360
        sc, offs, u, v = line_scores(model, tiles, p0, b, ROAD_KM * 1000)
        on, bg, hit = on_line_share(sc, offs)
        pcs = pieces_from(hit, offs)
        res[f"{b:.1f}"] = dict(delta=d, on_line=round(on, 3), background=round(bg, 3),
                               pieces=[[round(a), round(e), round(o)] for a, e, o in pcs])
        per_km = [round(float(hit[int(k * 1000):int(k * 1000) + 1000, np.abs(offs) <= NEAR_M].any(1).mean()), 2) for k in range(int(ROAD_KM))]
        say(f"{'Road A' if d == 0 else 'control':7s} {b:6.1f} deg: on-line {on:.0%} vs background {bg:.0%}; "
            f"per km {per_km}; pieces >= 100 m: {sum(e - a for a, e, _ in pcs):.0f} m {[(round(a), round(e), round(o)) for a, e, o in pcs]}")
        if d == 0:
            keep = (sc, offs, u, v, pcs)
    (OUT / "lowry_road_A.json").write_text(json.dumps(dict(road_A_bearing=brg, lines=res), indent=1))
    ctrl = [r["on_line"] for r in res.values() if r["delta"] != 0]
    if ctrl:
        a = res[f"{brg:.1f}"]["on_line"]
        say(f"Road A on-line {a:.0%}; controls {min(ctrl):.0%}-{max(ctrl):.0%}; Road A beats {sum(a > c for c in ctrl)} of {len(ctrl)}")

    sc, offs, u, v, pcs = keep
    ends = np.array([p0, p0 + u * ROAD_KM * 1000])
    lons, lats = ct.TO_LL.transform(ends[:, 0], ends[:, 1])
    tree = mr.road_tree(mr.fetch("lowry", lats, lons))
    views = [cv.render(tiles, tree, p0, u, v, s0, min(s0 + 1000, ROAD_KM * 1000), pcs,
                       f"Road A from Lowry at {brg:.1f} deg | {s0 / 1000:.0f}-{(s0 + 1000) / 1000:.0f} km | "
                       f"left = pueblo end; detector v3 pieces in red")
             for s0 in np.arange(0, ROAD_KM * 1000 - 50, 1000)]
    views = [im for im in views if im is not None]
    sheet = Image.new("RGB", (max(im.width for im in views), sum(im.height for im in views)), "white")
    y = 0
    for im in views:
        sheet.paste(im, (0, y))
        y += im.height
    sheet.save(OUT / "road_A_view_v3.png", optimize=True)
    (OUT / "report.txt").write_text("\n".join(log) + "\n")


if __name__ == "__main__":
    main()
