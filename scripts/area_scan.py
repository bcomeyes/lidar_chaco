"""Scan the ground around Lowry Pueblo with detector v3, on unplowed land only, and see whether
the documentary's roads (B, C, D, and A) come out without being pointed to.

Area: 7 x 7 km of 1 m 3DEP centred on Lowry. Every 5 deg bearing (0-175), every 1 m: the v3
score of the 100 m stretch centred there (scripts/learn_detector.py, output/detector_v3/model.npz).
Hit = score above the 99th percentile of scores on unplowed ground. Segment = hits held along one
bearing for >= 150 m (gaps <= 20 m bridged); near-duplicates (<= 10 m apart, <= 10 deg) merged.

Unplowed = NLCD 2021 forest, shrub, grass, wetland (cropland/pasture buffered 30 m, developed
land and mapped OSM roads within 10 m are left out). Fields abandoned before 2021 may still count
as unplowed.

Checks:
  drawn roads: segment metres within 25 m and 15 deg of each traced road (roads.json), vs the
               same traces turned around Lowry in 30 deg steps
  radial:      segments whose line passes within 50 m of Lowry, vs the same count for 200 random
               points in the area

Outputs: output/area_scan_lowry/ (segments.geojson, segments.kml, overview.png, lowry_zoom.png, report.txt)
"""
import datetime as dt
import json
import math
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw
from rasterio.warp import Resampling, reproject
from scipy import ndimage
from scipy.ndimage import map_coordinates, uniform_filter1d

import corridor_test as ct
import learn_detector as ld
import mapped_roads as mr

OUT = Path("output/area_scan_lowry")
R_M = 3500
BEARINGS = np.arange(0, 180, 5)
STRIP = 200
ROW_CHUNK = 2000
HIT_PCT = 99.0
MIN_SEG_M = 150
GAP_M = 20
NATURAL = (31, 41, 42, 43, 52, 71, 90, 95)
STOP_AT = dt.time(20, 15)

G = {}                                         # shared with the workers (fork)


def load():
    cx, cy = (float(a[0]) for a in ct.TO_UTM.transform(np.array([ld.LOWRY[1]]), np.array([ld.LOWRY[0]])))
    x0, y1 = cx - R_M, cy + R_M
    n = 2 * R_M
    tiles = ct.tile_index()
    dem = np.full((n, n), np.nan, np.float32)
    cols = x0 + 0.5 + np.arange(n)
    for r0 in range(0, n, 500):
        rows = y1 - 0.5 - np.arange(r0, min(r0 + 500, n))
        X, Y = np.meshgrid(cols, rows)
        dem[r0:r0 + rows.size] = ct.sample(tiles, X, Y)
    tr = rasterio.Affine(1, 0, x0, 0, -1, y1)
    with rasterio.open("output/nlcd/lowry_nlcd2021.tif") as s:
        lc = np.zeros((n, n), np.uint8)
        reproject(s.read(1), lc, src_transform=s.transform, src_crs=s.crs, dst_transform=tr,
                  dst_crs=ct.UTM, resampling=Resampling.nearest)
    farmed = np.isin(lc, (81, 82))
    ok = np.isin(lc, NATURAL) & ~(ndimage.distance_transform_edt(~farmed) <= 30)
    ends = np.array([[x0, y1 - n], [x0 + n, y1]])
    lons, lats = ct.TO_LL.transform(ends[:, 0], ends[:, 1])
    tree = mr.road_tree(mr.fetch("lowry_area", lats, lons))
    for r0 in range(0, n, 500):
        rows = y1 - 0.5 - np.arange(r0, min(r0 + 500, n))
        X, Y = np.meshgrid(cols, rows)
        d, _ = tree.query(np.c_[X.ravel(), Y.ravel()])
        ok[r0:r0 + rows.size] &= (d > 10).reshape(X.shape)
    return dem, ok & ~np.isnan(dem), lc, tr, (cx, cy), tree


def weights():
    m = np.load("output/detector_v3/model.npz")
    b, mu, sd = m["b"], m["mu"], m["sd"]
    k = ld.OFFS.size
    w = b[1:] / sd
    return (np.float32(b[0] - np.sum(mu / sd * b[1:])), (ld.A.T @ w[:k] * 100).astype(np.float32),
            (w[k:] / 10 * 1e4).astype(np.float32))                          # p below is in metres


def bearing_strips(b, every=1):
    """Yield (S, T, score) for strips of the rotated grid at bearing b."""
    dem, ok, (x0, y1) = G["dem"], G["ok"], G["origin"]
    c0, w1, w2 = G["w"]
    cx, cy = G["centre"]
    u = np.array([math.sin(math.radians(b)), math.cos(math.radians(b))])
    v = np.array([u[1], -u[0]])
    half = int(R_M * math.sqrt(2)) + 1
    ss = np.arange(-half, half, 1.0)
    for k, t0 in enumerate(range(-half, half, STRIP)):
        if k % every:
            continue
        tt = np.arange(t0 - ld.HALF_W, t0 + STRIP + ld.HALF_W, 1.0)
        S, T = np.meshgrid(ss, tt, indexing="ij")
        X, Y = cx + S * u[0] + T * v[0], cy + S * u[1] + T * v[1]
        rc = [y1 - Y - 0.5, X - x0 - 0.5]
        z = map_coordinates(dem, rc, order=1, cval=np.nan, mode="constant")
        good = map_coordinates(ok.astype(np.float32), rc, order=0, cval=0, mode="constant") > 0.5
        valid = ~np.isnan(z)
        if valid.mean() < 0.01:
            continue
        z = np.where(valid, z, np.nanmean(z)).astype(np.float32)
        avg = uniform_filter1d(z, ld.ALONG, axis=0)
        full = uniform_filter1d(valid.astype(np.float32), ld.ALONG, axis=0) > 0.95
        win = np.lib.stride_tricks.sliding_window_view(avg, ld.OFFS.size, axis=1)
        sc = np.empty(win.shape[:2], np.float32)
        for r0 in range(0, win.shape[0], ROW_CHUNK):
            wv = win[r0:r0 + ROW_CHUNK]
            p = wv @ ld.A.T.astype(np.float32)
            sc[r0:r0 + ROW_CHUNK] = c0 + wv @ w1 + (p * p) @ w2
        c = slice(ld.HALF_W, ld.HALF_W + sc.shape[1])
        keep = full[:, c] & good[:, c]
        yield S[:, c], T[:, c], np.where(keep, sc, np.nan)


def sample_scores(b):
    out = []
    rng = np.random.default_rng(int(b))
    for _, _, sc in bearing_strips(b, every=6):
        v = sc[~np.isnan(sc)]
        if v.size:
            out.append(rng.choice(v, min(v.size, 20000), replace=False))
    return np.concatenate(out) if out else np.array([])


def segments(b):
    if dt.datetime.now().time() >= STOP_AT:
        return b, None
    thr = G["thr"]
    cx, cy = G["centre"]
    u = np.array([math.sin(math.radians(b)), math.cos(math.radians(b))])
    v = np.array([u[1], -u[0]])
    segs = []
    for S, T, sc in bearing_strips(b):
        hit = np.nan_to_num(sc, nan=-np.inf) > thr
        hold = uniform_filter1d(hit.astype(np.float32), ld.ALONG, axis=0) > 0.5
        hold = ndimage.binary_closing(hold, np.ones((GAP_M + 1, 1))) | hold
        for j in np.nonzero(hold.any(0))[0]:
            lab, n = ndimage.label(hold[:, j])
            for sl in ndimage.find_objects(lab):
                a, e = sl[0].start, sl[0].stop
                if e - a < MIN_SEG_M:
                    continue
                s0, s1, t = S[a, j], S[e - 1, j], T[a, j]
                p0 = np.array([cx, cy]) + s0 * u + t * v
                p1 = np.array([cx, cy]) + s1 * u + t * v
                segs.append((float(b), *p0.tolist(), *p1.tolist(), float(e - a), float(np.nanmean(sc[a:e, j]))))
    return b, segs


def merge(segs):
    """Keep the longest of near-duplicates (<= 10 m apart, <= 10 deg, overlapping)."""
    segs = sorted(segs, key=lambda s: -s[5])
    kept = []
    for s in segs:
        b, x0, y0, x1, y1, L, sc = s
        mid = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        dup = False
        for k in kept:
            db = abs((b - k[0] + 90) % 180 - 90)
            if db > 10:
                continue
            p, q = np.array(k[1:3]), np.array(k[3:5])
            d = q - p
            tpar = np.dot(mid - p, d) / np.dot(d, d)
            if -0.25 <= tpar <= 1.25 and abs(d[0] * (mid - p)[1] - d[1] * (mid - p)[0]) / np.linalg.norm(d) <= 10:
                dup = True
                break
        if not dup:
            kept.append(s)
    return kept


def seg_points(s, step=5):
    p, q = np.array(s[1:3]), np.array(s[3:5])
    k = max(int(s[5] / step), 1)
    return p + np.linspace(0, 1, k + 1)[:, None] * (q - p)


def matched_length(segs, traces):
    """Segment metres within 25 m and 15 deg of any traced road (traces: list of UTM polylines)."""
    tot = 0.0
    for s in segs:
        pts = seg_points(s)
        ok = np.zeros(len(pts), bool)
        for tr_ in traces:
            for a, c in zip(tr_[:-1], tr_[1:]):
                d = c - a
                L = np.linalg.norm(d)
                if L < 1:
                    continue
                db = abs((s[0] - math.degrees(math.atan2(d[0], d[1])) + 90) % 180 - 90)
                if db > 15:
                    continue
                tpar = np.clip(((pts - a) @ d) / L ** 2, 0, 1)
                dist = np.linalg.norm(pts - (a + tpar[:, None] * d), axis=1)
                ok |= dist <= 25
        tot += s[5] * ok.mean()
    return tot


def points_at(segs, p, r=50):
    n = 0
    for s in segs:
        a, c = np.array(s[1:3]), np.array(s[3:5])
        d = (c - a) / np.linalg.norm(c - a)
        if abs(d[0] * (p - a)[1] - d[1] * (p - a)[0]) <= r:
            n += 1
    return n


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log = []

    def say(s):
        print(s, flush=True)
        log.append(s)

    dem, ok, lc, tr, centre, tree = load()
    G.update(dem=dem, ok=ok, origin=(tr.c, tr.f), centre=centre, w=weights())
    say(f"Area {2 * R_M / 1000:.0f} x {2 * R_M / 1000:.0f} km: data on {np.mean(~np.isnan(dem)):.0%}, "
        f"unplowed + usable {ok.mean():.0%} (NLCD cropland {np.isin(lc, (81, 82)).mean():.0%})")
    with Pool(3) as pool:
        smp = np.concatenate(pool.map(sample_scores, BEARINGS[::6]))
        G["thr"] = float(np.percentile(smp, HIT_PCT))
        say(f"Hit threshold (99th pct of {smp.size} sampled scores on unplowed ground): {G['thr']:.2f}")
    with Pool(3) as pool:                                                  # fresh workers see the threshold
        res = pool.map(segments, BEARINGS)
    raw = [s for b, ss in res if ss for s in ss]
    done = [b for b, ss in res if ss is not None]
    say(f"Bearings done: {len(done)} of {len(BEARINGS)}; raw segments {len(raw)}")
    segs = merge(raw)
    say(f"Segments after merging: {len(segs)} (>= 300 m: {sum(s[5] >= 300 for s in segs)}, "
        f">= 500 m: {sum(s[5] >= 500 for s in segs)}); total {sum(s[5] for s in segs) / 1000:.1f} km")

    # drawn roads vs the same traces turned around Lowry
    roads = json.loads(Path("output/lowry_roads/roads.json").read_text())["roads"]
    cxy = np.array(centre)
    traces = {}
    for name, items in roads.items():
        for it in items:
            ll = np.array(it["lonlat"])
            x, y = ct.TO_UTM.transform(ll[:, 0], ll[:, 1])
            traces.setdefault(name, []).append(np.c_[x, y])
    brg = json.loads(Path("output/lowry_roads/roads.json").read_text())["road_A_bearing"]
    ua = np.array([math.sin(math.radians(brg)), math.cos(math.radians(brg))])
    traces["A ray 0-3 km"] = [np.array([cxy, cxy + ua * 3000])]

    def turned(polys, deg):
        a = math.radians(deg)
        R = np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]])
        return [(p - cxy) @ R.T + cxy for p in polys]

    for name, polys in traces.items():
        real = matched_length(segs, polys)
        ctrl = [matched_length(segs, turned(polys, d)) for d in range(30, 360, 30)]
        tl = sum(np.linalg.norm(np.diff(p, axis=0), axis=1).sum() for p in polys) / len(polys)
        say(f"Drawn road {name:32s} (~{tl:.0f} m): {real:.0f} m of segments on it; turned copies "
            f"{min(ctrl):.0f}-{max(ctrl):.0f} m (median {np.median(ctrl):.0f}); beats {sum(real > c for c in ctrl)} of 11")

    rng = np.random.default_rng(3)
    rand = cxy + rng.uniform(-R_M + 500, R_M - 500, (200, 2))
    n_low = points_at(segs, cxy)
    n_rand = np.array([points_at(segs, p) for p in rand])
    say(f"Radial: {n_low} segments point at Lowry (line within 50 m); random points median "
        f"{np.median(n_rand):.0f}, 95th pct {np.percentile(n_rand, 95):.0f}; Lowry beats {np.mean(n_low > n_rand):.0%}")

    # outputs
    feats, pm = [], []
    for s in segs:
        (lo0, lo1), (la0, la1) = ct.TO_LL.transform(np.array([s[1], s[3]]), np.array([s[2], s[4]]))
        feats.append(dict(type="Feature", properties=dict(bearing=s[0], length_m=round(s[5]), score=round(s[6], 2)),
                          geometry=dict(type="LineString", coordinates=[[lo0, la0], [lo1, la1]])))
        pm.append(f"<Placemark><name>{s[5]:.0f} m, {s[0]:.0f} deg</name><styleUrl>#seg</styleUrl><LineString>"
                  f"<coordinates>{lo0:.6f},{la0:.6f} {lo1:.6f},{la1:.6f}</coordinates></LineString></Placemark>")
    (OUT / "segments.geojson").write_text(json.dumps(dict(type="FeatureCollection", features=feats)))
    dr = []
    for name, polys in traces.items():
        for p in polys:
            lo, la = ct.TO_LL.transform(p[:, 0], p[:, 1])
            dr.append(f"<Placemark><name>drawn {name}</name><styleUrl>#drawn</styleUrl><LineString><coordinates>"
                      + " ".join(f"{a:.6f},{b:.6f}" for a, b in zip(lo, la)) + "</coordinates></LineString></Placemark>")
    (OUT / "segments.kml").write_text(
        "<?xml version='1.0' encoding='UTF-8'?><kml xmlns='http://www.opengis.net/kml/2.2'><Document>"
        "<name>Lowry area scan (detector v3, unplowed only)</name>"
        "<Style id='seg'><LineStyle><color>ff0000ff</color><width>3</width></LineStyle></Style>"
        "<Style id='drawn'><LineStyle><color>ffffaa00</color><width>2</width></LineStyle></Style>"
        "<Folder><name>Detected segments (red)</name>" + "".join(pm) + "</Folder>"
        "<Folder><name>Documentary roads (blue)</name>" + "".join(dr) + "</Folder></Document></kml>")

    gy, gx = np.gradient(np.nan_to_num(dem, nan=float(np.nanmean(dem))))
    hs = np.clip(0.55 + (-gx * 0.7 + gy * 0.7) * 1.5, 0, 1).astype(np.float32)
    del gx, gy
    rgb = np.stack([hs * 255] * 3, -1)
    rgb[~ok] = rgb[~ok] * 0.6 + np.array([90, 60, 20]) * 0.4                # not scanned: brown tint
    img = Image.fromarray(rgb.astype(np.uint8))

    def draw(im, scale, off):
        d = ImageDraw.Draw(im)
        px = lambda x, y: ((x - tr.c - off[0]) * scale, (tr.f - y - off[1]) * scale)
        for polys in traces.values():
            for p in polys:
                d.line([px(*q) for q in p], fill=(0, 140, 255), width=3)
        for s in segs:
            d.line([px(s[1], s[2]), px(s[3], s[4])], fill=(230, 0, 0), width=3 if s[5] < 300 else 5)
        d.ellipse([*(np.array(px(*cxy)) - 8), *(np.array(px(*cxy)) + 8)], outline=(0, 200, 0), width=3)

    ov = img.resize((2 * R_M // 2, 2 * R_M // 2), Image.BILINEAR)
    draw(ov, 0.5, (0, 0))
    ov.save(OUT / "overview.png", optimize=True)
    z0 = int(R_M - 1000)
    zoom = img.crop((z0, z0, z0 + 2000, z0 + 2000))
    draw(zoom, 1.0, (z0, z0))
    zoom.save(OUT / "lowry_zoom.png", optimize=True)
    (OUT / "report.txt").write_text("\n".join(log) + "\n")
    say("done")


if __name__ == "__main__":
    main()
