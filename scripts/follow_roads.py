"""Follow confident road lines into fainter ground, the way an eye traces a road.

Start from each chain >= SEED_MIN_M (scripts/chain_roads.py). From each end, step
STEP_M at a time along the chain's own bearing. At each step, measure the dip in a
WINDOW_M stretch ahead (same 5 m centre / 4-10 m shoulders as the detector, each slice
capped at 15 cm), allowing the line to re-centre up to SHIFT_M sideways.
A step passes if its dip beats the PASS_PCT percentile of the same measurement at random
spots and bearings (the noise bar). Up to GAP_M of failing steps is bridged; after that
the road "faded" at the last passing step. Reaching the data edge is marked "edge".

Roads are never judged on where they go. Every end is kept as a point to inspect.

Inputs : output/output_be.tif, output/chains/road_chains.geojson,
         output/ground_truth/fig11_roads.tif (scoring only)
Outputs: output/follow/roads.geojson, road_ends.geojson, scores.json, overlay*.jpg
"""
import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw
from scipy import ndimage

from road_detector import load_1m, CAP_M, VALLEY_M

OUT = Path("output/follow")
SEED_MIN_M = 200
STEP_M = 10
WINDOW_M = 30
SHIFT_M = 2
GAP_M = 40
PASS_PCT = 90
MAX_EXTEND_M = 3000
OFFS = np.arange(-10, 10.01, 1.0)
CENTRE = np.abs(OFFS) <= 2.5
SHOULDER = (np.abs(OFFS) >= 4) & (np.abs(OFFS) <= 10)


def window_dip(dem, p, u, shift):
    """Capped mean dip (m) over WINDOW_M ahead of point p (row, col) along unit u."""
    n = np.array([-u[1], u[0]])
    t = np.arange(0, WINDOW_M, 1.0)
    rr = p[0] + t[:, None] * u[0] + (OFFS[None, :] + shift) * n[0]
    cc = p[1] + t[:, None] * u[1] + (OFFS[None, :] + shift) * n[1]
    if rr.min() < 0 or cc.min() < 0 or rr.max() >= dem.shape[0] - 1 or cc.max() >= dem.shape[1] - 1:
        return None, None
    z = ndimage.map_coordinates(dem, [rr, cc], order=1)
    dip = z[:, SHOULDER].mean(1) - z[:, CENTRE].mean(1)          # per slice
    return float(np.clip(dip, -CAP_M, CAP_M).mean()), float(dip.mean())


def best_dip(dem, valid, p, u):
    best = (None, None, 0.0)
    for s in np.arange(-SHIFT_M, SHIFT_M + 0.1, 1.0):
        capped, raw = window_dip(dem, p, u, s)
        if capped is None:
            return None, None, 0.0
        if raw >= VALLEY_M:
            continue
        if best[0] is None or capped > best[0]:
            best = (capped, raw, s)
    return best


def noise_bar(dem, valid, rng, n=4000):
    vals = []
    while len(vals) < n:
        p = np.array([rng.uniform(20, dem.shape[0] - 40), rng.uniform(20, dem.shape[1] - 40)])
        if not valid[int(p[0]), int(p[1])]:
            continue
        a = rng.uniform(0, np.pi)
        d, _, _ = best_dip(dem, valid, p, np.array([np.cos(a), np.sin(a)]))
        if d is not None:
            vals.append(d)
    return float(np.percentile(vals, PASS_PCT))


def follow(dem, valid, start, u, bar):
    """Walk from start along u. Returns (last passing point, how it ended, metres added)."""
    p = start.copy()
    last_ok, gap, walked = start.copy(), 0.0, 0.0
    n = np.array([-u[1], u[0]])
    while walked < MAX_EXTEND_M:
        d, _, s = best_dip(dem, valid, p, u)
        if d is None or not valid[int(p[0]), int(p[1])]:
            return last_ok, "edge", float(np.linalg.norm(last_ok - start))
        if d >= bar:
            p = p + s * n * 0.5                      # gentle re-centre
            last_ok, gap = p.copy(), 0.0
        else:
            gap += STEP_M
            if gap > GAP_M:
                return last_ok, "faded", float(np.linalg.norm(last_ok - start))
        p = p + STEP_M * u
        walked += STEP_M
    return last_ok, "max_length", float(np.linalg.norm(last_ok - start))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    dem, valid, prof = load_1m()
    inv = ~prof["transform"]
    bar = noise_bar(dem, valid, rng)
    print(f"noise bar: a step passes if its dip >= {bar * 100:.1f} cm "
          f"(beats {PASS_PCT}% of random ground)", flush=True)

    chains = json.loads(Path("output/chains/road_chains.geojson").read_text())["features"]
    seeds = [f for f in chains if f["properties"]["length_m"] >= SEED_MIN_M]
    roads, ends = [], []
    for f in seeds:
        (xa, ya), (xb, yb) = f["geometry"]["coordinates"]
        ca, ra = inv * (xa, ya)
        cb, rb = inv * (xb, yb)
        a, b = np.array([ra, ca]) - 0.5, np.array([rb, cb]) - 0.5
        u = (b - a) / np.linalg.norm(b - a)
        new_b, how_b, add_b = follow(dem, valid, b, u, bar)
        new_a, how_a, add_a = follow(dem, valid, a, -u, bar)
        props = dict(f["properties"], seed_length_m=f["properties"]["length_m"],
                     length_m=round(float(np.linalg.norm(new_b - new_a)), 1),
                     added_m=round(add_a + add_b, 1), end_a=how_a, end_b=how_b)
        roads.append((new_a, new_b, props))

    t = prof["transform"]
    xy = lambda rc: [round(v, 2) for v in (t * (rc[1] + 0.5, rc[0] + 0.5))]
    crs = {"type": "name", "properties": {"name": "EPSG:6342"}}
    feats = [dict(type="Feature", properties=p, geometry=dict(type="LineString", coordinates=[xy(a), xy(b)]))
             for a, b, p in roads]
    for a, b, p in roads:
        for e, rc in (("a", a), ("b", b)):
            ends.append(dict(type="Feature", properties=dict(road_id=p["id"], ended=p[f"end_{e}"],
                                                             length_m=p["length_m"]),
                             geometry=dict(type="Point", coordinates=xy(rc))))
    (OUT / "roads.geojson").write_text(json.dumps(dict(type="FeatureCollection", crs=crs, features=feats)))
    (OUT / "road_ends.geojson").write_text(json.dumps(dict(type="FeatureCollection", crs=crs, features=ends)))

    # score inside the Fig. 11 footprint, before vs after following
    with rasterio.open("output/ground_truth/fig11_roads.tif") as s:
        g = s.read(1)
    h, w = dem.shape
    gt = g[:h * 2, :w * 2].reshape(h, 2, w, 2).max(axis=(1, 3))
    road, area = gt == 2, gt >= 1
    core = road & ndimage.binary_erosion(road, iterations=3)
    marked_near = ndimage.binary_dilation(road, iterations=8)

    def draw(lines):
        im = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(im)
        for a, b in lines:
            d.line([(a[1], a[0]), (b[1], b[0])], fill=255, width=5)
        return np.asarray(im) > 0

    report = dict(noise_bar_cm=round(bar * 100, 2), seeds=len(seeds),
                  settings=dict(seed_min_m=SEED_MIN_M, step_m=STEP_M, window_m=WINDOW_M,
                                shift_m=SHIFT_M, gap_m=GAP_M, pass_pct=PASS_PCT))
    seed_lines = []
    for f in seeds:
        (xa, ya), (xb, yb) = f["geometry"]["coordinates"]
        ca, ra = inv * (xa, ya)
        cb, rb = inv * (xb, yb)
        seed_lines.append((np.array([ra, ca]), np.array([rb, cb])))
    for tag, lines in (("before", seed_lines), ("after", [(a, b) for a, b, _ in roads])):
        m = draw(lines) & area
        prec = (m & marked_near).sum() / max(m.sum(), 1)
        rec = (core & ndimage.binary_dilation(m, iterations=8)).sum() / core.sum()
        report[tag] = dict(precision=round(float(prec), 3), recall=round(float(rec), 3))
        print(f"{tag:6}: {prec:.0%} of detected line lies on marked roads; "
              f"{rec:.0%} of marked road length found", flush=True)
    kinds = [e["properties"]["ended"] for e in ends]
    report["ends"] = {k: kinds.count(k) for k in set(kinds)}
    print("ends:", report["ends"], flush=True)
    (OUT / "scores.json").write_text(json.dumps(report, indent=2))

    # overlay: seed part orange, followed extension green, ends yellow (faded) / red (edge)
    rows, cols = np.where(area)
    r0, c0 = rows.min(), cols.min()
    with rasterio.open("output/detector/road_z.tif") as s:
        z = s.read(1)
    base = (np.clip(np.nan_to_num(z, nan=0) / 4, 0, 1) * 90)[r0:rows.max(), c0:cols.max()]
    rgb = np.stack([base] * 3, -1).astype(np.uint8)
    rgb[(road & ~ndimage.binary_erosion(road))[r0:rows.max(), c0:cols.max()]] = [0, 200, 255]
    im = Image.fromarray(rgb)
    d = ImageDraw.Draw(im)
    for (a, b, p), (sa, sb) in zip(roads, seed_lines):
        d.line([(a[1] - c0, a[0] - r0), (b[1] - c0, b[0] - r0)], fill=(60, 220, 60), width=3)
        d.line([(sa[1] - c0, sa[0] - r0), (sb[1] - c0, sb[0] - r0)], fill=(255, 140, 0), width=3)
        for e, rc in (("a", a), ("b", b)):
            col = (255, 255, 0) if p[f"end_{e}"] == "faded" else (255, 60, 60)
            d.ellipse([rc[1] - c0 - 6, rc[0] - r0 - 6, rc[1] - c0 + 6, rc[0] - r0 + 6], outline=col, width=2)
    im.thumbnail((1600, 1600))
    im.save(OUT / "overlay_fig11_area.jpg", quality=88)


if __name__ == "__main__":
    main()
