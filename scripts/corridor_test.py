"""Corridor test: is there more road-like signal ON site-to-site lines than on control lines?

Usage: python scripts/corridor_test.py output/corridors/great_house_corridors.json

For each corridor A-B, a band BAND_M wide each side of the straight line is resampled
from the 1 m DEM tiles into a strip whose rows run along the line. The road detector's
matched filter (scripts/road_detector.py v2: 5 m dip vs 4-10 m shoulders, averaged
100 m, capped, valley-gated, local z) is applied at the corridor's own bearing only.
A "road stretch" = z >= 2 cells joined along the line (gaps <= 40 m bridged), at least
MIN_RUN_M long and straight (sideways spread <= MAX_SPREAD_M).

Score per line = road metres per km of line with data.
Controls for each corridor (same length, same band, same detector):
  - parallels: the same line shifted sideways by +-1, 2, 3 km (same bearing, so the
    same exposure to N-S/E-W farm grids and same general terrain);
  - rotations: the line turned +-10 and +-20 deg about its midpoint.
If roads survive between sites, real lines should rank above their own controls.
Modern lines (fences, section roads) hit real and control lines alike.

Outputs: output/corridor_test/results.json (incl. all pieces >= PIECE_MIN_M per line), hits.kml (stretches on the real lines),
         summary.txt
"""
import datetime as dt
import json
import os
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform as warp
from scipy import ndimage
from scipy.ndimage import convolve1d, map_coordinates, uniform_filter1d

sys.path.insert(0, str(Path(__file__).parent))
import road_detector as rd                                          # noqa: E402

OUT = Path(os.environ.get("CORRIDOR_OUT", "output/corridor_test"))
BAND_M = 200
EDGE_M = 12
CHUNK_M = 4000
PAD_M = 200                       # along-line padding per chunk (100 m filter + local z)
MIN_RUN_M = 150
PIECE_MIN_M = 100              # pieces saved for every line, for scripts/aligned_test.py
GAP_M = 40
MAX_SPREAD_M = 5.0
SHIFTS_KM = (-3, -2, -1, 1, 2, 3)
ROTATIONS_DEG = (-20, -10, 10, 20)
STOP_AT = dt.time(20, 25)
UTM = "EPSG:26912"


class Transformer:
    """x, y arrays from one CRS to another (rasterio's GDAL transform)."""
    def __init__(self, src, dst):
        self.src, self.dst = src, dst

    def transform(self, x, y):
        xa, ya = np.asarray(x, float), np.asarray(y, float)
        tx, ty = warp(self.src, self.dst, xa.ravel(), ya.ravel())
        if xa.ndim == 0:
            return round(tx[0], 6), round(ty[0], 6)
        return np.reshape(tx, xa.shape), np.reshape(ty, ya.shape)


TO_UTM = Transformer("EPSG:4326", UTM)
TO_LL = Transformer(UTM, "EPSG:4326")


def tile_index():
    """All cached tiles, best first (CO 2018 > UT 2020 > others; merged Hovenweep first)."""
    def rank(p):
        n = p.name
        return (0 if "merged" in str(p.parent) else 1 if "CO_Southwest" in n else 2 if "2020" in n else 3, n)
    seen, out = set(), []
    for p in sorted(Path("output/3dep").rglob("*.tif"), key=rank):
        if p.name in seen:
            continue
        seen.add(p.name)
        with rasterio.open(p) as s:
            crs = s.crs.to_string()
            tr = Transformer(UTM, crs)
            out.append(dict(path=str(p), crs=crs, bounds=tuple(s.bounds), nodata=s.nodata,
                            utm=(s.crs.to_epsg() == 26912), tr=tr))
    return out


def sample(tiles, x, y):
    """Bilinear DEM values at UTM points x, y (2-D arrays); NaN where no tile has data."""
    z = np.full(x.shape, np.nan)
    for t in tiles:
        need = np.isnan(z)
        if not need.any():
            break
        tx, ty = (x, y) if t["utm"] else t["tr"].transform(x, y)
        l, b, r, tp = t["bounds"]
        inside = need & (tx > l + 1) & (tx < r - 1) & (ty > b + 1) & (ty < tp - 1)
        if not inside.any():
            continue
        with rasterio.open(t["path"]) as s:
            col, row = ~s.transform * (tx[inside], ty[inside])
            r0, c0 = max(int(row.min()) - 2, 0), max(int(col.min()) - 2, 0)
            r1, c1 = min(int(row.max()) + 3, s.height), min(int(col.max()) + 3, s.width)
            arr = s.read(1, window=((r0, r1), (c0, c1))).astype(np.float64)
        bad = arr == t["nodata"]
        if bad.all():
            continue
        v = map_coordinates(np.where(bad, 0, arr), [row - r0 - 0.5, col - c0 - 0.5], order=1, mode="nearest")
        nb = map_coordinates(bad.astype(float), [row - r0 - 0.5, col - c0 - 0.5], order=1, mode="nearest")
        v[nb > 0.01] = np.nan
        zz = z[inside]
        zz[np.isnan(zz)] = v[np.isnan(zz)]
        z[inside] = zz
    return z


def line_strip_z(tiles, p0, p1):
    """Detector z-score strip (rows = along line, cols = sideways offset) for line p0->p1 (UTM)."""
    L = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    u = np.array([(p1[0] - p0[0]) / L, (p1[1] - p0[1]) / L])     # along
    v = np.array([u[1], -u[0]])                                   # sideways
    k = rd.cross_kernel()
    n = int(rd.LENGTH_M)
    offs = np.arange(-BAND_M, BAND_M + 1, 1.0)
    zs = []
    for s0 in np.arange(0, L, CHUNK_M):
        s1 = min(s0 + CHUNK_M, L)
        ss = np.arange(s0 - PAD_M, s1 + PAD_M, 1.0)
        S, T = np.meshgrid(ss, offs, indexing="ij")
        x = p0[0] + S * u[0] + T * v[0]
        y = p0[1] + S * u[1] + T * v[1]
        dem = sample(tiles, x, y)
        ok = ~np.isnan(dem)
        if ok.mean() < 0.05:
            zs.append(np.full((int(s1 - s0), offs.size), np.nan, np.float32))
            continue
        dem = np.where(ok, dem, np.nanmean(dem))
        dip = convolve1d(dem, k, axis=1, mode="nearest")
        den = uniform_filter1d(ok.astype(float), n, axis=0)
        raw = uniform_filter1d(np.where(ok, dip, 0), n, axis=0) / np.maximum(den, 1e-9)
        capped = uniform_filter1d(np.where(ok, np.clip(dip, -rd.CAP_M, rd.CAP_M), 0), n, axis=0) / np.maximum(den, 1e-9)
        along = np.where((den > 0.9) & (raw < rd.VALLEY_M) & ok, capped, np.nan)
        z = rd.local_z(along).astype(np.float32)
        z[:, np.abs(offs) > BAND_M - EDGE_M] = np.nan          # filters are unreliable at the band edge
        a = int(PAD_M)
        zs.append(z[a:a + int(s1 - s0)])
    return np.concatenate(zs), L


def stretches(z, min_run=MIN_RUN_M):
    """Straight z >= 2 runs along the strip: list of (start_m, end_m, offset_m)."""
    m = np.nan_to_num(z, nan=0) >= 2
    m = ndimage.binary_closing(m, structure=np.ones((GAP_M + 1, 1))) | m
    lab, _ = ndimage.label(m, structure=np.ones((3, 3)))
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl[0].stop - sl[0].start < min_run:
            continue
        r, c = np.where(lab[sl] == i)
        # one offset per along-position (mean), its spread around a straight fit
        rows = np.unique(r)
        cm = ndimage.mean(c, r, rows)
        fit = np.polyfit(rows, cm, 1)
        if abs(fit[0]) > 0.05 or np.std(cm - np.polyval(fit, rows)) > MAX_SPREAD_M:
            continue                          # drifts or wanders: not straight along this bearing
        out.append((int(sl[0].start), int(sl[0].stop), float(cm.mean() + sl[1].start - BAND_M)))
    return out


def endpoints(a, b):
    xa, ya = TO_UTM.transform(a["lon"], a["lat"])
    xb, yb = TO_UTM.transform(b["lon"], b["lat"])
    return np.array([xa, ya]), np.array([xb, yb])


def control_lines(p0, p1):
    L = np.linalg.norm(p1 - p0)
    u = (p1 - p0) / L
    v = np.array([u[1], -u[0]])
    out = [(f"shift {s:+d} km", p0 + v * s * 1000, p1 + v * s * 1000) for s in SHIFTS_KM]
    mid = (p0 + p1) / 2
    for d in ROTATIONS_DEG:
        c, s_ = math.cos(math.radians(d)), math.sin(math.radians(d))
        R = np.array([[c, -s_], [s_, c]])
        out.append((f"rotate {d:+d} deg", mid + R @ (p0 - mid), mid + R @ (p1 - mid)))
    return out


def run_line(job):
    name, kind, p0, p1 = job
    if dt.datetime.now().time() >= STOP_AT:
        return dict(corridor=name, line=kind, skipped="stop time")
    tiles = tile_index()
    z, L = line_strip_z(tiles, p0, p1)
    have = ~np.all(np.isnan(z), axis=1)
    km = have.sum() / 1000
    pieces = stretches(z, PIECE_MIN_M)
    st = [p for p in pieces if p[1] - p[0] >= MIN_RUN_M]
    road_m = sum(e - s for s, e, _ in st)
    res = dict(corridor=name, line=kind, length_km=round(L / 1000, 2), km_with_data=round(km, 2),
               stretches=len(st), road_m=road_m, road_m_per_km=round(road_m / km, 2) if km > 1 else None,
               longest_m=max((e - s for s, e, _ in st), default=0),
               pieces=[[s, e, round(o, 1)] for s, e, o in pieces])
    if kind == "real":
        u = (p1 - p0) / L
        v = np.array([u[1], -u[0]])
        res["hits"] = [dict(from_m=s, to_m=e, offset_m=round(o, 1),
                            a=TO_LL.transform(*(p0 + u * s + v * o)), b=TO_LL.transform(*(p0 + u * e + v * o)))
                       for s, e, o in st]
    print(f"{name} | {kind}: {res['km_with_data']} km with data, {len(st)} stretches, "
          f"{res['road_m_per_km']} m/km", flush=True)
    return res


def main():
    pairs = json.loads(Path(sys.argv[1]).read_text())
    sites = {}
    for f in ("sites_chacoan.json", "great_houses.json", "sites.json", "sites_local.json"):
        for s in json.loads((Path("output/corridors") / f).read_text()):
            sites.setdefault(s["name"], s)
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for p in pairs:
        name = f"{p['a']} - {p['b']}"
        p0, p1 = endpoints(sites[p["a"]], sites[p["b"]])
        jobs.append((name, "real", p0, p1))
        jobs += [(name, kind, a, b) for kind, a, b in control_lines(p0, p1)]
    print(f"{len(pairs)} corridors, {len(jobs)} lines", flush=True)
    with Pool(2) as pool:
        results = pool.map(run_line, jobs, chunksize=1)
    (OUT / "results.json").write_text(json.dumps(results, indent=1))

    lines, ranks = [], []
    for p in pairs:
        name = f"{p['a']} - {p['b']}"
        rs = [r for r in results if r["corridor"] == name and r.get("road_m_per_km") is not None]
        real = [r for r in rs if r["line"] == "real"]
        if not real:
            lines.append(f"{name}: no data on the real line")
            continue
        ctrl = sorted(r["road_m_per_km"] for r in rs if r["line"] != "real")
        rv = real[0]["road_m_per_km"]
        beat = sum(c < rv for c in ctrl) + 0.5 * sum(c == rv for c in ctrl)
        ranks.append((beat, len(ctrl)))
        lines.append(f"{name}: real {rv} m/km ({real[0]['stretches']} stretches, {real[0]['km_with_data']} km) | "
                     f"controls median {np.median(ctrl):.1f}, range {ctrl[0]}-{ctrl[-1]} | beats {beat:g} of {len(ctrl)}")
    if ranks:
        share = sum(b for b, _ in ranks) / sum(n for _, n in ranks)
        # chance level is 0.5; permutation p from shuffling the real line among its controls
        rng = np.random.default_rng(0)
        sims = [np.mean([rng.integers(0, n + 1) / n for _, n in ranks]) for _ in range(20000)]
        obs = np.mean([b / n for b, n in ranks])
        pval = float(np.mean(np.array(sims) >= obs))
        lines.append(f"\nreal lines beat {share:.0%} of their controls (chance = 50%); one-sided p = {pval:.3f}")
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)

    pm = []
    for r in results:
        for h in r.get("hits", []):
            pm.append(f"<Placemark><name>{r['corridor']} {h['from_m']}-{h['to_m']} m</name>"
                      f"<description>{h['to_m'] - h['from_m']} m long, {h['offset_m']} m off the line</description>"
                      f"<styleUrl>#hit</styleUrl><LineString><coordinates>{h['a'][0]},{h['a'][1]},0 "
                      f"{h['b'][0]},{h['b'][1]},0</coordinates></LineString></Placemark>")
    (OUT / "hits.kml").write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document><name>Road-like stretches on great-house corridors</name>
<Style id="hit"><LineStyle><color>ff00ffff</color><width>4</width></LineStyle></Style>
{chr(10).join(pm)}
</Document></kml>
""")


if __name__ == "__main__":
    main()
