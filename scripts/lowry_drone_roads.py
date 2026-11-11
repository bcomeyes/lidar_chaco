"""Lowry Pueblo roads from the PBS documentary drone frames (output/book/chacopictures).

The drone circles Lowry Pueblo with the roads drawn in as pale paths. To turn those
into map lines:
1. Camera fit: the true skyline seen from above Lowry is computed from the 30 m DEM
   (output/dem30, earth curvature + refraction). Each frame's photographed skyline is
   fitted to it, giving heading, tilt, roll and lens (focal length in px). Every
   direction is tried; the best fit must beat the next-best clearly.
2. Roads: points traced by hand on each frame (crop pixels below) are projected onto
   flat ground (drone height H_M; bearings do not depend on it, distances scale with it).
3. Test: our road detector is run along Road A (the long straight road), and a
   side-lit strip is drawn for the eye (scripts/corridor_views.py panels).

Outputs: output/lowry_roads/cameras.json, roads.json, lowry_roads.kml, road_A_view.png,
         skyline_check.png
"""
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).parent))
import corridor_test as ct                                          # noqa: E402
import corridor_views as cv                                         # noqa: E402
import mapped_roads as mr                                           # noqa: E402

FRAMES = Path("output/book/chacopictures")
OUT = Path("output/lowry_roads")
LOWRY = (37.58473, -108.91967)            # OSM "Lowry Ruin Great Kiva"
TOP, BOTTOM = 975, 1648                   # video picture inside the phone screenshot
W, H = 1206, BOTTOM - TOP
H_M = 120.0                               # assumed drone height above the pueblo (m); matches the parking loop
R_EFF = 6371000 / (1 - 0.13)              # earth radius with standard refraction
ROAD_A_KM = 3.0

# hand-traced points, crop pixels (x, y - 975). Pueblo = middle of the ruin under the shelter roof.
TRACE = {
    "0180": dict(pueblo=(650, 425), roads={
        "A (straight, away from camera)": [(548, 350), (575, 270), (600, 205), (640, 185), (630, 140), (618, 110)],
        "B (far cross road), left": [(0, 270), (200, 240), (450, 215), (600, 200)],
        "B (far cross road), right": [(620, 205), (800, 225), (1000, 245), (1200, 262)],
        "C (near cross road), left": [(0, 420), (250, 415), (500, 405)],
        "C (near cross road), right": [(720, 370), (900, 335), (1200, 290)],
        "D (toward camera)": [(700, 450), (850, 530), (1000, 620), (1080, 670)]}),
    "0175": dict(pueblo=(600, 415), roads={
        "A (straight, away from camera)": [(650, 345), (800, 280), (950, 200), (1030, 185), (1060, 150), (1100, 115)],
        "B (far road / oval)": [(40, 300), (100, 280), (300, 235), (600, 215), (900, 205), (1000, 215), (1200, 240)],
        "C (near cross road), left": [(0, 360), (300, 370), (500, 385)],
        "C (near cross road), right": [(700, 375), (900, 370), (1200, 360)],
        "D (toward camera)": [(640, 440), (700, 520), (790, 670)]}),
}
SKY_FRAMES = ("0175", "0177", "0178", "0179", "0180")


class Dem:
    def __init__(self):
        self.t = [rasterio.open(f) for f in sorted(glob.glob("output/dem30/USGS_1_n3*.tif"))]
        self.a = [t.read(1) for t in self.t]

    def __call__(self, lat, lon):
        out = np.full(lat.shape, np.nan)
        for t, a in zip(self.t, self.a):
            c, r = ~t.transform * (lon, lat)
            ok = (r >= 0) & (c >= 0) & (r < a.shape[0] - 1) & (c < a.shape[1] - 1) & np.isnan(out)
            out[ok] = map_coordinates(a, [r[ok], c[ok]], order=1)
        return out


def skyline(dem, h_agl=H_M):
    az = np.arange(0, 360, 0.05)
    z0 = dem(np.array([LOWRY[0]]), np.array([LOWRY[1]]))[0] + h_agl
    d = np.arange(500, 160000, 45.0)
    A, D = np.meshgrid(np.radians(az), d, indexing="ij")
    lat = LOWRY[0] + D * np.cos(A) / 111320
    lon = LOWRY[1] + D * np.sin(A) / (111320 * math.cos(math.radians(LOWRY[0])))
    ang = np.degrees(np.arctan((dem(lat, lon) - z0 - D ** 2 / (2 * R_EFF)) / D))
    return az, np.nanmax(ang, axis=1)


def photo_skyline(n):
    g = np.asarray(Image.open(FRAMES / f"IMG_{n}.PNG").convert("L")).astype(float)[TOP:BOTTOM]
    ys = []
    for x in range(W):
        col = g[:200, x]
        k = np.where(col < 0.85 * np.maximum.accumulate(col))[0]
        ys.append(k[0] if len(k) else np.nan)
    return np.array(ys, float)


def axes(p):
    psi, th = math.radians(p[0]), math.radians(-p[1])
    fwd = np.array([math.sin(psi) * math.cos(th), math.cos(psi) * math.cos(th), math.sin(th)])
    right = np.array([math.cos(psi), -math.sin(psi), 0.0])
    return fwd, right, np.cross(right, fwd)


def project(p, az, el):
    """World direction (azimuth, elevation, deg) -> pixel. p = heading, tilt down, roll, focal px."""
    fwd, right, up = axes(p)
    a, e = np.radians(az), np.radians(el)
    d = np.stack([np.sin(a) * np.cos(e), np.cos(a) * np.cos(e), np.sin(e)], -1)
    zc = d @ fwd
    x, y = p[3] * (d @ right) / zc, -p[3] * (d @ up) / zc
    r = math.radians(p[2])
    return W / 2 + x * math.cos(r) - y * math.sin(r), H / 2 + x * math.sin(r) + y * math.cos(r), zc


def ground(p, pts):
    """Pixel -> (east, north) m from the point under the drone, on flat ground H_M below."""
    fwd, right, up = axes(p)
    r = math.radians(-p[2])
    out = []
    for x, y in pts:
        u, v = x - W / 2, y - H / 2
        u, v = u * math.cos(r) - v * math.sin(r), u * math.sin(r) + v * math.cos(r)
        d = fwd + (u / p[3]) * right - (v / p[3]) * up
        out.append((d[0] * H_M / -d[2], d[1] * H_M / -d[2]) if d[2] < 0 else (np.nan, np.nan))
    return np.array(out)


def fit_camera(ys, az, el):
    xs, ok = np.arange(W), ~np.isnan(ys)

    def res(p):
        x, y, zc = project(p, az, el)
        m = zc > 0
        o = np.argsort(x[m])
        return np.nan_to_num(ys[ok] - np.interp(xs[ok], x[m][o], y[m][o], left=np.nan, right=np.nan), nan=50.0)
    fits = []
    for psi in np.arange(0, 360, 4):
        r = least_squares(res, [psi, 11, 0, 1150], loss="soft_l1", f_scale=3,
                          bounds=([psi - 30, 0, -10, 700], [psi + 30, 30, 10, 2000]))
        fits.append((float(np.mean(np.minimum(np.abs(r.fun), 30))), r.x))
    fits.sort(key=lambda t: t[0])
    best = fits[0]
    other = next(f for f in fits if abs((f[1][0] - best[1][0] + 180) % 360 - 180) > 8)
    return best[1], best[0], other[1][0] % 360, other[0]


def to_lonlat(en):
    lat = LOWRY[0] + en[:, 1] / 111320
    lon = LOWRY[1] + en[:, 0] / (111320 * math.cos(math.radians(LOWRY[0])))
    return lon, lat


def bearing(a, b):
    return (math.degrees(math.atan2(b[0] - a[0], b[1] - a[1])) + 360) % 360


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dem = Dem()
    az, el = skyline(dem)
    cams = {}
    for n in SKY_FRAMES:
        p, err, alt, alt_err = fit_camera(photo_skyline(n), az, el)
        cams[n] = dict(heading=round(p[0] % 360, 2), tilt_down=round(p[1], 2), roll=round(p[2], 2), focal_px=round(p[3]),
                       hfov_deg=round(2 * math.degrees(math.atan(W / 2 / p[3])), 1), sky_err_px=round(err, 2),
                       next_best_heading=round(alt, 1), next_best_err_px=round(alt_err, 2), p=p.tolist())
        print(n, {k: v for k, v in cams[n].items() if k != "p"}, flush=True)
    (OUT / "cameras.json").write_text(json.dumps(cams, indent=1))

    # skyline check picture (frames with roads traced)
    sk = []
    for n in TRACE:
        im = Image.open(FRAMES / f"IMG_{n}.PNG").convert("RGB").crop((0, TOP, W, BOTTOM))
        d = ImageDraw.Draw(im)
        x, y, zc = project(np.array(cams[n]["p"]), az, el)
        o = np.argsort(x)
        d.line([(a, b) for a, b, c in zip(x[o], y[o], zc[o]) if c > 0 and -5 < a < W + 5], fill=(255, 0, 0), width=1)
        d.text((8, 8), f"IMG_{n}: computed skyline (red), heading {cams[n]['heading']} deg", fill=(0, 0, 255))
        sk.append(im.crop((0, 0, W, 160)))
    s = Image.new("RGB", (W, 160 * len(sk)), "white")
    for i, im in enumerate(sk):
        s.paste(im, (0, 160 * i))
    s.save(OUT / "skyline_check.png")

    # roads on the ground, relative to the pueblo
    roads = {}
    for n, tr in TRACE.items():
        p = np.array(cams[n]["p"])
        P = ground(p, [tr["pueblo"]])[0]
        for name, pts in tr["roads"].items():
            g = ground(p, pts) - P
            lon, lat = to_lonlat(g)
            roads.setdefault(name, []).append(dict(frame=n, bearing=round(bearing(g[0], g[-1]), 1),
                                                   lonlat=[[round(a, 6), round(b, 6)] for a, b in zip(lon, lat)]))
    a_brg = float(np.mean([r["bearing"] for r in roads["A (straight, away from camera)"]]))
    (OUT / "roads.json").write_text(json.dumps(dict(road_A_bearing=round(a_brg, 1), drone_height_m=H_M, roads=roads), indent=1))
    print("Road A bearing:", [r["bearing"] for r in roads["A (straight, away from camera)"]], "mean", round(a_brg, 1), flush=True)

    # detector along Road A
    tiles = ct.tile_index()
    p0 = np.array(ct.TO_UTM.transform(np.array([LOWRY[1]]), np.array([LOWRY[0]]))).ravel()
    u = np.array([math.sin(math.radians(a_brg)), math.cos(math.radians(a_brg))])
    v = np.array([u[1], -u[0]])
    p1 = p0 + u * ROAD_A_KM * 1000
    z, L = ct.line_strip_z(tiles, p0, p1)
    pieces = ct.stretches(z, min_run=ct.PIECE_MIN_M)
    offs = np.arange(-ct.BAND_M, ct.BAND_M + 1)
    zmean = np.nanmean(np.where(np.nan_to_num(z) >= 2, 1.0, 0.0), axis=0)
    print(f"Road A detector: {len(pieces)} pieces >= {ct.PIECE_MIN_M} m:",
          [(round(s), round(e), round(o)) for s, e, o in pieces], flush=True)
    print("share of line length at z>=2, by offset (every 10 m):",
          {int(o): round(float(m), 2) for o, m in zip(offs[::10], zmean[::10])}, flush=True)
    ends = np.array([p0, p1])
    lons, lats = ct.TO_LL.transform(ends[:, 0], ends[:, 1])
    tree = mr.road_tree(mr.fetch("lowry", lats, lons))
    views = [cv.render(tiles, tree, p0, u, v, s0, min(s0 + 1000, L), pieces,
                       f"Road A from Lowry at {a_brg:.1f} deg | {s0 / 1000:.0f}-{min(s0 + 1000, L) / 1000:.0f} km | "
                       f"left = pueblo end; up = west side; detector pieces in red")
             for s0 in np.arange(0, L - 50, 1000)]
    views = [im for im in views if im is not None]
    s = Image.new("RGB", (views[0].size[0], sum(im.size[1] for im in views)), "white")
    y = 0
    for im in views:
        s.paste(im, (0, y))
        y += im.size[1]
    s.save(OUT / "road_A_view.png", optimize=True)

    # KML
    cols = {"0180": "ff0000ff", "0175": "ffff8800"}
    pm = []
    for name, lst in roads.items():
        for r in lst:
            c = " ".join(f"{a},{b},0" for a, b in r["lonlat"])
            pm.append(f"<Placemark><name>{name} (IMG_{r['frame']}, {r['bearing']} deg)</name><Style><LineStyle>"
                      f"<color>{cols[r['frame']]}</color><width>3</width></LineStyle></Style>"
                      f"<LineString><coordinates>{c}</coordinates></LineString></Placemark>")
    ray = f"{LOWRY[1]},{LOWRY[0]},0 {lons[1]},{lats[1]},0"
    (OUT / "lowry_roads.kml").write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document><name>Lowry roads from the PBS drone frames</name>
<description>Traced from IMG_0175 (blue) and IMG_0180 (red); camera fitted to the Sleeping Ute skyline. Bearings are
reliable to a few degrees; distances assume a {H_M:.0f} m drone height and get rougher with distance.</description>
<Placemark><name>Road A ray, {a_brg:.1f} deg, {ROAD_A_KM:.0f} km</name><Style><LineStyle><color>ff00ffff</color><width>2</width></LineStyle></Style>
<LineString><coordinates>{ray}</coordinates></LineString></Placemark>
<Placemark><name>Lowry Pueblo</name><Point><coordinates>{LOWRY[1]},{LOWRY[0]},0</coordinates></Point></Placemark>
{''.join(pm)}
</Document></kml>
""")
    print("wrote", OUT / "lowry_roads.kml", flush=True)


if __name__ == "__main__":
    main()
