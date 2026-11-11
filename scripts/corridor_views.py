"""Pictures for the eye: candidate road chains on the best Chacoan-era corridors.

For each aligned chain on a real site-to-site line (output/corridor_test_chacoan/
aligned_results.json), the 1 m DEM is resampled into a strip that runs along the line
(left to right = from the first site towards the second; up = the right-hand side, + offset).
Each 1 km window gets four panels, all ±200 m across the line:
  1. side-lit relief, sun low (8 deg) from the + side, at right angles to the line
  2. the same, sun from the - side (a road dip throws a shadow on one side, a lit edge on the other)
  3. openness (how much sky each point sees within 20 m; dips and trenches are darker)
  4. local relief (ground minus its 5 m smoothed self, ±10 cm), with mapped OSM roads in yellow
Red ticks above and below each panel mark the detector's pieces (start/end, at their offset);
the black ticks at the ends mark the site-to-site line itself (offset 0).

Corridors: those whose real line beat all 10 controls with mapped roads removed
(mapped_roads_summary.txt), or a list given on the command line.

Outputs: output/corridor_views/<corridor>/*.png, output/corridor_views/index.html
"""
import datetime as dt
import html
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

import corridor_test as ct
import mapped_roads as mr

SRC = Path("output/corridor_test_chacoan")
OUT = Path("output/corridor_views")
HALF_M = 200                   # panel half-height across the line (= the detector band)
WIN_M = 1000                   # window length along the line
PAD_M = 250                    # context before/after each chain
SUN_ALT = 8.0
VE = 3.0                       # vertical exaggeration for the side-lit panels
OPEN_R = 20                    # openness search radius, m
RELIEF_SIGMA = 5
RELIEF_CLIP = 0.10
STOP_AT = dt.time(20, 15)


def side_lit(z, sign):
    """Hillshade in strip coords (rows = offset, cols = along); sun across the line from the `sign` side."""
    dzt, dzs = np.gradient(z * VE)           # d/d(row) = d/d(offset, but rows go downwards), d/d(along)
    dzt = -dzt                                # rows run from +offset (top) to -offset
    alt = math.radians(SUN_ALT)
    lt, lz = sign * math.cos(alt), math.sin(alt)
    return (-dzt * lt + lz) / np.sqrt(dzs ** 2 + dzt ** 2 + 1)


def openness(z):
    """Positive openness (mean of 90 deg minus the highest horizon angle over 8 directions), degrees."""
    zp = np.pad(z, OPEN_R, mode="reflect")
    acc = np.zeros_like(zp)
    for dy, dx in ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)):
        step = math.hypot(dy, dx)
        best = np.full_like(zp, -np.inf)
        for k in range(1, int(OPEN_R / step) + 1):
            sh = np.roll(zp, (k * dy, k * dx), axis=(0, 1))
            best = np.maximum(best, np.degrees(np.arctan((sh - zp) / (k * step))))
        acc += 90 - best
    return (acc / 8)[OPEN_R:-OPEN_R, OPEN_R:-OPEN_R]


def to_img(a, lo, hi):
    return Image.fromarray((np.clip((a - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8))


def stretch(a, pct=1):
    lo, hi = np.nanpercentile(a, (pct, 100 - pct))
    return to_img(a, lo, hi)


def render(tiles, tree, p0, u, v, s0, s1, pieces, title):
    ss = np.arange(s0, s1, 1.0)
    offs = np.arange(HALF_M, -HALF_M - 1, -1.0)          # top row = +offset
    T, S = np.meshgrid(offs, ss, indexing="ij")
    x, y = p0[0] + S * u[0] + T * v[0], p0[1] + S * u[1] + T * v[1]
    z = ct.sample(tiles, x, y)
    if np.isnan(z).mean() > 0.5:
        return None
    z = np.where(np.isnan(z), np.nanmean(z), z)
    rel = z - ndimage.gaussian_filter(z, RELIEF_SIGMA)
    panels = [stretch(side_lit(z, +1)), stretch(side_lit(z, -1)), stretch(openness(z), 5),
              to_img(rel, -RELIEF_CLIP, RELIEF_CLIP)]
    labels = ["sun from + side, 8 deg", "sun from - side, 8 deg", "openness, 20 m (darker = more enclosed)", "local relief ±10 cm; yellow = mapped roads"]
    h, w, gap, top = offs.size, ss.size, 22, 30
    im = Image.new("RGB", (w, top + len(panels) * (h + gap)), "white")
    d = ImageDraw.Draw(im)
    d.text((4, 8), title, fill="black")
    dist, _ = tree.query(np.c_[x.ravel(), y.ravel()])
    near = (dist <= 3).reshape(z.shape)
    for k, (p, lab) in enumerate(zip(panels, labels)):
        y0 = top + k * (h + gap)
        rgb = p.convert("RGB")
        if k == 3:
            a = np.asarray(rgb).copy()
            a[near] = (255, 210, 0)
            rgb = Image.fromarray(a)
        im.paste(rgb, (0, y0))
        d.text((4, y0 + h + 4), lab, fill="black")
        for ps, pe, o in pieces:                          # red ticks just outside the piece, both sides
            if pe < s0 or ps > s1:
                continue
            a, b = max(ps, s0) - s0, min(pe, s1) - s0
            r = HALF_M - o
            for yy in (y0 + r - 9, y0 + r + 9):
                if y0 <= yy < y0 + h:
                    d.line([(a, yy), (b, yy)], fill=(230, 0, 0), width=2)
        for xx in (0, w - 6):                              # site-to-site line, offset 0
            d.line([(xx, y0 + HALF_M), (xx + 6, y0 + HALF_M)], fill="black", width=3)
    return im


def corridors_to_show():
    if len(sys.argv) > 1:
        return sys.argv[1:]
    out = []
    for line in (SRC / "mapped_roads_summary.txt").read_text().splitlines():
        m = re.match(r"(.+?): aligned pieces .* beats ([\d.]+) of (\d+)", line)
        if m and float(m.group(2)) >= int(m.group(3)) - 0.5:
            out.append(m.group(1))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tiles = ct.tile_index()
    sites = {}
    for f in ("sites_chacoan.json", "great_houses.json", "sites.json", "sites_local.json"):
        for s in json.loads((Path("output/corridors") / f).read_text()):
            sites.setdefault(s["name"], s)
    res = {r["corridor"]: r for r in json.loads((SRC / "aligned_results.json").read_text()) if r["line"] == "real"}
    rows = []
    for name in corridors_to_show():
        if dt.datetime.now().time() >= STOP_AT:
            print("stop time reached", flush=True)
            break
        r = res.get(name)
        if not r or not r["aligned"]:
            continue
        a, b = name.split(" - ")
        p0, p1 = ct.endpoints(sites[a], sites[b])
        u = (p1 - p0) / np.linalg.norm(p1 - p0)
        v = np.array([u[1], -u[0]])
        ends = np.array([p0, p1])
        lons, lats = ct.TO_LL.transform(ends[:, 0], ends[:, 1])
        tree = mr.road_tree(mr.fetch(name, lats, lons))
        slug = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
        (OUT / slug).mkdir(exist_ok=True)
        for ci, ch in enumerate(r["aligned"], 1):
            c0, c1 = min(p[0] for p in ch) - PAD_M, max(p[1] for p in ch) + PAD_M
            for wi, s0 in enumerate(np.arange(max(c0, 0), c1, WIN_M), 1):
                s1 = min(s0 + WIN_M, c1)
                title = (f"{name} | chain {ci} ({len(ch)} pieces) | {s0 / 1000:.2f}-{s1 / 1000:.2f} km from {a} "
                         f"| left = towards {a}, right = towards {b}")
                im = render(tiles, tree, p0, u, v, s0, s1, ch, title)
                if im is None:
                    continue
                f = OUT / slug / f"chain{ci}_{wi}.png"
                im.save(f, optimize=True)
                pcs = ", ".join(f"{s / 1000:.2f}-{e / 1000:.2f} km at {o:+.0f} m" for s, e, o in ch)
                rows.append((name, f.relative_to(OUT), title, pcs))
                print(f, flush=True)
    body = []
    for name in dict.fromkeys(r[0] for r in rows):
        body.append(f"<h2>{html.escape(name)}</h2>")
        for _, f, title, pcs in (r for r in rows if r[0] == name):
            body.append(f"<p>{html.escape(title)}<br><small>pieces: {html.escape(pcs)}</small></p>"
                        f"<img src='{f}' style='max-width:100%'>")
    (OUT / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Corridor views</title>"
        "<body style='font-family:sans-serif;background:#fff;margin:16px'><h1>Candidate road chains, Chacoan-era corridors</h1>"
        "<p>Each window is 1 km along the site-to-site line, ±200 m across. Red ticks bracket the detector's pieces. "
        "A road should show as a thin straight line between the red ticks, in step across all four panels.</p>"
        + "\n".join(body) + "</body>")
    print(f"{len(rows)} images; open {OUT / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
