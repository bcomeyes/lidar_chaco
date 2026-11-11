"""Pictures for the eye: the known Chaco roads (Friedman, Sofaer & Weiner 2017 Fig. 11) drawn
in the same four panels as the corridor views, so you learn what a real road looks like here.

Each marked segment of output/ground_truth/fig11_roads.tif (>= 100 m²) gets a strip along its
main direction, ±200 m across, sampled at 1 m from the NCALM 0.5 m DTM (same scale as the
Montezuma views). Everything else in the panel is ordinary
ground to compare against. Many "segments" are bent or branched road networks, so instead of
ticks a fifth panel repeats panel 1 with the published road outline painted in red.

Outputs: output/known_road_views/segNN_W.png, index.html
"""
import datetime as dt
import html
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import rasterio
from scipy import ndimage
from scipy.spatial import cKDTree

import corridor_test as ct
import corridor_views as cv

DEM = "output/output_be.tif"
OUT = Path("output/known_road_views")
PAD_M = 100
STOP_AT = dt.time(20, 15)


def add_marked_panel(im, tiles, road, tr, p0, u, v, s0, s1):
    """Panel 1 again, with the Fig. 11 road pixels tinted red."""
    offs = np.arange(cv.HALF_M, -cv.HALF_M - 1, -1.0)
    T, S = np.meshgrid(offs, np.arange(s0, s1, 1.0), indexing="ij")
    x, y = p0[0] + S * u[0] + T * v[0], p0[1] + S * u[1] + T * v[1]
    z = ct.sample(tiles, x, y)
    z = np.where(np.isnan(z), np.nanmean(z), z)
    col, row = ~tr * (x, y)
    r, c = row.astype(int), col.astype(int)
    ok = (r >= 0) & (c >= 0) & (r < road.shape[0]) & (c < road.shape[1])
    mark = np.zeros(z.shape, bool)
    mark[ok] = road[r[ok], c[ok]]
    a = np.asarray(cv.stretch(cv.side_lit(z, +1)).convert("RGB")).astype(float)
    a[mark] = 0.45 * a[mark] + 0.55 * np.array([230, 0, 0])
    h, gap = offs.size, 22
    out = Image.new("RGB", (im.width, im.height + h + gap), "white")
    out.paste(im, (0, 0))
    out.paste(Image.fromarray(a.astype(np.uint8)), (0, im.height))
    ImageDraw.Draw(out).text((4, im.height + h + 4), "panel 1 again, published road (Fig. 11) in red", fill="black")
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with rasterio.open(DEM) as s:
        tiles = [dict(path=DEM, bounds=tuple(s.bounds), nodata=s.nodata, utm=True)]   # sample in its own CRS
    with rasterio.open("output/ground_truth/fig11_roads.tif") as s:
        road, tr = s.read(1) == 2, s.transform
    no_roads = cKDTree([[1e12, 1e12]])                                    # no OSM layer here

    lab, _ = ndimage.label(road, structure=np.ones((3, 3)))
    rows_out = []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        if dt.datetime.now().time() >= STOP_AT:
            print("stop time reached", flush=True)
            break
        r, c = np.where(lab[sl] == i)
        if r.size < 400:
            continue
        x, y = tr * (c + sl[1].start + 0.5, r + sl[0].start + 0.5)
        pts = np.c_[x, y]
        ctr = pts.mean(0)
        _, vecs = np.linalg.eigh(np.cov((pts - ctr).T))
        u = vecs[:, 1] if vecs[1, 1] >= 0 else -vecs[:, 1]               # along, pointing north-ish
        v = np.array([u[1], -u[0]])
        t, o = (pts - ctr) @ u, (pts - ctr) @ v
        bearing = np.degrees(np.arctan2(u[0], u[1])) % 180
        p0 = ctr + t.min() * u
        L = t.max() - t.min()
        for wi, s0 in enumerate(np.arange(-PAD_M, L + PAD_M, cv.WIN_M), 1):
            s1 = min(s0 + cv.WIN_M, L + PAD_M)
            title = (f"Known road, Fig. 11 segment {i} | {L:.0f} m marked, bearing {bearing:.0f} deg | "
                     "left = south end")
            im = cv.render(tiles, no_roads, p0, u, v, s0, s1, [], title)
            if im is None:
                continue
            im = add_marked_panel(im, tiles, road, tr, p0, u, v, s0, s1)
            f = OUT / f"seg{i:02d}_{wi}.png"
            im.save(f, optimize=True)
            rows_out.append((i, f.name, title))
            print(f, flush=True)
    body = "\n".join(f"<p>{html.escape(t)}</p><img src='{f}' style='max-width:100%'>" for _, f, t in rows_out)
    (OUT / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Known Chaco roads</title>"
        "<body style='font-family:sans-serif;background:#fff;margin:16px'><h1>Known Chaco roads, same views as the corridors</h1>"
        "<p>Each strip runs along a road marked in Friedman, Sofaer &amp; Weiner 2017 Fig. 11 (Pueblo Alto area), "
        "±200 m across, 1 m pixels. The bottom panel paints the published road in red; compare it with the clean panels above.</p>" + body + "</body>")
    print(f"{len(rows_out)} images; open {OUT / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
