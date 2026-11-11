"""Georeference Friedman, Sofaer & Weiner 2017 Fig. 11 onto our DTM and extract
the marked road segments as ground truth.

Stage 1 (register): fit each figure panel to our hillshade by normalized
cross-correlation over pixel size and rotation, then refine with tile-by-tile
matching and a least-squares affine fit. Residuals are reported in metres.
Stage 2 (extract): white road marks in panel (b) -> mask on the DTM's 0.5 m grid.

Inputs : refs/fig11_pueblo_alto_landscape.jpg, output/hillshade_az*_alt*.tif
Outputs: output/ground_truth/  (written fresh each run; nothing else is touched)
Run    : .venv/bin/python scripts/fig11_ground_truth.py
"""
import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from scipy import ndimage
from scipy.signal import fftconvolve

FIG = Path("refs/fig11_pueblo_alto_landscape.jpg")
OUT = Path("output/ground_truth")
HILLSHADES = sorted(Path("output").glob("hillshade_az*_alt*.tif"))

PANELS = {"a": (0, 658), "b": (676, 1334)}   # column ranges in the figure image
PANEL_ROWS = 659
SCALEBAR_M_PER_PX = 400 / 148               # 400 m bar measured at 148 px
LEGEND_MASK = (slice(0, 110), slice(0, 200))  # scale bar + north arrow, panel px


def load_panels():
    rgb = np.asarray(Image.open(FIG).convert("RGB")).astype(float)
    return {k: rgb[:PANEL_ROWS, c0:c1] for k, (c0, c1) in PANELS.items()}


def road_marks(panels):
    """The authors' road lines = where panel (b) is much brighter than (a).

    Panel (b) sits 1 px right of (a); it is shifted back so the returned mask
    is in panel (a)'s pixel frame. Edge columns/rows hold frame borders.
    """
    a, b = panels["a"], np.zeros_like(panels["a"])
    b[:, :-1] = panels["b"][:, 1:]
    lift = b.mean(axis=2) - a.mean(axis=2)
    sat = b.max(axis=2) - b.min(axis=2)
    m = (lift > 35) & (sat < 60)
    m[:120, :220] = False                       # scale bar + north arrow
    m[:, :3] = m[:, -4:] = m[-3:] = False
    return m


def ncc_map(image, template, weight):
    """Masked normalized cross-correlation of template over image (valid mode)."""
    w = weight.astype(float)
    t = template * w
    n = w.sum()
    t = (t - t.sum() / n) * w
    t_norm = np.sqrt((t ** 2).sum())
    flip = lambda a: a[::-1, ::-1]
    num = fftconvolve(image, flip(t), mode="valid")
    s1 = fftconvolve(image, flip(w), mode="valid")
    s2 = fftconvolve(image ** 2, flip(w), mode="valid")
    var = np.clip(s2 - s1 ** 2 / n, 1e-9, None)
    return num / (np.sqrt(var) * t_norm)


def subpixel_peak(c):
    r, k = np.unravel_index(np.argmax(c), c.shape)
    def par(m1, m0, p1):
        d = m1 - 2 * m0 + p1
        return 0.0 if d == 0 else 0.5 * (m1 - p1) / d
    dr = par(c[r - 1, k], c[r, k], c[r + 1, k]) if 0 < r < c.shape[0] - 1 else 0.0
    dk = par(c[r, k - 1], c[r, k], c[r, k + 1]) if 0 < k < c.shape[1] - 1 else 0.0
    return r + dr, k + dk, c[r, k]


def coarse_fit(gray, weight, hs, hs_transform):
    """Grid search over pixel size and rotation; translation by FFT NCC."""
    best = None
    for px in np.arange(2.55, 2.86, 0.025):
        img = ndimage.zoom(ndimage.gaussian_filter(hs, px / 0.5 / 2.5), 0.5 / px, order=1)
        for rot in np.arange(-2.0, 2.01, 0.5):
            tpl = ndimage.rotate(gray, rot, reshape=False, order=1, mode="nearest")
            wt = ndimage.rotate(weight.astype(float), rot, reshape=False, order=0) > 0.5
            c = ncc_map(img, tpl, wt)
            r, k, score = subpixel_peak(c)
            if best is None or score > best["score"]:
                best = dict(score=float(score), px=float(px), rot=float(rot), r=r, k=k)
    return best


def panel_to_dtm_rc(best):
    """Map figure-panel pixel (row, col) -> DTM grid (row, col) at 0.5 m."""
    f = best["px"] / 0.5
    th = -np.deg2rad(best["rot"])     # undo the template rotation
    cy, cx = (PANEL_ROWS - 1) / 2, (658 - 1) / 2
    def fn(rows, cols):
        # template was rotated by `rot` about its centre before matching
        y, x = rows - cy, cols - cx
        yr = np.cos(th) * y + np.sin(th) * x
        xr = -np.sin(th) * y + np.cos(th) * x
        return (best["r"] + cy + yr) * f, (best["k"] + cx + xr) * f
    return fn


def refine(gray, weight, hs, fn, tile=96, search=12):
    """Tile matching at full DTM resolution -> affine fit + residuals (m)."""
    src, dst = [], []
    for r0 in range(0, PANEL_ROWS - tile, tile // 2):
        for c0 in range(0, 658 - tile, tile // 2):
            w = weight[r0:r0 + tile, c0:c0 + tile]
            if w.mean() < 0.9:
                continue
            tpl = gray[r0:r0 + tile, c0:c0 + tile]
            if tpl.std() < 8:            # featureless mesa top: no reliable match
                continue
            rc, cc = r0 + tile / 2, c0 + tile / 2
            # sample the hillshade onto the figure's pixel grid around the tile
            rr, kk = np.mgrid[r0 - search:r0 + tile + search, c0 - search:c0 + tile + search]
            Rg, Cg = fn(rr.astype(float), kk.astype(float))
            patch = ndimage.map_coordinates(hs, [Rg, Cg], order=1, mode="nearest")
            c = ncc_map(patch, tpl, np.ones_like(tpl, bool))
            pr, pk, score = subpixel_peak(c)
            if score < 0.5 or not (1 < pr < 2 * search - 1 and 1 < pk < 2 * search - 1):
                continue
            Rm, Cm = fn(np.array([rc + pr - search]), np.array([cc + pk - search]))
            src.append((rc, cc))
            dst.append((Rm[0], Cm[0]))
    src, dst = np.array(src), np.array(dst)
    A = np.c_[src, np.ones(len(src))]
    coef, *_ = np.linalg.lstsq(A, dst, rcond=None)        # 3x2 affine
    pred = A @ coef
    res_m = np.hypot(*(pred - dst).T) * 0.5
    return coef, res_m, len(src)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panels = load_panels()
    marks = road_marks(panels)
    report = {"m_per_px_from_scalebar": SCALEBAR_M_PER_PX, "panels": {}}

    hs_by_name = {}
    for p in HILLSHADES:
        with rasterio.open(p) as src:
            hs_by_name[p.stem] = src.read(1).astype(np.float32)
            profile, transform = src.profile, src.transform

    fits = {}
    for name, rgb in panels.items():
        gray = rgb.mean(axis=2)
        weight = np.ones(gray.shape, bool)
        weight[LEGEND_MASK] = False
        if name == "b":
            weight &= ~ndimage.binary_dilation(marks, iterations=2)
        results = {}
        for hname, hs in hs_by_name.items():
            results[hname] = coarse_fit(gray, weight, hs, transform)
            print(f"panel {name} vs {hname}: {results[hname]}", flush=True)
        hname = max(results, key=lambda h: results[h]["score"])
        best = results[hname]
        fn = panel_to_dtm_rc(best)
        hs_smooth = ndimage.gaussian_filter(hs_by_name[hname], best["px"] / 0.5 / 2.5)
        coef, res_m, n = refine(gray, weight, hs_smooth, fn)
        fits[name] = coef
        report["panels"][name] = dict(
            hillshade=hname, coarse=best, tiles_used=n,
            residual_m=dict(rms=float(np.sqrt((res_m ** 2).mean())),
                            median=float(np.median(res_m)), max=float(res_m.max())),
            affine_rowcol=coef.tolist())
        print(f"panel {name}: best={hname} tiles={n} "
              f"rms={np.sqrt((res_m**2).mean()):.2f} m max={res_m.max():.2f} m", flush=True)

    # cross-check: the same figure point mapped through each panel's own fit
    # (panel b's content sits 1 px right of panel a's)
    probe = np.array([[r, c, 1.0] for r in (100, 330, 560) for c in (100, 330, 560)])
    probe_b = probe + [0, 1, 0]
    diff_m = np.hypot(*((probe @ fits["a"]) - (probe_b @ fits["b"])).T) * 0.5
    report["panel_a_vs_b_disagreement_m"] = dict(mean=float(diff_m.mean()), max=float(diff_m.max()))
    print("panel a vs b disagreement (m):", diff_m.round(2))

    (OUT / "registration.json").write_text(json.dumps(report, indent=2))

    # Stage 2: road marks (panel a frame) -> DTM 0.5 m grid via panel a's affine
    coef = fits["a"]
    M, t = coef[:2].T, coef[2]                       # [R, C] = M @ [r, c] + t
    corners = np.array([[0, 0], [0, 657], [658, 0], [658, 657]], float)
    RC = corners @ M.T + t
    H, W = profile["height"], profile["width"]
    R0, C0 = np.floor(RC.min(0)).astype(int).clip(0)
    R1, C1 = np.ceil(RC.max(0)).astype(int)
    R1, C1 = min(R1, H), min(C1, W)
    RR, CC = np.mgrid[R0:R1, C0:C1]
    rc = (np.stack([RR.ravel() - t[0], CC.ravel() - t[1]]).T @ np.linalg.inv(M).T)
    r = np.rint(rc[:, 0]).astype(int).reshape(RR.shape)
    c = np.rint(rc[:, 1]).astype(int).reshape(RR.shape)
    inside = (r >= 0) & (r < PANEL_ROWS) & (c >= 0) & (c < 658)
    legend = (r < 120) & (c < 220)                    # no marks possible under the legend
    roads = np.zeros((H, W), np.uint8)               # 0 outside figure, 1 figure, 2 road mark
    sub = np.where(inside & ~legend, 1, 0).astype(np.uint8)
    sub[inside & marks[r.clip(0, PANEL_ROWS - 1), c.clip(0, 657)]] = 2
    roads[R0:R1, C0:C1] = sub
    prof = profile.copy()
    prof.update(dtype="uint8", count=1, compress="lzw", nodata=None)
    with rasterio.open(OUT / "fig11_roads.tif", "w", **prof) as dst:
        dst.write(roads, 1)
        dst.update_tags(values="0=outside Fig.11 footprint or under legend, 1=Fig.11 area, 2=marked road",
                        source="Friedman, Sofaer & Weiner 2017, Fig. 11b, doi:10.1017/aap.2017.25")
    report["footprint_dtm_rowcol"] = [int(R0), int(C0), int(R1), int(C1)]
    report["road_cells"] = int((roads == 2).sum())
    report["footprint_cells"] = int((roads >= 1).sum())
    (OUT / "registration.json").write_text(json.dumps(report, indent=2))
    print(f"fig11_roads.tif: footprint rows {R0}-{R1} cols {C0}-{C1}, "
          f"road cells {report['road_cells']} ({report['road_cells'] * 0.25:.0f} m2)", flush=True)


if __name__ == "__main__":
    main()
