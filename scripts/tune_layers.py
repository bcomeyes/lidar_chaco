"""Tune the road-detection layer against the Fig. 11 ground truth.

Search, one stage at a time (each keeps the best of the previous):
  1. LRM scales: fine smoothing sigma x coarse window
  2. slope suppression: damp cells steeper than a limit (cliffs, talus)
  3. straight-line filter: average along lines at 24 bearings, keep the max
     (straight roads add up along their length; blobs and branching drainages don't)

Honest scoring: segments are split in two. Settings are chosen on the TRAIN half
and reported on the TEST half too. Metric = local AUC (each road vs the ground
15-40 m beside it; 0.5 = chance, 1 = perfect).

Inputs : output/output_be.tif, output/ground_truth/fig11_roads.tif
Outputs: output/ground_truth/tuning.json, best_layer.tif, overlay_best_layer*.jpg
"""
import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import Window
from scipy import ndimage
from scipy.ndimage import gaussian_filter, uniform_filter, uniform_filter1d

from score_layers import auc

OUT = Path("output/ground_truth")
CELL = 0.5
PAD = 300                                   # cells of context around the Fig. 11 footprint


def load():
    with rasterio.open(OUT / "fig11_roads.tif") as s:
        gt_full = s.read(1)
    rows, cols = np.where(gt_full >= 1)
    r0, r1 = max(rows.min() - PAD, 0), min(rows.max() + PAD, gt_full.shape[0])
    c0, c1 = max(cols.min() - PAD, 0), min(cols.max() + PAD, gt_full.shape[1])
    win = Window(c0, r0, c1 - c0, r1 - r0)
    with rasterio.open("output/output_be.tif") as s:
        dem = s.read(1, window=win).astype(np.float64)
        profile = s.profile
        transform = s.window_transform(win)
        nd = s.nodata
    dem[dem == nd] = np.nan
    dem = np.nan_to_num(dem, nan=np.nanmean(dem))
    return dem, gt_full[r0:r1, c0:c1], profile, transform


def split_segments(gt):
    road = gt == 2
    lab, n = ndimage.label(road, structure=np.ones((3, 3)))
    sizes = ndimage.sum(road, lab, range(1, n + 1))
    keep = [i + 1 for i in range(n) if sizes[i] >= 400]
    near = ndimage.binary_dilation(road, iterations=30)
    sets = {"train": [], "test": []}
    for j, i in enumerate(keep):           # alternate by label order (north to south)
        seg = lab == i
        ring = ndimage.binary_dilation(seg, iterations=80) & ~near & (gt >= 1)
        sets["train" if j % 2 == 0 else "test"].append((seg, ring))
    return sets


def score(layer, segs):
    return float(np.median([auc(layer[s], layer[r], n=60_000) for s, r in segs]))


def lrm(dem, sigma_m, window_m):
    return -(gaussian_filter(dem, sigma_m / CELL) - uniform_filter(dem, size=int(window_m / CELL)))


def line_filter(layer, length_m, n_angles=24):
    best = np.full(layer.shape, -np.inf)
    size = int(length_m / CELL)
    for ang in np.arange(n_angles) * 180.0 / n_angles:
        rot = ndimage.rotate(layer, ang, reshape=True, order=1, mode="nearest")
        rot = uniform_filter1d(rot, size, axis=0)
        back = ndimage.rotate(rot, -ang, reshape=True, order=1, mode="nearest")
        dr = (back.shape[0] - layer.shape[0]) // 2
        dc = (back.shape[1] - layer.shape[1]) // 2
        best = np.maximum(best, back[dr:dr + layer.shape[0], dc:dc + layer.shape[1]])
    return best


def main():
    dem, gt, profile, transform = load()
    sets = split_segments(gt)
    print(f"segments: train {len(sets['train'])}, test {len(sets['test'])}", flush=True)
    report = {"stages": []}

    def record(stage, params, layer):
        tr, te = score(layer, sets["train"]), score(layer, sets["test"])
        report["stages"].append(dict(stage=stage, params=params, train=tr, test=te))
        print(f"{stage:10} {json.dumps(params):45} train {tr:.3f}  test {te:.3f}", flush=True)
        return tr

    # baseline = notebook LRM (sigma 1.5 cells = 0.75 m, window 50 cells = 25 m)
    base = lrm(dem, 0.75, 25)
    record("baseline", {"sigma_m": 0.75, "window_m": 25}, base)

    best = (-1, None, None)
    for sigma_m in (0.75, 1.5, 3.0):
        for window_m in (10, 15, 25, 40):
            L = lrm(dem, sigma_m, window_m)
            s = record("lrm", {"sigma_m": sigma_m, "window_m": window_m}, L)
            if s > best[0]:
                best = (s, {"sigma_m": sigma_m, "window_m": window_m}, L)
    lrm_params, layer = best[1], best[2]

    gy, gx = np.gradient(dem, CELL)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    slope = gaussian_filter(slope, 4)
    best = (best[0], {"slope_limit_deg": None}, layer)
    for lim in (25, 15, 10, 6):
        L = layer * np.clip((lim - slope) / (lim / 2), 0, 1)
        s = record("slope", {"slope_limit_deg": lim}, L)
        if s > best[0]:
            best = (s, {"slope_limit_deg": lim}, L)
    slope_params, layer = best[1], best[2]

    best = (best[0], {"line_length_m": None}, layer)
    for length_m in (20, 40, 80):
        L = line_filter(layer, length_m)
        s = record("line", {"line_length_m": length_m}, L)
        if s > best[0]:
            best = (s, {"line_length_m": length_m}, L)
    line_params, layer = best[1], best[2]

    chosen = {**lrm_params, **slope_params, **line_params}
    report["chosen"] = chosen
    report["baseline"] = {"train": score(base, sets["train"]), "test": score(base, sets["test"])}
    report["final"] = {"train": score(layer, sets["train"]), "test": score(layer, sets["test"])}
    (OUT / "tuning.json").write_text(json.dumps(report, indent=2))
    print("chosen:", chosen, "| final", report["final"], "| baseline", report["baseline"], flush=True)

    prof = profile.copy()
    prof.update(dtype="float32", count=1, compress="lzw", nodata=None,
                height=layer.shape[0], width=layer.shape[1], transform=transform)
    with rasterio.open(OUT / "best_layer.tif", "w", **prof) as dst:
        dst.write(layer.astype(np.float32), 1)
        dst.update_tags(settings=json.dumps(chosen))

    # overlay images: tuned layer in grey, paper's roads outlined in cyan
    for tag, lay in (("baseline", base), ("best", layer)):
        v = np.clip(lay / np.percentile(lay, 99.5), 0, 1)
        rgb = np.stack([v * 255] * 3, -1).astype(np.uint8)
        road = gt == 2
        rgb[road & ~ndimage.binary_erosion(road, iterations=2)] = [0, 220, 255]
        Image.fromarray(rgb).resize((1400, 1400 * rgb.shape[0] // rgb.shape[1]),
                                    Image.BILINEAR).save(OUT / f"overlay_{tag}_layer.jpg", quality=85)


if __name__ == "__main__":
    main()
