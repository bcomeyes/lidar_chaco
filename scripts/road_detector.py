"""Straight shallow-dip road detector (matched filter), run on 1 m data.

Built from what the stacked cross-sections measured (scripts/road_profiles.py):
Chaco roads are dips ~4-7 cm deep, ~5 m wide, straight for hundreds of metres.

For each of 36 bearings (5 deg steps):
  1. dip:   mean of the shoulders (4-10 m each side) minus mean of the centre
            (|x| <= 2.5 m), measured across the bearing. Symmetric shoulders cancel
            any plain slope.
  2. along: average the dip over LENGTH_M along the bearing.
Keep the strongest bearing per cell, then express it as a local z-score
(how unusual vs the surrounding 150 m), so cliffs and flats are judged fairly.

Settings were fixed from the measured profiles, not tuned against the scoring.
v2 (after viewing v1): per-slice dip capped at 15 cm and mean dips > 25 cm rejected,
because deep valleys and drainages swamped the 5 cm roads.

Inputs : output/output_be.tif, output/ground_truth/fig11_roads.tif
Outputs: output/detector/road_z.tif, road_bearing.tif, road_dip_cm.tif,
         scores.json, overlay*.jpg
"""
import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from scipy import ndimage
from scipy.ndimage import convolve1d, uniform_filter, uniform_filter1d

from score_layers import auc

OUT = Path("output/detector")
CELL = 1.0                   # metres; 0.5 m DTM block-averaged 2x2
LENGTH_M = 100
N_BEARINGS = 36
CENTRE_M, SHOULDER_M = (0, 2.5), (4, 10)
LOCAL_M = 150
CAP_M = 0.15                 # per-slice dip cap (measured roads: 4-7 cm)
VALLEY_M = 0.25              # mean dip above this = valley/drainage, rejected


def cross_kernel():
    x = np.arange(-SHOULDER_M[1], SHOULDER_M[1] + CELL, CELL)
    k = np.zeros_like(x)
    centre = np.abs(x) <= CENTRE_M[1]
    shoulder = (np.abs(x) >= SHOULDER_M[0]) & (np.abs(x) <= SHOULDER_M[1])
    k[shoulder] = 1.0 / shoulder.sum()
    k[centre] = -1.0 / centre.sum()
    return k                                   # response = shoulders - centre (m)


def load_1m():
    with rasterio.open("output/output_be.tif") as s:
        dem = s.read(1).astype(np.float64)
        nd, profile = s.nodata, s.profile
    valid = dem != nd
    dem[~valid] = np.nan
    h, w = (dem.shape[0] // 2) * 2, (dem.shape[1] // 2) * 2
    dem = np.nanmean(dem[:h, :w].reshape(h // 2, 2, w // 2, 2), axis=(1, 3))
    valid = ~np.isnan(dem)
    dem = np.where(valid, dem, np.nanmean(dem))
    prof = profile.copy()
    t = prof["transform"]
    prof.update(height=dem.shape[0], width=dem.shape[1],
                transform=rasterio.Affine(t.a * 2, t.b, t.c, t.d, t.e * 2, t.f),
                dtype="float32", count=1, compress="lzw", nodata=None)
    return dem, valid, prof


def detect(dem, valid):
    k = cross_kernel()
    n = int(LENGTH_M / CELL)
    best = np.full(dem.shape, -np.inf, np.float32)
    bearing = np.zeros(dem.shape, np.float32)
    for b in np.arange(N_BEARINGS) * 180.0 / N_BEARINGS:
        # rotate so bearing b points along axis 0 (image "up")
        rot = ndimage.rotate(dem, -b, reshape=True, order=1, mode="nearest")
        ok = ndimage.rotate(valid.astype(np.float32), -b, reshape=True, order=0) > 0.5
        dip = convolve1d(rot, k, axis=1, mode="nearest")
        den = uniform_filter1d(ok.astype(float), n, axis=0)
        raw = uniform_filter1d(np.where(ok, dip, 0), n, axis=0) / np.maximum(den, 1e-9)
        # each slice may add at most a road-sized dip, so only consistency wins
        capped = uniform_filter1d(np.where(ok, np.clip(dip, -CAP_M, CAP_M), 0), n, axis=0) / np.maximum(den, 1e-9)
        # deeper than any measured road on average -> valley or drainage, not a road
        along = np.where((den > 0.9) & (raw < VALLEY_M), capped, np.nan)
        back = ndimage.rotate(np.nan_to_num(along, nan=-1.0), b, reshape=True, order=1, mode="constant", cval=-1.0)
        dr = (back.shape[0] - dem.shape[0]) // 2
        dc = (back.shape[1] - dem.shape[1]) // 2
        back = back[dr:dr + dem.shape[0], dc:dc + dem.shape[1]].astype(np.float32)
        better = back > best
        best[better] = back[better]
        bearing[better] = b
        print(f"bearing {b:5.1f} done", flush=True)
    best[~valid] = np.nan
    return best, bearing


def local_z(r):
    size = int(LOCAL_M / CELL)
    x = np.nan_to_num(r, nan=0.0)
    w = (~np.isnan(r)).astype(float)
    m = uniform_filter(x, size) / np.maximum(uniform_filter(w, size), 1e-9)
    v = uniform_filter(x ** 2, size) / np.maximum(uniform_filter(w, size), 1e-9) - m ** 2
    return (r - m) / np.sqrt(np.maximum(v, 1e-12))


def score(z, gt):
    """Per-segment local AUC (road vs 15-40 m beside it), as in score_layers.py."""
    road = gt == 2
    lab, n = ndimage.label(road, structure=np.ones((3, 3)))
    near = ndimage.binary_dilation(road, iterations=15)
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        pad = 45
        r0, c0 = max(sl[0].start - pad, 0), max(sl[1].start - pad, 0)
        r1, c1 = sl[0].stop + pad, sl[1].stop + pad
        seg = lab[r0:r1, c0:c1] == i
        if seg.sum() < 100:                       # < 100 m2 at 1 m
            continue
        ring = ndimage.binary_dilation(seg, iterations=40) & ~near[r0:r1, c0:c1] & (gt[r0:r1, c0:c1] >= 1)
        sub = z[r0:r1, c0:c1]
        ok = ~np.isnan(sub)
        out.append(dict(segment=i, auc=round(auc(sub[seg & ok], sub[ring & ok]), 3)))
    return out


def overlay(z, gt, path, window=None):
    v = np.clip(np.nan_to_num(z, nan=0) / 4.0, 0, 1)          # z 0..4 -> black..white
    rgb = np.stack([v * 255] * 3, -1).astype(np.uint8)
    if gt is not None:
        road = gt == 2
        rgb[road & ~ndimage.binary_erosion(road, iterations=1)] = [0, 220, 255]
    if window:
        rgb = rgb[window[0]:window[1], window[2]:window[3]]
    img = Image.fromarray(rgb)
    img.thumbnail((1600, 1600))
    img.save(path, quality=85)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dem, valid, prof = load_1m()
    resp, bearing = detect(dem, valid)
    z = local_z(resp).astype(np.float32)

    with rasterio.open("output/ground_truth/fig11_roads.tif") as s:
        gt05 = s.read(1)
    h, w = dem.shape
    gt = gt05[:h * 2, :w * 2].reshape(h, 2, w, 2).max(axis=(1, 3))     # any road in the 2x2 block

    for name, arr in (("road_z", z), ("road_bearing", bearing), ("road_dip_cm", resp * 100)):
        with rasterio.open(OUT / f"{name}.tif", "w", **prof) as dst:
            dst.write(arr.astype(np.float32), 1)

    segs = score(z, gt)
    v = np.array([s["auc"] for s in segs])
    report = dict(settings=dict(version=2, cap_m=CAP_M, valley_m=VALLEY_M, cell_m=CELL, length_m=LENGTH_M, bearings=N_BEARINGS,
                                centre_m=CENTRE_M, shoulder_m=SHOULDER_M, local_m=LOCAL_M),
                  segments=segs, median_auc=float(np.median(v)),
                  above_0_6=int((v > 0.6).sum()), above_0_7=int((v > 0.7).sum()), n=len(v))
    (OUT / "scores.json").write_text(json.dumps(report, indent=2))
    print(f"detector: median segment AUC {report['median_auc']:.3f}, "
          f"{report['above_0_7']}/{len(v)} above 0.7, {report['above_0_6']}/{len(v)} above 0.6", flush=True)

    rows, cols = np.where(gt >= 1)
    win = (rows.min(), rows.max(), cols.min(), cols.max())
    overlay(z, gt, OUT / "overlay_fig11_area.jpg", win)
    overlay(z, None, OUT / "overlay_fig11_area_clean.jpg", win)
    overlay(z, None, OUT / "overlay_whole_aoi.jpg")


if __name__ == "__main__":
    main()
