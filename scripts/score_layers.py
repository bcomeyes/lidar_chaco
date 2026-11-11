"""Score the notebook's detection layers against Fig. 11 ground truth.

Layers are recomputed exactly as in the notebook (Sections 6-8):
  composite = 255 - multidirectional hillshade   ("shadowed")
  lrm       = -(gaussian(dem,1.5) - uniform(dem,50))   ("sunken")
  fusion    = clip(-lrm,0,.25)/.25 * (255-multi)/255

Metrics (higher = better, 0.5 = chance):
  global AUC     road cells vs all other cells in the Fig. 11 footprint
  segment AUC    each marked segment vs its own 15-40 m surroundings

Inputs : output/output_be.tif, output/hillshade_multidirectional.tif,
         output/ground_truth/fig11_roads.tif (from fig11_ground_truth.py)
Outputs: output/ground_truth/scores.json, scores_segments.csv
"""
import csv
import json
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage
from scipy.ndimage import gaussian_filter, uniform_filter

OUT = Path("output/ground_truth")
RNG = np.random.default_rng(0)


def auc(pos, neg, n=400_000):
    """Mann-Whitney AUC on (sub)samples."""
    pos = RNG.choice(pos, min(n, pos.size), replace=False)
    neg = RNG.choice(neg, min(n, neg.size), replace=False)
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    _, start, counts = np.unique(allv[order], return_index=True, return_counts=True)
    r = np.empty(allv.size)
    r[order] = np.repeat(start + (counts + 1) / 2.0, counts)   # average ranks for ties
    return float((r[:pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def main():
    with rasterio.open("output/output_be.tif") as s:
        dem = s.read(1).astype(np.float64)
        nd = s.nodata
    dem[dem == nd] = np.nan
    dem = np.nan_to_num(dem, nan=np.nanmean(dem))
    with rasterio.open("output/hillshade_multidirectional.tif") as s:
        multi = s.read(1).astype(np.float64)
    with rasterio.open(OUT / "fig11_roads.tif") as s:
        gt = s.read(1)

    lrm = gaussian_filter(dem, 1.5) - uniform_filter(dem, size=50)
    layers = {
        "composite": 255.0 - multi,
        "lrm": -lrm,
        "fusion": np.clip(-lrm, 0, 0.25) / 0.25 * (255.0 - multi) / 255.0,
    }
    del dem

    road = gt == 2
    area = gt >= 1
    report = {"road_cells": int(road.sum()), "footprint_cells": int(area.sum()), "global_auc": {}}
    for name, L in layers.items():
        report["global_auc"][name] = auc(L[road], L[area & ~road])
        print(f"global AUC {name}: {report['global_auc'][name]:.3f}", flush=True)

    # per segment: local surroundings 15-40 m (30-80 cells) away, not on any road
    lab, n = ndimage.label(road, structure=np.ones((3, 3)))
    near = ndimage.binary_dilation(road, iterations=30)
    rows = []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        pad = 90
        r0, r1 = max(sl[0].start - pad, 0), sl[0].stop + pad
        c0, c1 = max(sl[1].start - pad, 0), sl[1].stop + pad
        seg = lab[r0:r1, c0:c1] == i
        if seg.sum() < 400:                      # < 100 m2: legend/edge fragments
            continue
        ring = (ndimage.binary_dilation(seg, iterations=80) & ~near[r0:r1, c0:c1]
                & area[r0:r1, c0:c1])
        cy, cx = ndimage.center_of_mass(seg)
        row = dict(segment=i, cells=int(seg.sum()),
                   centre_row=int(cy + r0), centre_col=int(cx + c0))
        for name, L in layers.items():
            sub = L[r0:r1, c0:c1]
            row[f"auc_{name}"] = round(auc(sub[seg], sub[ring]), 3)
        rows.append(row)
    with open(OUT / "scores_segments.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    report["segments"] = len(rows)
    for name in layers:
        v = np.array([r[f"auc_{name}"] for r in rows])
        report[f"segment_auc_{name}"] = dict(median=float(np.median(v)),
                                             above_0_6=int((v > 0.6).sum()),
                                             above_0_7=int((v > 0.7).sum()))
    (OUT / "scores.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "global_auc"}, indent=2))


if __name__ == "__main__":
    main()
