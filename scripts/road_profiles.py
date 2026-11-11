"""Stacked cross-sections: is each marked road a real dip in the ground, and does it
survive at 1 m resolution?

For each straight Fig. 11 segment: find its direction, cut elevation profiles straight
across it every 2 m along its length (-30..+30 m), remove each profile's local tilt
(fitted on the outer 15-30 m), and average. Repeat on the DTM coarsened to 1 m.

Inputs : output/output_be.tif, output/ground_truth/fig11_roads.tif
Outputs: output/ground_truth/road_profiles.png, road_profiles.json
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from scipy import ndimage

OUT = Path("output/ground_truth")
OFFSETS = np.arange(-30, 30.01, 0.5)                    # metres across the road
OUTER = np.abs(OFFSETS) >= 15


def coarsen(dem, factor):
    """Block-average to factor x the cell size, then back to the original grid."""
    h, w = (dem.shape[0] // factor) * factor, (dem.shape[1] // factor) * factor
    small = dem[:h, :w].reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))
    big = ndimage.zoom(small, factor, order=1)
    out = dem.copy()
    out[:big.shape[0], :big.shape[1]] = big
    return out


def stacked_profile(dem, rows, cols, cell, shift_m=0.0):
    """Average detrended cross-section of the line through (rows, cols),
    optionally moved shift_m sideways (a control line with no road)."""
    pts = np.c_[rows, cols].astype(float)
    centre = pts.mean(0)
    _, vecs = np.linalg.eigh(np.cov((pts - centre).T))
    along, across = vecs[:, 1], vecs[:, 0]
    t = (pts - centre) @ along
    stations = np.arange(t.min() + 4, t.max() - 4, 2 / cell)
    profs = []
    for s in stations:
        # centre on the drawn line at this station
        near = np.abs(t - s) < 2 / cell
        if near.sum() < 3:
            continue
        c = pts[near].mean(0) + shift_m / cell * across
        rr = c[0] + OFFSETS[:, None] / cell * across[0]
        cc = c[1] + OFFSETS[:, None] / cell * across[1]
        z = ndimage.map_coordinates(dem, [rr.ravel(), cc.ravel()], order=1)
        fit = np.polyfit(OFFSETS[OUTER], z[OUTER], 1)
        profs.append(z - np.polyval(fit, OFFSETS))
    return np.array(profs), len(stations) * 2


def main():
    with rasterio.open("output/output_be.tif") as s:
        dem = s.read(1).astype(np.float64)
        nd, cell = s.nodata, s.res[0]
    dem[dem == nd] = np.nan
    dem = np.nan_to_num(dem, nan=np.nanmean(dem))
    with rasterio.open(OUT / "fig11_roads.tif") as s:
        road = s.read(1) == 2
    dems = {"0.5 m": dem, "1 m": coarsen(dem, 2)}

    lab, n = ndimage.label(road, structure=np.ones((3, 3)))
    results = []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        rows, cols = np.where(lab[sl] == i)
        if rows.size < 400:
            continue
        rows, cols = rows + sl[0].start, cols + sl[1].start
        ev = np.linalg.eigvalsh(np.cov(np.c_[rows, cols].T))
        if ev[1] / max(ev[0], 1e-9) < 100:              # not a single straight line
            continue
        res = dict(segment=i, centre_rowcol=[int(rows.mean()), int(cols.mean())])
        for name, d in dems.items():
            profs, length_m = stacked_profile(d, rows, cols, cell)
            mean = profs.mean(0)
            se = profs.std(0) / np.sqrt(len(profs))
            centre = np.abs(OFFSETS) <= 5
            depth = -mean[centre].min()
            res[name] = dict(n_profiles=len(profs), length_m=float(length_m),
                             depth_cm=round(float(depth) * 100, 1),
                             depth_se_cm=round(float(se[centre][np.argmin(mean[centre])]) * 100, 1),
                             profile_cm=(mean * 100).round(2).tolist())
        ctrl = []
        for sh in (-60.0, 60.0):
            profs, _ = stacked_profile(dems["0.5 m"], rows, cols, cell, shift_m=sh)
            ctrl.append(-profs.mean(0)[np.abs(OFFSETS) <= 5].min())
        res["control_depth_cm"] = round(float(np.mean(ctrl)) * 100, 1)
        results.append(res)
        print(f"segment {i:2d}: {res['0.5 m']['length_m']:5.0f} m long, dip "
              f"{res['0.5 m']['depth_cm']:5.1f} ± {res['0.5 m']['depth_se_cm']:.1f} cm at 0.5 m, "
              f"{res['1 m']['depth_cm']:5.1f} ± {res['1 m']['depth_se_cm']:.1f} cm at 1 m, "
              f"control {res['control_depth_cm']:5.1f} cm", flush=True)

    (OUT / "road_profiles.json").write_text(json.dumps(
        dict(offsets_m=OFFSETS.tolist(), segments=results), indent=1))

    k = len(results)
    cols_n = 4
    fig, axes = plt.subplots(int(np.ceil(k / cols_n)), cols_n, figsize=(16, 3.2 * np.ceil(k / cols_n)),
                             sharex=True, squeeze=False)
    for ax, res in zip(axes.ravel(), results):
        for name, colr in (("0.5 m", "#1f5fa8"), ("1 m", "#d07a12")):
            ax.plot(OFFSETS, res[name]["profile_cm"], color=colr, label=name)
        ax.axhline(0, color="0.7", lw=0.8)
        ax.axvspan(-5, 5, color="0.92")
        ax.set_title(f"seg {res['segment']} ({res['0.5 m']['length_m']:.0f} m)", fontsize=10)
        ax.set_ylabel("cm")
    for ax in axes.ravel()[k:]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("metres across road")
    fig.suptitle("Average cross-section of each Fig. 11 road (grey band = road corridor)")
    fig.tight_layout()
    fig.savefig(OUT / "road_profiles.png", dpi=110)


if __name__ == "__main__":
    main()
