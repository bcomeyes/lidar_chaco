"""Georeference a photographed book map from town markers, then find site dots.

Usage: .venv/bin/python scripts/georef_book_map.py CONFIG.json
CONFIG: {"image": ..., "name": ..., "control": {"Town": [x, y, lat, lon], ...},
         "frame": [x0, y0, x1, y1], "exclude": [[x0, y0, x1, y1], ...]}
Pixel coords are full-resolution, after EXIF rotation.

A projective transform (homography) maps photo pixels to lon/lat; it absorbs the
camera's perspective but not page curl, so leave-one-out errors are reported in km.
Dots = solid dark round blobs that survive a 5 px erosion (thin text strokes do not).

Outputs: output/book/<name>_dots.json, <name>_check.jpg
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage

KM_PER_DEG_LAT = 111.0


def homography(src, dst):
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    _, _, vt = np.linalg.svd(np.array(A))
    return vt[-1].reshape(3, 3) / vt[-1][-1]


def apply(H, pts):
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:]


def km_err(a, b, lat0):
    dx = (a[:, 0] - b[:, 0]) * KM_PER_DEG_LAT * np.cos(np.radians(lat0))
    dy = (a[:, 1] - b[:, 1]) * KM_PER_DEG_LAT
    return np.hypot(dx, dy)


def main():
    cfg = json.loads(Path(sys.argv[1]).read_text())
    out = Path("output/book")
    im = ImageOps.exif_transpose(Image.open(cfg["image"])).convert("RGB")
    rgb = np.asarray(im).astype(int)

    names = list(cfg["control"])
    src = np.array([cfg["control"][n][:2] for n in names], float)
    dst = np.array([[cfg["control"][n][3], cfg["control"][n][2]] for n in names], float)   # lon, lat
    H = homography(src, dst)
    lat0 = dst[:, 1].mean()
    fit_err = km_err(apply(H, src), dst, lat0)
    loo = []
    for i in range(len(src)):
        keep = np.arange(len(src)) != i
        Hi = homography(src[keep], dst[keep])
        loo.append(km_err(apply(Hi, src[i:i + 1]), dst[i:i + 1], lat0)[0])
    print("control fit (km):", dict(zip(names, fit_err.round(2))))
    print("leave-one-out (km):", dict(zip(names, np.round(loo, 2))))

    # dots: dark, not pink, solid
    gray = rgb.mean(axis=2)
    pink = (rgb[..., 0] - rgb[..., 1] > 40)
    dark = (gray < 95) & ~pink
    x0, y0, x1, y1 = cfg["frame"]
    mask = np.zeros_like(dark)
    mask[y0:y1, x0:x1] = True
    for ex in cfg.get("exclude", []):
        mask[ex[1]:ex[3], ex[0]:ex[2]] = False
    solid = ndimage.binary_opening(dark & mask, structure=np.ones((3, 3)), iterations=cfg.get("erode", 5))
    lab, n = ndimage.label(solid)
    dots = []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        blob = lab[sl] == i
        h, w = blob.shape
        area = blob.sum()
        if area < cfg.get("min_area", 120) or not (0.6 < h / w < 1.6) or area / (h * w) < 0.6:
            continue
        cy, cx = ndimage.center_of_mass(blob)
        cx, cy = cx + sl[1].start, cy + sl[0].start
        # a dot stands alone: little ink in a ring around it (letters have neighbours,
        # river wiggles continue into lines)
        r = np.sqrt(area / np.pi)
        r_in, r_out = r + 4, r + 4 + cfg.get("ring_px", 14)
        y0r, y1r = int(cy - r_out), int(cy + r_out) + 1
        x0r, x1r = int(cx - r_out), int(cx + r_out) + 1
        if y0r < 0 or x0r < 0 or y1r > dark.shape[0] or x1r > dark.shape[1]:
            continue
        yy, xx = np.mgrid[y0r:y1r, x0r:x1r]
        rr = np.hypot(yy - cy, xx - cx)
        ring = (rr >= r_in) & (rr <= r_out)
        ink = dark[y0r:y1r, x0r:x1r][ring].mean()
        if ink > cfg.get("max_ring_ink", 0.12):
            continue
        # size class from the original dark blob around it
        d = float(2 * np.sqrt(ndimage.binary_dilation(blob, iterations=5).sum() / np.pi))
        lon, lat = apply(H, np.array([[cx, cy]]))[0]
        dots.append(dict(px=[round(cx, 1), round(cy, 1)], diameter_px=round(d, 1),
                         lat=round(float(lat), 5), lon=round(float(lon), 5)))
    # manual corrections, given in check-image coordinates (after eyeballing it)
    scale = min(1800 / (x1 - x0), 1800 / (y1 - y0))
    to_full = lambda t: (x0 + t[0] / scale, y0 + t[1] / scale)
    for t in cfg.get("drop_thumb", []):
        fx, fy = to_full(t)
        dots = [d for d in dots if np.hypot(d["px"][0] - fx, d["px"][1] - fy) > 25]
    for t in cfg.get("add_thumb", []):
        fx, fy = to_full(t)
        win = dark[int(fy) - 25:int(fy) + 26, int(fx) - 25:int(fx) + 26]
        lab2, _ = ndimage.label(win)
        if lab2.max() == 0:
            print("no ink near added point", t)
            continue
        big = np.argmax(ndimage.sum(win, lab2, range(1, lab2.max() + 1))) + 1
        cy, cx = ndimage.center_of_mass(lab2 == big)
        cx, cy = cx + int(fx) - 25, cy + int(fy) - 25
        lon, lat = apply(H, np.array([[cx, cy]]))[0]
        dots.append(dict(px=[round(cx, 1), round(cy, 1)], diameter_px=round(float(2 * np.sqrt((lab2 == big).sum() / np.pi)), 1),
                         lat=round(float(lat), 5), lon=round(float(lon), 5), added_by_hand=True))
    dia = np.array([d["diameter_px"] for d in dots])
    print(f"{len(dots)} dots; diameters px: {np.sort(dia).round(0).tolist()}")
    (out / f"{cfg['name']}_dots.json").write_text(json.dumps(dict(
        image=cfg["image"], control=cfg["control"], homography=H.tolist(),
        control_fit_km=dict(zip(names, fit_err.round(3).tolist())),
        leave_one_out_km=dict(zip(names, np.round(loo, 3).tolist())), dots=dots), indent=1))

    chk = im.copy()
    dr = ImageDraw.Draw(chk)
    for k, d in enumerate(dots):
        x, y = d["px"]
        dr.ellipse([x - 30, y - 30, x + 30, y + 30], outline=(0, 160, 255), width=6)
        dr.text((x + 34, y - 20), str(k), fill=(0, 100, 255))
    for nme in names:
        x, y = cfg["control"][nme][:2]
        dr.rectangle([x - 25, y - 25, x + 25, y + 25], outline=(0, 200, 0), width=6)
    chk = chk.crop((x0, y0, x1, y1))
    chk.thumbnail((1800, 1800))
    chk.save(out / f"{cfg['name']}_check.jpg", quality=88)


if __name__ == "__main__":
    main()
