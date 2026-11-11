"""Georeference Varien's community-centre maps (Seeking the Center Place, ch. 8) and read their dots.

The maps are shaded relief with no towns, so:
1. Fig. 8.11 (IMG_0100): its white dots are the AD 1150-1225 centres, the same Crow Canyon
   database as Lipe's Fig. 10.1, already georeferenced (scripts/georef_book_map.py).
   A similarity transform is found by matching the two dot patterns (pairs of dots vote),
   then refined to a homography on mutual-nearest pairs.
2. Fig. 8.7 (IMG_0099, AD 1050-1150, the Chacoan era): printed on the same base relief,
   so its terrain texture is matched to Fig. 8.11 with SIFT + RANSAC (photo to photo),
   then chained through step 1. Local misalignment is measured in 320 px windows.
3. Dots: bright (white) and dark (black) round blobs; on 8.7 only large white dots
   (area >= 150 px) are centres, smaller white hits are pieces of polygon lines.

Outputs: output/book/varien_fig8_7_dots.json, varien_fig8_11_dots.json, *_check.jpg,
         varien_georef.json (transforms + error figures)
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from rasterio.warp import transform as warp
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from georef_book_map import apply, homography                       # noqa: E402

OUT = Path("output/book")
IMG_811, IMG_87 = OUT / "images/IMG_0100.jpeg", OUT / "images/IMG_0099.jpeg"
FRAME_811 = (30, 760, 3000, 2530)            # map area, full-res px (bottom: excludes legend)
FRAME_87 = (10, 260, 3020, 2290)
DROP_811 = [(2252, 1933)]                    # bright summit of Sleeping Ute Mountain, not a dot
UTM = "EPSG:26912"


def gray(path):
    return np.asarray(ImageOps.exif_transpose(Image.open(path)).convert("L"))


def find_dots(path, frame, min_white=60):
    g = gray(path).astype(float)
    loc = ndimage.uniform_filter(g, 61)
    x0, y0, x1, y1 = frame
    out = {}
    for kind, m in (("white", (g - loc > 45) & (g > 150)), ("black", (loc - g > 35) & (g < 80))):
        mm = np.zeros_like(m)
        mm[y0:y1, x0:x1] = m[y0:y1, x0:x1]
        mm = ndimage.binary_opening(mm, iterations=2)
        lab, _ = ndimage.label(mm)
        pts = []
        for i, sl in enumerate(ndimage.find_objects(lab), 1):
            b = lab[sl] == i
            a, (h, w) = b.sum(), b.shape
            if a < (min_white if kind == "white" else 60) or a > 2500 or not (0.5 < h / w < 2) or a / (h * w) < 0.5:
                continue
            cy, cx = ndimage.center_of_mass(b)
            p = (cx + sl[1].start, cy + sl[0].start)
            if all(np.hypot(p[0] - q[0], p[1] - q[1]) > 25 for q in pts):
                pts.append((round(p[0], 1), round(p[1], 1), int(a)))
        out[kind] = pts
    return out


def match_points(P, Q):
    """Similarity from dot-pair votes (photo px, y down -> UTM m), then homography refine."""
    Pf = P * [1, -1]
    I, J = np.triu_indices(len(P), 1)
    K, L = np.triu_indices(len(Q), 1)
    dq = Q[L] - Q[K]
    best = (0,)
    for a in range(len(I)):
        v = Pf[J[a]] - Pf[I[a]]
        lp = np.hypot(*v)
        if lp < 250:
            continue
        for flip in (False, True):
            w = -dq if flip else dq
            s = np.hypot(w[:, 0], w[:, 1]) / lp
            ang = (np.degrees(np.arctan2(w[:, 1], w[:, 0]) - np.arctan2(v[1], v[0])) + 180) % 360 - 180
            for b in np.where((s > 30) & (s < 60) & (np.abs(ang) < 15))[0]:
                th = np.radians(ang[b])
                R = s[b] * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
                dst = Q[L[b]] if flip else Q[K[b]]
                T = (Pf - Pf[I[a]]) @ R.T + dst
                n = (np.min(np.hypot(*(T[:, None] - Q[None]).transpose(2, 0, 1)), axis=1) < 2500).sum()
                if n > best[0]:
                    best = (n, s[b], ang[b], Pf[I[a]], dst, R)
    n, s, ang, src, dst, R = best
    T = (Pf - src) @ R.T + dst
    for tol in (2500, 2000, 1500, 1500):
        D = np.hypot(*(T[:, None] - Q[None]).transpose(2, 0, 1))
        qi = D.argmin(1)
        keep = (D[np.arange(len(P)), qi] < tol) & (D.argmin(0)[qi] == np.arange(len(P)))
        H = homography(P[keep], Q[qi[keep]])
        T = apply(H, P)
    e = np.hypot(*(T[keep] - Q[qi[keep]]).T)
    return H, dict(similarity_matches=int(n), m_per_px=round(float(s), 1), rotation_deg=round(float(ang), 1),
                   pairs=int(keep.sum()), median_m=round(float(np.median(e))), p90_m=round(float(np.percentile(e, 90))))


def sift_to(a_path, a_frame, b_path, b_frame):
    """Homography photo A -> photo B from terrain texture (same printed base map)."""
    def load(p, f):
        a = gray(p)
        m = np.zeros_like(a)
        m[f[1]:f[3], f[0]:f[2]] = 255
        return a, m
    (a, ma), (b, mb) = load(a_path, a_frame), load(b_path, b_frame)
    cl = cv2.createCLAHE(3.0, (16, 16))
    sift = cv2.SIFT_create(nfeatures=20000)
    ka, da = sift.detectAndCompute(cl.apply(a), ma)
    kb, db = sift.detectAndCompute(cl.apply(b), mb)
    good = [x for x, y in cv2.BFMatcher().knnMatch(da, db, k=2) if x.distance < 0.75 * y.distance]
    pa = np.float32([ka[g.queryIdx].pt for g in good])
    pb = np.float32([kb[g.trainIdx].pt for g in good])
    H, inl = cv2.findHomography(pa, pb, cv2.RANSAC, 25.0)
    # local check: texture shift left after warping, in 320 px windows
    w = cv2.warpPerspective(a.astype(np.float32), H, (b.shape[1], b.shape[0]))
    hp = lambda x: cv2.GaussianBlur(x - cv2.GaussianBlur(x, (0, 0), 20), (0, 0), 2)
    W, B = hp(w), hp(b.astype(np.float32))
    shifts = []
    for y in range(b_frame[1], b_frame[3] - 320, 320):
        for x in range(b_frame[0], b_frame[2] - 320, 320):
            if (w[y:y + 320, x:x + 320] == 0).mean() > 0.05:
                continue
            r = cv2.matchTemplate(B[y:y + 320, x:x + 320], W[y + 60:y + 260, x + 60:x + 260], cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(r)
            if mx > 0.3:
                shifts.append(np.hypot(loc[0] - 60, loc[1] - 60))
    return H, dict(sift_inliers=int(inl.sum()), windows=len(shifts),
                   local_shift_px_median=round(float(np.median(shifts)), 1),
                   local_shift_px_p90=round(float(np.percentile(shifts, 90)), 1))


def lonlat(H, pts):
    T = apply(H, np.asarray(pts, float))
    lon, lat = warp(UTM, "EPSG:4326", T[:, 0], T[:, 1])
    return lat, lon


def check_image(path, frame, dots, out):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    d = ImageDraw.Draw(im)
    for kind, col in (("white", (0, 200, 255)), ("black", (255, 60, 0))):
        for k, p in enumerate(dots.get(kind, [])):
            d.ellipse([p[0] - 30, p[1] - 30, p[0] + 30, p[1] + 30], outline=col, width=6)
            d.text((p[0] + 34, p[1] - 20), str(k), fill=col)
    im = im.crop(frame)
    im.thumbnail((1800, 1800))
    im.save(out, quality=88)


def main():
    lip = json.loads((OUT / "fig10_1_dots.json").read_text())["dots"]
    qx, qy = warp("EPSG:4326", UTM, [d["lon"] for d in lip], [d["lat"] for d in lip])
    Q = np.c_[qx, qy]

    d811 = find_dots(IMG_811, FRAME_811)
    for k in ("white", "black"):
        d811[k] = [p for p in d811[k] if all(np.hypot(p[0] - x, p[1] - y) > 40 for x, y in DROP_811)]
    H811, rep811 = match_points(np.array([p[:2] for p in d811["white"]]), Q)
    print("Fig 8.11 -> UTM via Lipe 10.1 dots:", rep811, flush=True)

    H87, rep87 = sift_to(IMG_87, FRAME_87, IMG_811, FRAME_811)
    print("Fig 8.7 -> Fig 8.11 via terrain:", rep87, "(1 px ~ 44 m)", flush=True)
    d87 = find_dots(IMG_87, FRAME_87, min_white=150)
    d87 = {"white": d87["white"]}                     # 8.7 shows centres as white dots only

    for name, dots, H, period in (("fig8_11", d811, H811, None), ("fig8_7", d87, H811 @ H87, "AD 1050-1150")):
        rows = []
        for kind, pts in dots.items():
            if not pts:
                continue
            lat, lon = lonlat(H, [p[:2] for p in pts])
            per = period or {"white": "AD 1150-1225", "black": "AD 1225-1290 (nearest neighbours)"}[kind]
            rows += [dict(px=p[:2], kind=kind, period=per, lat=round(float(a), 5), lon=round(float(o), 5))
                     for p, a, o in zip(pts, lat, lon)]
        (OUT / f"varien_{name}_dots.json").write_text(json.dumps(rows, indent=1))
        print(f"{name}: {len(rows)} dots", flush=True)
    check_image(IMG_811, FRAME_811, d811, OUT / "varien_fig8_11_check.jpg")
    check_image(IMG_87, FRAME_87, d87, OUT / "varien_fig8_7_check.jpg")
    (OUT / "varien_georef.json").write_text(json.dumps(dict(
        fig8_11_to_utm=H811.tolist(), fig8_7_to_fig8_11=H87.tolist(), crs=UTM,
        fig8_11_fit=rep811, fig8_7_fit=rep87), indent=1))


if __name__ == "__main__":
    main()
