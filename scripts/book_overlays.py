"""All photographed maps (book pages + PBS documentary map) as Google Earth overlays.

Each map is warped onto a lon/lat grid and written as a semi-transparent PNG
GroundOverlay, so pages can be switched on and off and compared with each other and
with Google Earth's own towns, state lines and canyons. Dots read from each map are
added as pins, and the tie points used to place each map as small markers.

How each map is placed (photo pixel -> ground):
- Lipe Fig. 10.1 / 10.2 (IMG_0102, 0103): homography from printed towns (fig10_*.json).
- Varien Fig. 8.11 (IMG_0100): dot pattern matched to Lipe 10.1 (varien_georef.json).
- Varien Fig. 8.7 (IMG_0099): printed terrain matched to Fig. 8.11 with SIFT + RANSAC.
- Varien Fig. 8.8 (IMG_0098, AD 1150-1225 centres): its dots matched to Lipe 10.1's.
- Varien Fig. 8.12 (IMG_0101, centres AD 950-1290): its dots matched to all centres already read
  (Varien 8.7 + 8.11, Lipe 10.1). (Terrain matching fails on these two: different printing.)
- PBS map (chacopictures/IMG_0172): homography from its five green town squares and the
  Four Corners (crossing of the blue state lines). Yellow and magenta dots are read off.

Outputs: output/overlays/*.png, output/overlays/maps.kml, output/overlays/pbs_map_dots.json,
         output/overlays/placement.json (fit errors per map)
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from rasterio.warp import transform as warp
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from georef_book_map import apply, homography, km_err          # noqa: E402
from georef_varien import FRAME_87, FRAME_811, find_dots, match_points, sift_to   # noqa: E402

BOOK = Path("output/book")
IMG = BOOK / "images"
OUT = Path("output/overlays")
UTM = "EPSG:26912"
OUT_W = 1600                                   # overlay width in px
OPACITY = "c0"                                 # KML alpha (hex) for overlays
FRAME_98 = (180, 520, 3000, 1980)              # IMG_0098 = Varien Fig. 8.8
FRAME_101 = (330, 1300, 3900, 3050)            # IMG_0101 = Varien Fig. 8.12
PBS = BOOK / "chacopictures/IMG_0172.PNG"
PBS_FRAME = (0, 975, 1206, 1648)               # the video picture inside the phone screenshot
PBS_CAPTION = (205, 1500, 625, 1605)           # subtitle bubble: dots under it are hidden
PBS_TOWNS = {                                  # green squares (px) and true positions (lat, lon)
    "Cortez": ((746.5, 1003.5), (37.3489, -108.5859)),
    "Shiprock": ((726.5, 1133.0), (36.7856, -108.6870)),
    "Gallup": ((720.5, 1406.5), (35.5281, -108.7426)),
    "Flagstaff": ((70.0, 1478.5), (35.1983, -111.6513)),
    "Albuquerque": ((1183.5, 1500.5), (35.0844, -106.6504)),
}
FOUR_CORNERS = (36.99898, -109.04517)


def load(path):
    return np.asarray(ImageOps.exif_transpose(Image.open(path)).convert("RGB"))


def utm_to_px(H_px_to_utm):
    Hi = np.linalg.inv(H_px_to_utm)

    def f(lon, lat):
        x, y = warp("EPSG:4326", UTM, lon.ravel().tolist(), lat.ravel().tolist())
        return apply(Hi, np.c_[x, y]).reshape(lon.shape + (2,))
    return f


def lonlat_to_px(H_px_to_ll):
    Hi = np.linalg.inv(H_px_to_ll)
    return lambda lon, lat: apply(Hi, np.c_[lon.ravel(), lat.ravel()]).reshape(lon.shape + (2,))


def render(rgb, frame, to_px, corners_ll, name):
    """Warp the framed part of a photo onto a north-up lon/lat grid; outside the frame = transparent."""
    lon0, lat0 = corners_ll[:, 0].min(), corners_ll[:, 1].min()
    lon1, lat1 = corners_ll[:, 0].max(), corners_ll[:, 1].max()
    h = int(OUT_W * (lat1 - lat0) / ((lon1 - lon0) * np.cos(np.radians((lat0 + lat1) / 2))))
    lon, lat = np.meshgrid(np.linspace(lon0, lon1, OUT_W), np.linspace(lat1, lat0, h))
    p = to_px(lon, lat).astype(np.float32)
    img = cv2.remap(rgb, p[..., 0], p[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    x0, y0, x1, y1 = frame
    inside = (p[..., 0] >= x0) & (p[..., 0] < x1) & (p[..., 1] >= y0) & (p[..., 1] < y1)
    Image.fromarray(np.dstack([img, np.where(inside, 255, 0).astype(np.uint8)])).save(OUT / f"{name}.png", optimize=True)
    return dict(north=lat1, south=lat0, east=lon1, west=lon0)


def frame_corners(frame, to_ll):
    x0, y0, x1, y1 = frame
    xs = np.r_[np.linspace(x0, x1, 20), np.full(20, x1), np.linspace(x1, x0, 20), np.full(20, x0)]
    ys = np.r_[np.full(20, y0), np.linspace(y0, y1, 20), np.full(20, y1), np.linspace(y1, y0, 20)]
    return to_ll(np.c_[xs, ys])


def pbs_map():
    rgb = load(PBS)
    names = list(PBS_TOWNS)
    src = [PBS_TOWNS[n][0] for n in names]
    dst = [(PBS_TOWNS[n][1][1], PBS_TOWNS[n][1][0]) for n in names]
    # Four Corners = crossing of the blue state lines
    blue = (rgb[..., 2] > 180) & (rgb[..., 0] < 90) & (rgb[..., 1] < 150)
    x0, y0, x1, y1 = PBS_FRAME
    col = x0 + int(np.argmax(blue[y0:y1, x0:x1].sum(0)))
    rows = np.where(blue[y0:y1, col - 40:col - 10].any(1))[0] + y0
    fc = (col + 0.5, float(np.median(rows)))
    src.append(fc)
    dst.append((FOUR_CORNERS[1], FOUR_CORNERS[0]))
    names.append("Four Corners")
    src, dst = np.array(src, float), np.array(dst, float)
    H = homography(src, dst)
    lat0 = dst[:, 1].mean()
    fit = km_err(apply(H, src), dst, lat0)
    loo = []
    for i in range(len(src)):
        k = np.arange(len(src)) != i
        loo.append(float(km_err(apply(homography(src[k], dst[k]), src[i:i + 1]), dst[i:i + 1], lat0)[0]))
    # state-line check: blue pixels of the vertical line should map to lon -109.045
    vy = np.where(blue[y0:y1, col])[0] + y0
    line_lon = apply(H, np.c_[np.full(vy.size, col + 0.5), vy])[:, 0]
    # dots
    dots = []
    R, G, B = (rgb[..., i].astype(int) for i in range(3))
    for kind, m in (("yellow", (R > 200) & (G > 200) & (B < 90)), ("magenta (Chaco Canyon)", (R > 150) & (B > 150) & (G < 80))):
        mm = np.zeros_like(m)
        mm[y0:y1, x0:x1] = m[y0:y1, x0:x1]
        cx0, cy0, cx1, cy1 = PBS_CAPTION
        mm[cy0:cy1, cx0:cx1] = False
        lab, _ = ndimage.label(mm)
        for i, sl in enumerate(ndimage.find_objects(lab), 1):
            a = (lab[sl] == i).sum()
            if a < 12:
                continue
            cy, cx = ndimage.center_of_mass(lab[sl] == i)
            lon, lat = apply(H, np.array([[cx + sl[1].start, cy + sl[0].start]]))[0]
            dots.append(dict(kind=kind, px=[round(cx + sl[1].start, 1), round(cy + sl[0].start, 1)], area=int(a),
                             lat=round(float(lat), 4), lon=round(float(lon), 4)))
    rep = dict(tie_points=dict(zip(names, [list(map(float, s)) for s in src])),
               fit_km=dict(zip(names, fit.round(1).tolist())), leave_one_out_km=dict(zip(names, np.round(loo, 1).tolist())),
               state_line_lon_median=round(float(np.median(line_lon)), 3), state_line_lon_spread=round(float(np.ptp(line_lon)), 3),
               km_per_px=round(float(km_err(apply(H, np.array([[600, 1300]])), apply(H, np.array([[601, 1300]])), lat0)[0]), 2))
    return rgb, H, rep, dots, list(zip(names, dst.tolist()))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    vg = json.loads((BOOK / "varien_georef.json").read_text())
    H811 = np.array(vg["fig8_11_to_utm"])
    maps, report = [], {}

    def to_ll_utm(Hpx):
        def f(px):
            u = apply(Hpx, px)
            lon, lat = warp(UTM, "EPSG:4326", u[:, 0].tolist(), u[:, 1].tolist())
            return np.c_[lon, lat]
        return f

    # Varien pages
    pages = [("varien_fig8_11", "Varien Fig. 8.11 (AD 1150-1225 centres)", IMG / "IMG_0100.jpeg", FRAME_811, H811, None)]
    Hs, rep = sift_to(IMG / "IMG_0099.jpeg", FRAME_87, IMG / "IMG_0100.jpeg", FRAME_811)
    pages.append(("varien_fig8_7", "Varien Fig. 8.7 (AD 1050-1150 centres, Chacoan era)", IMG / "IMG_0099.jpeg",
                  FRAME_87, H811 @ Hs, dict(placed_by="terrain match to Fig. 8.11 (SIFT)", **rep)))

    def utm(rows):
        x, y = warp("EPSG:4326", UTM, [r["lon"] for r in rows], [r["lat"] for r in rows])
        return np.c_[x, y]
    lip = json.loads((BOOK / "fig10_1_dots.json").read_text())["dots"]
    allc = lip + json.loads((BOOK / "varien_fig8_7_dots.json").read_text()) + json.loads((BOOK / "varien_fig8_11_dots.json").read_text())
    for name, title, img, frame, Q, mw, how in (
            ("varien_fig8_8", "Varien Fig. 8.8 (AD 1150-1225 centres, catchments)", "IMG_0098.jpeg", FRAME_98, utm(lip), 60,
             "dot pattern vs Lipe Fig. 10.1"),
            ("varien_fig8_12", "Varien Fig. 8.12 (persistent communities, AD 950-1290)", "IMG_0101.jpeg", FRAME_101, utm(allc), 40,
             "dot pattern vs all centres read so far")):
        d = find_dots(IMG / img, frame, min_white=mw)
        Hpx, rep = match_points(np.array([p[:2] for p in d["white"]]), Q)
        print(name, how, rep, flush=True)
        pages.append((name, title, IMG / img, frame, Hpx, dict(placed_by=how, **rep)))
    for name, title, img, frame, Hpx, rep in pages:
        rgb = load(img)
        corners = frame_corners(frame, to_ll_utm(Hpx))
        box = render(rgb, frame, utm_to_px(Hpx), corners, name)
        maps.append((name, title, box))
        report[name] = rep or dict(placed_by="dot pattern vs Lipe Fig. 10.1", **vg["fig8_11_fit"])
    # Lipe pages
    for cfgf, title in (("fig10_1.json", "Lipe Fig. 10.1 (AD 1150-1225 large sites)"),
                        ("fig10_2.json", "Lipe Fig. 10.2 (AD 1225-1290 large sites)")):
        cfg = json.loads((BOOK / cfgf).read_text())
        ctl = cfg["control"]
        src = np.array([v[:2] for v in ctl.values()], float)
        dst = np.array([[v[3], v[2]] for v in ctl.values()], float)
        H = homography(src, dst)
        rgb = load(cfg["image"])
        corners = frame_corners(cfg["frame"], lambda px: apply(H, px))
        box = render(rgb, cfg["frame"], lonlat_to_px(H), corners, cfg["name"])
        maps.append((cfg["name"], title, box))
        report[cfg["name"]] = dict(placed_by="printed towns", towns=list(ctl),
                                   fit_km=km_err(apply(H, src), dst, dst[:, 1].mean()).round(1).tolist())
    # PBS map
    rgb, H, rep, dots, ties = pbs_map()
    corners = frame_corners(PBS_FRAME, lambda px: apply(H, px))
    box = render(rgb, PBS_FRAME, lonlat_to_px(H), corners, "pbs_map")
    maps.append(("pbs_map", "PBS documentary map (IMG_0172)", box))
    report["pbs_map"] = dict(placed_by="5 towns + Four Corners", **rep)
    (OUT / "pbs_map_dots.json").write_text(json.dumps(dots, indent=1))
    (OUT / "placement.json").write_text(json.dumps(report, indent=1))
    print("PBS map:", json.dumps(rep), f"{len(dots)} dots", flush=True)

    # KML
    def pin(name, lat, lon, style, desc=""):
        return (f"<Placemark><name>{name}</name><description>{desc}</description><styleUrl>#{style}</styleUrl>"
                f"<Point><coordinates>{lon},{lat},0</coordinates></Point></Placemark>")
    ov = [f"<GroundOverlay><name>{t}</name><visibility>{1 if n in ('pbs_map',) else 0}</visibility>"
          f"<color>{OPACITY}ffffff</color><Icon><href>{n}.png</href></Icon><LatLonBox><north>{b['north']}</north>"
          f"<south>{b['south']}</south><east>{b['east']}</east><west>{b['west']}</west></LatLonBox></GroundOverlay>"
          for n, t, b in maps]
    dot_folders = []
    for f, title, style in (("varien_fig8_7_dots.json", "Varien 8.7 dots (AD 1050-1150)", "red"),
                            ("varien_fig8_11_dots.json", "Varien 8.11 dots (AD 1150-1290)", "orange")):
        pins = [pin("", d["lat"], d["lon"], style, d["period"]) for d in json.loads((BOOK / f).read_text())]
        dot_folders.append(f"<Folder><name>{title}</name><visibility>0</visibility>{''.join(pins)}</Folder>")
    for f, title, style in (("fig10_1_dots.json", "Lipe 10.1 dots (AD 1150-1225)", "blue"),
                            ("fig10_2_dots.json", "Lipe 10.2 dots (AD 1225-1290)", "cyan")):
        pins = [pin("", d["lat"], d["lon"], style) for d in json.loads((BOOK / f).read_text())["dots"]]
        dot_folders.append(f"<Folder><name>{title}</name><visibility>0</visibility>{''.join(pins)}</Folder>")
    pins = [pin("", d["lat"], d["lon"], "yellow" if d["kind"] == "yellow" else "magenta", d["kind"]) for d in dots]
    dot_folders.append(f"<Folder><name>PBS map dots</name>{''.join(pins)}</Folder>")
    ties_kml = [pin(n, lat, lon, "tie", "tie point used to place the PBS map") for n, (lon, lat) in ties]
    for cfgf in ("fig10_1.json", "fig10_2.json"):
        for n, v in json.loads((BOOK / cfgf).read_text())["control"].items():
            ties_kml.append(pin(n, v[2], v[3], "tie", f"tie point used to place {cfgf[:-5]}"))
    style = lambda i, c, s, href="http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png": (
        f"<Style id='{i}'><IconStyle><color>{c}</color><scale>{s}</scale><Icon><href>{href}</href></Icon></IconStyle>"
        f"<LabelStyle><scale>0.7</scale></LabelStyle></Style>")
    (OUT / "maps.kml").write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document><name>Book and documentary maps</name>
{style('red', 'ff0000ff', 0.8)}{style('orange', 'ff00a5ff', 0.7)}{style('blue', 'ffff0000', 0.7)}{style('cyan', 'ffffff00', 0.7)}
{style('yellow', 'ff00ffff', 0.8)}{style('magenta', 'ffff00ff', 0.8)}
{style('tie', 'ff00ff00', 0.9, 'http://maps.google.com/mapfiles/kml/shapes/target.png')}
<Folder><name>Map overlays (tick one at a time; opacity slider in Google Earth)</name>{''.join(ov)}</Folder>
<Folder><name>Dots read from the maps</name>{''.join(dot_folders)}</Folder>
<Folder><name>Tie points (towns, Four Corners)</name><visibility>0</visibility>{''.join(ties_kml)}</Folder>
</Document></kml>
""")
    print(f"wrote {OUT / 'maps.kml'} with {len(maps)} overlays", flush=True)


if __name__ == "__main__":
    main()
