"""Run the road detector + chaining on a window of any 1 m 3DEP tile.

Same settings as the Chaco runs (scripts/road_detector.py v2, scripts/chain_roads.py).
Usage: .venv/bin/python scripts/run_area.py NAME TILE.tif LON LAT [SIZE_KM]

Outputs: output/areas/NAME/ road_z.tif, road_chains.geojson, road_ends.geojson,
         overlay.jpg (hillshade, chains >= 150 m orange, ends yellow), summary.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw
from rasterio.warp import transform as warp
from rasterio.windows import Window

import road_detector as rd
from chain_roads import pieces_from, chain, describe

SHOW_MIN_M = 150


def hillshade(dem, az=315, alt=45):
    dy, dx = np.gradient(dem, 1.0)
    slope = np.pi / 2 - np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    a, h = np.radians(360 - az + 90), np.radians(alt)
    return np.sin(h) * np.sin(slope) + np.cos(h) * np.cos(slope) * np.cos(a - aspect)


def main():
    name, tile, lon, lat = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
    size = int(float(sys.argv[5]) * 1000) if len(sys.argv) > 5 else 4000
    out = Path("output/areas") / name
    out.mkdir(parents=True, exist_ok=True)

    with rasterio.open(tile) as s:
        (x,), (y,) = warp("EPSG:4326", s.crs, [lon], [lat])
        r, c = s.index(x, y)
        win = Window(max(c - size // 2, 0), max(r - size // 2, 0), size, size).intersection(
            Window(0, 0, s.width, s.height))
        dem = s.read(1, window=win).astype(np.float64)
        nd, crs, tr = s.nodata, s.crs, s.window_transform(win)
        prof = s.profile
    valid = dem != nd
    dem = np.where(valid, dem, dem[valid].mean())
    print(f"{name}: window {dem.shape} at ({lon}, {lat}), valid {valid.mean():.0%}", flush=True)

    resp, bearing = rd.detect(dem, valid)
    z = rd.local_z(resp).astype(np.float32)
    prof.update(height=z.shape[0], width=z.shape[1], transform=tr, dtype="float32",
                count=1, compress="lzw", nodata=None)
    with rasterio.open(out / "road_z.tif", "w", **prof) as dst:
        dst.write(z, 1)

    pieces = pieces_from(z, bearing, 1.0)
    chains = [describe(g, pieces, resp * 100, 1.0) for g in chain(pieces)] if pieces else []
    xy = lambda rc: [round(v, 2) for v in (tr * (rc[1] + 0.5, rc[0] + 0.5))]
    crs_obj = {"type": "name", "properties": {"name": crs.to_string()}}
    feats, ends = [], []
    for k, ch in enumerate(sorted(chains, key=lambda c: -c["length_m"])):
        props = {kk: vv for kk, vv in ch.items() if not kk.startswith("end_")}
        props["id"] = k
        feats.append(dict(type="Feature", properties=props,
                          geometry=dict(type="LineString", coordinates=[xy(ch["end_a_rc"]), xy(ch["end_b_rc"])])))
        for e in ("a", "b"):
            ends.append(dict(type="Feature", properties=dict(chain_id=k, end=e, length_m=ch["length_m"]),
                             geometry=dict(type="Point", coordinates=xy(ch[f"end_{e}_rc"]))))
    (out / "road_chains.geojson").write_text(json.dumps(dict(type="FeatureCollection", crs=crs_obj, features=feats)))
    (out / "road_ends.geojson").write_text(json.dumps(dict(type="FeatureCollection", crs=crs_obj, features=ends)))

    lengths = np.array([c["length_m"] for c in chains]) if chains else np.zeros(0)
    summary = dict(name=name, tile=tile, centre_lonlat=[lon, lat], size_m=list(dem.shape),
                   pieces=len(pieces), chains=len(chains),
                   chains_ge_150m=int((lengths >= 150).sum()), chains_ge_300m=int((lengths >= 300).sum()),
                   z_ge_2_share=float(np.nanmean(z >= 2)))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)

    hs = hillshade(dem)
    g = (np.clip((hs - np.percentile(hs, 1)) / (np.percentile(hs, 99) - np.percentile(hs, 1)), 0, 1) * 200 + 30)
    rgb = np.stack([g] * 3, -1).astype(np.uint8)
    im = Image.fromarray(rgb)
    d = ImageDraw.Draw(im)
    for ch in chains:
        if ch["length_m"] < SHOW_MIN_M:
            continue
        a, b = ch["end_a_rc"], ch["end_b_rc"]
        d.line([(a[1], a[0]), (b[1], b[0])], fill=(255, 120, 0), width=5)
        for e in (a, b):
            d.ellipse([e[1] - 9, e[0] - 9, e[1] + 9, e[0] + 9], outline=(255, 230, 0), width=3)
    im.thumbnail((1600, 1600))
    im.save(out / "overlay.jpg", quality=88)
    hsi = Image.fromarray(rgb)
    hsi.thumbnail((1600, 1600))
    hsi.save(out / "hillshade.jpg", quality=88)


if __name__ == "__main__":
    main()
