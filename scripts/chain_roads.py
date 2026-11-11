"""Chain the detector's straight pieces into road lines, and record where each ends.

Roads are judged on what is there (straight, shallow, consistent, long), never on
where they go: some Chacoan roads just stop (the Great North Road ends at the Kutz
Canyon badlands). Every chain's two endpoints are written out as "ended here" points.

1. pieces: cells with z >= Z_MIN, split by detector bearing; each connected group
   that is straight and >= PIECE_MIN_M long is a piece.
2. chains: pieces with matching direction (<= ANGLE_TOL deg), lying on the same line
   (<= OFFSET_M sideways) and separated by <= GAP_M along it are joined.
   Gaps up to GAP_M are bridged (erosion breaks roads); beyond that the road ends.
3. each chain: length, bearing, mean dip, coverage (share of its length with signal).

Inputs : output/detector/road_z.tif, road_bearing.tif, road_dip_cm.tif,
         output/ground_truth/fig11_roads.tif (for scoring only)
Outputs: output/chains/road_chains.geojson, road_ends.geojson, chain_score.tif,
         scores.json, overlay*.jpg
"""
import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw
from scipy import ndimage

from score_layers import auc

OUT = Path("output/chains")
DET = Path("output/detector")
Z_MIN = 2.0
PIECE_MIN_M = 20
ANGLE_TOL = 10.0
OFFSET_M = 6.0
GAP_M = 40.0
MIN_LENGTHS_M = (100, 150, 200, 300)


def pieces_from(z, bearing, cell):
    m = (z >= Z_MIN) & ~np.isnan(z)
    out = []
    for ang in np.unique(bearing[m]):
        lab, n = ndimage.label(m & (bearing == ang), np.ones((3, 3)))
        for i, sl in enumerate(ndimage.find_objects(lab), start=1):
            r, c = np.where(lab[sl] == i)
            if r.size < PIECE_MIN_M / cell:
                continue
            pts = np.c_[r + sl[0].start, c + sl[1].start].astype(float) * cell
            ctr = pts.mean(0)
            w, v = np.linalg.eigh(np.cov((pts - ctr).T))
            u = v[:, 1]
            t = (pts - ctr) @ u
            if t.max() - t.min() < PIECE_MIN_M or np.sqrt(max(w[0], 0)) * 4 > (t.max() - t.min()) / 3:
                continue                                   # too short or not line-like
            out.append(dict(pts=pts, ctr=ctr, u=u, t0=t.min(), t1=t.max()))
    return out


def chain(pieces):
    n = len(pieces)
    parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    ctr = np.array([p["ctr"] for p in pieces])
    u = np.array([p["u"] for p in pieces])
    half = np.array([(p["t1"] - p["t0"]) / 2 for p in pieces])
    mid = ctr + u * np.array([(p["t0"] + p["t1"]) / 2 for p in pieces])[:, None]
    cos_tol = np.cos(np.deg2rad(ANGLE_TOL))
    for i in range(n):
        d = mid - mid[i]
        along = d @ u[i]
        side = np.abs(d @ np.array([-u[i][1], u[i][0]]))
        gap = np.abs(along) - half - half[i]
        ok = (np.abs(u @ u[i]) >= cos_tol) & (side <= OFFSET_M) & (gap <= GAP_M)
        ok[i] = False
        for j in np.where(ok)[0]:
            parent[find(j)] = find(i)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def describe(members, pieces, dip, cell):
    pts = np.vstack([pieces[i]["pts"] for i in members])
    ctr = pts.mean(0)
    _, v = np.linalg.eigh(np.cov((pts - ctr).T))
    u = v[:, 1]
    t = (pts - ctr) @ u
    t0, t1 = t.min(), t.max()
    covered = np.unique(np.floor((t - t0) / 5.0)).size * 5.0       # 5 m bins with signal
    length = t1 - t0
    rc = (pts / cell).astype(int)
    a, b = ctr + u * t0, ctr + u * t1
    # bearing clockwise from grid north; rows increase southward
    brg = (np.degrees(np.arctan2(u[1], -u[0])) + 360) % 180
    return dict(length_m=round(float(length), 1), coverage=round(float(min(covered / max(length, 1), 1)), 2),
                bearing_deg=round(float(brg), 1),
                mean_dip_cm=round(float(np.nanmean(dip[rc[:, 0], rc[:, 1]])), 1),
                end_a_rc=(a / cell).tolist(), end_b_rc=(b / cell).tolist(), pieces=len(members))


def to_xy(rc, transform):
    x, y = transform * (rc[1] + 0.5, rc[0] + 0.5)
    return [round(x, 2), round(y, 2)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with rasterio.open(DET / "road_z.tif") as s:
        z = s.read(1)
        prof, transform, crs = s.profile, s.transform, s.crs
    with rasterio.open(DET / "road_bearing.tif") as s:
        bearing = s.read(1)
    with rasterio.open(DET / "road_dip_cm.tif") as s:
        dip = s.read(1)
    cell = transform.a

    pieces = pieces_from(z, bearing, cell)
    groups = chain(pieces)
    chains = [describe(g, pieces, dip, cell) for g in groups]
    print(f"pieces {len(pieces)}, chains {len(chains)}", flush=True)

    feats, ends = [], []
    for k, c in enumerate(sorted(chains, key=lambda c: -c["length_m"])):
        a, b = to_xy(c["end_a_rc"], transform), to_xy(c["end_b_rc"], transform)
        props = {kk: vv for kk, vv in c.items() if not kk.startswith("end_")}
        props["id"] = k
        feats.append(dict(type="Feature", properties=props,
                          geometry=dict(type="LineString", coordinates=[a, b])))
        for e, xy in (("a", a), ("b", b)):
            ends.append(dict(type="Feature", properties=dict(chain_id=k, end=e, length_m=c["length_m"]),
                             geometry=dict(type="Point", coordinates=xy)))
        c["id"] = k
    crs_obj = {"type": "name", "properties": {"name": "EPSG:6342"}}
    (OUT / "road_chains.geojson").write_text(json.dumps(dict(type="FeatureCollection", crs=crs_obj, features=feats)))
    (OUT / "road_ends.geojson").write_text(json.dumps(dict(type="FeatureCollection", crs=crs_obj, features=ends)))

    # chain-length map: each chain drawn 5 m wide, value = its length in metres
    img = Image.new("F", (z.shape[1], z.shape[0]), 0.0)
    draw = ImageDraw.Draw(img)
    for c in sorted(chains, key=lambda c: c["length_m"]):
        a, b = c["end_a_rc"], c["end_b_rc"]
        draw.line([(a[1], a[0]), (b[1], b[0])], fill=float(c["length_m"]), width=max(int(5 / cell), 1))
    score_map = np.asarray(img, dtype=np.float32)
    p = prof.copy()
    p.update(dtype="float32", count=1, compress="lzw", nodata=None)
    with rasterio.open(OUT / "chain_score.tif", "w", **p) as dst:
        dst.write(score_map, 1)

    # scoring against Fig. 11, inside its footprint (1 m grid)
    with rasterio.open("output/ground_truth/fig11_roads.tif") as s:
        g = s.read(1)
    h, w = z.shape
    gt = g[:h * 2, :w * 2].reshape(h, 2, w, 2).max(axis=(1, 3))
    road, area = gt == 2, gt >= 1
    report = dict(settings=dict(z_min=Z_MIN, piece_min_m=PIECE_MIN_M, angle_tol=ANGLE_TOL,
                                offset_m=OFFSET_M, gap_m=GAP_M),
                  pieces=len(pieces), chains=len(chains), by_min_length={})
    near = ndimage.binary_dilation(road, iterations=8)          # within ~8 m of a marked road
    lab, _ = ndimage.label(road, np.ones((3, 3)))
    seg_ids = [i for i in range(1, lab.max() + 1) if (lab == i).sum() >= 100]
    for L in MIN_LENGTHS_M:
        keep = (score_map >= L) & area
        found = keep & near
        hit = [i for i in seg_ids if (keep & (lab == i)).sum() / (lab == i).sum() > 0.2]
        # fraction of each marked road's cells that a kept chain covers
        cov = (keep & road).sum() / road.sum()
        prec = found.sum() / max(keep.sum(), 1)
        report["by_min_length"][L] = dict(
            chains_in_area=int(len({c['id'] for c in chains if c['length_m'] >= L})),
            road_cells_covered=round(float(cov), 3), roads_hit=f"{len(hit)}/{len(seg_ids)}",
            share_of_chain_on_marked_roads=round(float(prec), 3))
        print(f"min length {L:3d} m: covers {cov:.0%} of marked road, hits {len(hit)}/{len(seg_ids)} roads, "
              f"{prec:.0%} of detected line lies on a marked road", flush=True)
    (OUT / "scores.json").write_text(json.dumps(report, indent=2))

    # overlays: chains >= 150 m in orange, paper's roads in cyan, ends as dots
    rows, cols = np.where(area)
    r0, r1, c0, c1 = rows.min(), rows.max(), cols.min(), cols.max()
    base = np.clip(np.nan_to_num(z, nan=0) / 4, 0, 1)[r0:r1, c0:c1] * 90
    rgb = np.stack([base] * 3, -1).astype(np.uint8)
    edge = road & ~ndimage.binary_erosion(road)
    rgb[edge[r0:r1, c0:c1]] = [0, 200, 255]
    im = Image.fromarray(rgb)
    d = ImageDraw.Draw(im)
    for c in chains:
        if c["length_m"] < 150:
            continue
        a, b = c["end_a_rc"], c["end_b_rc"]
        d.line([(a[1] - c0, a[0] - r0), (b[1] - c0, b[0] - r0)], fill=(255, 140, 0), width=3)
        for e in (a, b):
            d.ellipse([e[1] - c0 - 5, e[0] - r0 - 5, e[1] - c0 + 5, e[0] - r0 + 5], outline=(255, 255, 0), width=2)
    im.thumbnail((1600, 1600))
    im.save(OUT / "overlay_fig11_area.jpg", quality=88)


if __name__ == "__main__":
    main()
