"""Do the aligned pieces sit on roads already on the map (modern roads, tracks, railways)?

Fetches OpenStreetMap highway=* and railway=* ways around each corridor (all 11 lines)
from Overpass, cached by 0.2-degree cell in output/corridor_test/osm_cells/. A piece counts as "on a mapped road"
if at least half its length lies within NEAR_M of a mapped way. The aligned score is
then recomputed with those pieces removed (scripts/aligned_test.py rules).

Outputs: output/corridor_test/mapped_roads_summary.txt
"""
import json
import math
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import aligned_test as at
import corridor_test as ct

OUT = Path(os.environ.get("CORRIDOR_OUT", "output/corridor_test"))
CACHE = Path("output/corridor_test/osm_cells")
CELL_DEG = 0.2
NEAR_M = 15
OVERPASS = "https://overpass-api.de/api/interpreter"


def fetch_cell(i, j):
    f = CACHE / f"cell_{i}_{j}.json"
    if f.exists():
        return json.loads(f.read_text())
    bbox = f"{i * CELL_DEG},{j * CELL_DEG},{(i + 1) * CELL_DEG},{(j + 1) * CELL_DEG}"
    q = f"[out:json][timeout:300];(way[highway]({bbox});way[railway]({bbox}););out geom;"
    for attempt in range(6):
        try:
            req = urllib.request.Request(OVERPASS, data=urllib.parse.urlencode({"data": q}).encode(),
                                         headers={"User-Agent": "lidar_chaco research script"})
            with urllib.request.urlopen(req, timeout=360) as r:
                data = json.load(r)
            break
        except Exception as e:
            print(f"cell {i},{j}: overpass attempt {attempt + 1} failed: {e}", flush=True)
            time.sleep(30 * (attempt + 1))
    else:
        raise RuntimeError("Overpass kept failing")
    ways = [[(p["lon"], p["lat"]) for p in w["geometry"]] for w in data["elements"] if "geometry" in w]
    f.write_text(json.dumps(ways))
    return ways


def fetch(name, lats, lons):
    """Mapped ways around a corridor, from 0.2-degree cells cached in CACHE (shared by corridors)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    ways = []
    for i in range(math.floor((min(lats) - 0.01) / CELL_DEG), math.floor((max(lats) + 0.01) / CELL_DEG) + 1):
        for j in range(math.floor((min(lons) - 0.01) / CELL_DEG), math.floor((max(lons) + 0.01) / CELL_DEG) + 1):
            ways += fetch_cell(i, j)
    return ways


def road_tree(ways):
    pts = []
    for w in ways:
        a = np.array(w)
        x, y = ct.TO_UTM.transform(a[:, 0], a[:, 1])
        xy = np.c_[x, y]
        for p, q in zip(xy[:-1], xy[1:]):
            n = max(int(np.linalg.norm(q - p) / 5), 1)
            pts.append(p + (q - p) * np.linspace(0, 1, n + 1)[:, None])
    return cKDTree(np.concatenate(pts))


def main():
    pairs = json.loads(Path(os.environ.get("CORRIDORS", "output/corridors/great_house_corridors.json")).read_text())
    results = json.loads((OUT / "results.json").read_text())
    sites = {}
    for f in ("sites_chacoan.json", "great_houses.json", "sites.json", "sites_local.json"):
        for s in json.loads((Path("output/corridors") / f).read_text()):
            sites.setdefault(s["name"], s)
    rows, ranks_all, ranks_off = [], [], []
    for p in pairs:
        name = f"{p['a']} - {p['b']}"
        p0, p1 = ct.endpoints(sites[p["a"]], sites[p["b"]])
        lines = {"real": (p0, p1)} | {k: (a, b) for k, a, b in ct.control_lines(p0, p1)}
        ends = np.array([q for ab in lines.values() for q in ab])
        lons, lats = ct.TO_LL.transform(ends[:, 0], ends[:, 1])
        tree = road_tree(fetch(name, lats, lons))
        scores = {}
        on_share = {}
        for r in results:
            if r["corridor"] != name or r.get("km_with_data", 0) <= 1:
                continue
            a, b = lines[r["line"]]
            u = (b - a) / np.linalg.norm(b - a)
            v = np.array([u[1], -u[0]])
            keep, on = [], 0
            for s, e, o in r["pieces"]:
                ss = np.arange(s, e, 10.0)
                xy = a + ss[:, None] * u + o * v
                d, _ = tree.query(xy)
                if np.mean(d <= NEAR_M) >= 0.5:
                    on += 1
                else:
                    keep.append([s, e, o])
            ch_all = at.chains(r["pieces"])
            ch_off = at.chains(keep)
            on_ch = sum(1 for c in ch_all for pc in c if pc not in keep)
            scores[r["line"]] = (sum(e - s for c in ch_all for s, e, _ in c) / r["km_with_data"],
                                 sum(e - s for c in ch_off for s, e, _ in c) / r["km_with_data"])
            on_share[r["line"]] = on_ch / max(sum(len(c) for c in ch_all), 1)
        if "real" not in scores:
            continue
        for i, ranks in ((0, ranks_all), (1, ranks_off)):
            rv = scores["real"][i]
            ctrl = [sc[i] for k, sc in scores.items() if k != "real"]
            ranks.append((sum(c < rv for c in ctrl) + 0.5 * sum(c == rv for c in ctrl), len(ctrl)))
        ctrl_on = np.mean([v for k, v in on_share.items() if k != "real"])
        rows.append(f"{name}: aligned pieces on mapped roads: real {on_share['real']:.0%}, controls {ctrl_on:.0%} | "
                    f"off-road score real {scores['real'][1]:.1f} m/km vs controls median "
                    f"{np.median([sc[1] for k, sc in scores.items() if k != 'real']):.1f} | "
                    f"beats {ranks_off[-1][0]:g} of {ranks_off[-1][1]}")
        print(rows[-1], flush=True)
    rng = np.random.default_rng(0)
    for label, ranks in (("all pieces", ranks_all), ("mapped roads removed", ranks_off)):
        obs = np.mean([b / n for b, n in ranks])
        sims = np.array([np.mean([rng.integers(0, n + 1) / n for _, n in ranks]) for _ in range(20000)])
        rows.append(f"{label}: real lines beat {obs:.0%} of controls; one-sided p = {np.mean(sims >= obs):.3f}")
        print(rows[-1], flush=True)
    (OUT / "mapped_roads_summary.txt").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
