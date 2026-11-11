"""Aligned-pieces score: do road-like pieces line up along the site-to-site lines?

Reads output/corridor_test/results.json (pieces >= 100 m per line, from
scripts/corridor_test.py). A surviving road broken by fields should leave pieces at the
same sideways offset, one after another. Scattered field edges should not line up.

Chaining: two pieces join if the gap between them along the line is <= MAX_GAP_M and
their offsets differ by <= OFF_TOL_M + DRIFT * gap (lets a road run at a slight angle
to the site-to-site line). An aligned chain = >= MIN_PIECES pieces spanning >= MIN_SPAN_M.
Score per line = metres of pieces in aligned chains per km of line with data.
Real lines vs their own shifted/rotated controls, as in corridor_test.py.

Outputs: output/corridor_test/aligned_summary.txt, aligned_results.json, aligned_chains.kml
"""
import json
import os
from pathlib import Path

import numpy as np

import corridor_test as ct

OUT = Path(os.environ.get("CORRIDOR_OUT", "output/corridor_test"))
MAX_GAP_M = 2000
OFF_TOL_M = 6.0
DRIFT = 0.003                 # 3 m per km of gap (~0.2 deg)
MIN_PIECES = 2
MIN_SPAN_M = 1000


def chains(pieces):
    n = len(pieces)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(n):
            si, ei, oi = pieces[i]
            sj, ej, oj = pieces[j]
            gap = sj - ei
            if i != j and 0 <= gap <= MAX_GAP_M and abs(oj - oi) <= OFF_TOL_M + DRIFT * gap:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(pieces[i])
    out = []
    for g in groups.values():
        span = max(p[1] for p in g) - min(p[0] for p in g)
        if len(g) >= MIN_PIECES and span >= MIN_SPAN_M:
            out.append(sorted(g))
    return out


def main():
    results = json.loads((OUT / "results.json").read_text())
    rows, ranks, kml = [], [], []
    for r in results:
        if r.get("km_with_data", 0) <= 1:
            continue
        ch = chains(r["pieces"])
        r["aligned"] = ch
        r["aligned_m_per_km"] = round(sum(e - s for c in ch for s, e, _ in c) / r["km_with_data"], 2)
    for name in dict.fromkeys(r["corridor"] for r in results):
        rs = [r for r in results if r["corridor"] == name and "aligned_m_per_km" in r]
        real = [r for r in rs if r["line"] == "real"]
        if not real:
            continue
        real = real[0]
        ctrl = sorted(r["aligned_m_per_km"] for r in rs if r["line"] != "real")
        rv = real["aligned_m_per_km"]
        beat = sum(c < rv for c in ctrl) + 0.5 * sum(c == rv for c in ctrl)
        ranks.append((beat, len(ctrl)))
        ctrl_chains = [len(r["aligned"]) for r in rs if r["line"] != "real"]
        rows.append(f"{name}: real {rv} m/km in {len(real['aligned'])} chains | controls median "
                    f"{np.median(ctrl):.1f} m/km, {np.mean(ctrl_chains):.1f} chains | beats {beat:g} of {len(ctrl)}")
        for c in real["aligned"]:
            s0, e1 = c[0][0], max(p[1] for p in c)
            desc = f"{len(c)} pieces over {e1 - s0} m: " + ", ".join(f"{s}-{e} m @ {o:+.0f} m" for s, e, o in c)
            kml.append((name, s0, e1, c, desc))
    obs = np.mean([b / n for b, n in ranks])
    rng = np.random.default_rng(0)
    sims = np.array([np.mean([rng.integers(0, n + 1) / n for _, n in ranks]) for _ in range(20000)])
    rows.append(f"\nreal lines beat {obs:.0%} of their controls (chance = 50%); one-sided p = {np.mean(sims >= obs):.3f}")
    (OUT / "aligned_summary.txt").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))
    (OUT / "aligned_results.json").write_text(json.dumps(
        [{k: v for k, v in r.items() if k in ("corridor", "line", "km_with_data", "aligned", "aligned_m_per_km")}
         for r in results], indent=1))

    # chains on the real lines, for Google Earth: pieces solid, the chain's span thin
    sites = {}
    for f in ("sites_chacoan.json", "great_houses.json", "sites.json", "sites_local.json"):
        for s in json.loads((Path("output/corridors") / f).read_text()):
            sites.setdefault(s["name"], s)
    pm = []
    for name, s0, e1, c, desc in kml:
        a, b = name.split(" - ")
        p0, p1 = ct.endpoints(sites[a], sites[b])
        u = (p1 - p0) / np.linalg.norm(p1 - p0)
        v = np.array([u[1], -u[0]])
        ll = lambda s, o: ct.TO_LL.transform(*(p0 + u * s + v * o))
        segs = [(ll(s, o), ll(e, o)) for s, e, o in c]
        pm.append(f"<Placemark><name>{name} {s0}-{e1} m</name><description>{desc}</description><styleUrl>#span</styleUrl>"
                  f"<LineString><coordinates>{segs[0][0][0]},{segs[0][0][1]},0 {segs[-1][1][0]},{segs[-1][1][1]},0"
                  f"</coordinates></LineString></Placemark>")
        for (x0, y0), (x1, y1) in segs:
            pm.append(f"<Placemark><styleUrl>#piece</styleUrl><LineString><coordinates>{x0},{y0},0 {x1},{y1},0"
                      f"</coordinates></LineString></Placemark>")
    (OUT / "aligned_chains.kml").write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document><name>Aligned road-like pieces, great-house corridors</name>
<Style id="span"><LineStyle><color>9900ffff</color><width>1.5</width></LineStyle></Style>
<Style id="piece"><LineStyle><color>ff0000ff</color><width>5</width></LineStyle></Style>
{chr(10).join(pm)}
</Document></kml>
""")


if __name__ == "__main__":
    main()
