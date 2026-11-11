"""One site list from every source: book maps (Varien, Lipe), OpenStreetMap, named lookups.

Entries (one per dot per figure) are grouped into PLACES: dots from different figures or
periods within MERGE_KM of each other are taken to be the same community. Each place
gets every period it appears in, a best position (mean of its dots) and a name when a
public named location (OSM or Wikipedia/Wikidata, title must contain the site's own name)
lies within NAME_KM.

Outputs: output/sites/places.json, places.kml (by period, Chacoan era highlighted),
         entries.json
"""
import json
import math
import re
from pathlib import Path

import numpy as np

BOOK = Path("output/book")
OUT = Path("output/sites")
MERGE_KM = 2.0
NAME_KM = 2.5
# OSM sites are labelled by one of their features; show the site's usual name
DISPLAY = {"Great Kiva": "Yucca House", "Lowry Ruin Great Kiva": "Lowry Pueblo",
           "Edge of the Cedars Great Kiva": "Edge of the Cedars", "Aztec West Great Kiva": "Aztec Ruins",
           "Chimney Rock Guard House": "Chimney Rock", "Bluff Great House (Anasazi)": "Bluff Great House",
           "pithouses": "Square Tower House (Mesa Verde)", "Sun Temple": "Sun Temple / Cliff Palace",
           "Kiva": "Kiva (Sand Canyon area)"}
GENERIC = {"ruin", "ruins", "pueblo", "site", "cluster", "house", "complex", "village", "canyon", "mesa",
           "great", "kiva", "no", "the", "of", "and", "locality", "main", "middle", "upper", "lower"}


def km(a, b):
    return math.hypot((a[0] - b[0]) * 111.0, (a[1] - b[1]) * 111.0 * math.cos(math.radians(37.4)))


def tokens(s):
    return {t for t in re.findall(r"[a-z]+", s.lower()) if t not in GENERIC and len(t) > 2}


def entries():
    out = []
    for f, src in (("varien_fig8_7_dots.json", "Varien Fig. 8.7"), ("varien_fig8_11_dots.json", "Varien Fig. 8.11")):
        for d in json.loads((BOOK / f).read_text()):
            out.append(dict(lat=d["lat"], lon=d["lon"], period=d["period"], source=src, err_km=1.5))
    for f, per, src in (("fig10_1_dots.json", "AD 1150-1225", "Lipe Fig. 10.1"),
                        ("fig10_2_dots.json", "AD 1225-1290", "Lipe Fig. 10.2")):
        for d in json.loads((BOOK / f).read_text())["dots"]:
            out.append(dict(lat=d["lat"], lon=d["lon"], period=per, source=src, err_km=1.5))
    return out


def named():
    out = []
    for s in json.loads(Path("output/corridors/sites_local.json").read_text()):
        out.append(dict(name=DISPLAY.get(s["name"], s["name"]), lat=s["lat"], lon=s["lon"], source="OpenStreetMap",
                        great_house=s["era"].startswith("chacoan")))
    geo = BOOK / "names/geocoded.json"
    if geo.exists():
        for n, h in json.loads(geo.read_text()).items():
            if h and tokens(n) & tokens(h["page"]):          # page title must carry the site's own name
                out.append(dict(name=n, lat=h["lat"], lon=h["lon"], source=f"{h['source']}: {h['page']}",
                                great_house=False))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    E = entries()
    # single-linkage grouping of dots into places
    n = len(E)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            if km((E[i]["lat"], E[i]["lon"]), (E[j]["lat"], E[j]["lon"])) <= MERGE_KM:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(E[i])
    N = named()
    places = []
    for g in groups.values():
        lat, lon = float(np.mean([e["lat"] for e in g])), float(np.mean([e["lon"] for e in g]))
        periods = sorted({e["period"] for e in g})
        near = sorted(((km((lat, lon), (s["lat"], s["lon"])), s) for s in N), key=lambda t: t[0])
        near = [(d, s) for d, s in near if d <= NAME_KM]
        places.append(dict(lat=round(lat, 5), lon=round(lon, 5), periods=periods,
                           chacoan_era=any("1050-1150" in p for p in periods),
                           sources=sorted({e["source"] for e in g}), n_dots=len(g),
                           spread_km=round(max(km((e["lat"], e["lon"]), (lat, lon)) for e in g), 2),
                           name=near[0][1]["name"] if near else None,
                           name_source=near[0][1]["source"] if near else None,
                           name_km=round(near[0][0], 2) if near else None,
                           name_lat=near[0][1]["lat"] if near else None,
                           name_lon=near[0][1]["lon"] if near else None,
                           other_names=[s["name"] for _, s in near[1:4]]))
    # named public sites with no book dot nearby are places too (OSM great houses, visitor sites)
    for s in N:
        if s["source"] == "OpenStreetMap" and all(km((s["lat"], s["lon"]), (p["lat"], p["lon"])) > NAME_KM for p in places):
            places.append(dict(lat=s["lat"], lon=s["lon"], periods=[], chacoan_era=s["great_house"],
                               sources=["OpenStreetMap"], n_dots=0, spread_km=0, name=s["name"],
                               name_source="OpenStreetMap", name_km=0.0, name_lat=s["lat"], name_lon=s["lon"],
                               other_names=[]))
    places.sort(key=lambda p: (-p["lat"], p["lon"]))
    for k, p in enumerate(places):
        p["id"] = f"P{k:03d}"
    (OUT / "entries.json").write_text(json.dumps(E, indent=1))
    (OUT / "places.json").write_text(json.dumps(places, indent=1))

    def pm(p, colour):
        label = p["name"] or p["id"]
        desc = (f"{', '.join(p['periods']) or 'period not in book maps'}; from {', '.join(p['sources'])}; "
                f"position ±~1.5 km" + (f"; name from {p['name_source']} ({p['name_km']} km away)" if p["name"] else ""))
        return (f"<Placemark><name>{label}</name><description>{desc}</description><Style><IconStyle>"
                f"<color>{colour}</color><scale>{1.3 if p['chacoan_era'] else 0.9}</scale></IconStyle></Style>"
                f"<Point><coordinates>{p['lon']},{p['lat']},0</coordinates></Point></Placemark>")
    chac = [pm(p, "ff0000ff") for p in places if p["chacoan_era"]]
    other = [pm(p, "ff00a5ff") for p in places if not p["chacoan_era"]]
    (OUT / "places.kml").write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document><name>Community centres and large sites, central Mesa Verde</name>
<Folder><name>Chacoan era (AD 1050-1150) and great houses</name>{''.join(chac)}</Folder>
<Folder><name>Later periods and other public sites</name>{''.join(other)}</Folder>
</Document></kml>
""")
    print(f"{len(E)} dots -> {len(places)} places; Chacoan era {len(chac)}; named {sum(p['name'] is not None for p in places)}")
    print("by number of periods:", {k: sum(len(p['periods']) == k for p in places) for k in range(0, 5)})


if __name__ == "__main__":
    main()
