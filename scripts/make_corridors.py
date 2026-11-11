"""Site-to-site corridors: KML for Google Earth + the 1 m tiles each corridor crosses.

Reads output/corridors/sites.json, or the SITES file given (public coordinates, with sources). Every pair of
sites within MAX_KM becomes a straight corridor line. Lines between two Chacoan-era
great houses are drawn solid red, others dashed-looking orange (by colour only).

Usage: python scripts/make_corridors.py [SITES.json OUT_STEM [MAX_KM [MIN_KM]]]
Outputs: output/corridors/<OUT_STEM or corridors>.kml/.json (pairs, km, bearing, tiles, MB)
"""
import json
import math
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path("output/corridors")
MAX_KM = 35
PROJECT = "CO_Southwest_NRCS_2018"        # one consistent survey where possible


def km_bearing(a, b):
    lat1, lon1, lat2, lon2 = map(math.radians, (a["lat"], a["lon"], b["lat"], b["lon"]))
    d = 2 * 6371 * math.asin(math.sqrt(math.sin((lat2 - lat1) / 2) ** 2 +
                                       math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2))
    y = math.sin(lon2 - lon1) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(lon2 - lon1)
    return d, (math.degrees(math.atan2(y, x)) + 360) % 360


def tiles_along(a, b, step_km=2.0):
    """1 m DEM tiles touched by the line (sampled every step_km, small bbox each)."""
    n = max(int(km_bearing(a, b)[0] / step_km), 1)
    found = {}
    for i in range(n + 1):
        lat = a["lat"] + (b["lat"] - a["lat"]) * i / n
        lon = a["lon"] + (b["lon"] - a["lon"]) * i / n
        bbox = f"{lon - 0.003},{lat - 0.003},{lon + 0.003},{lat + 0.003}"
        q = urllib.parse.urlencode({"bbox": bbox, "datasets": "Digital Elevation Model (DEM) 1 meter",
                                    "max": 20, "outputFormat": "JSON"})
        for attempt in range(6):
            try:
                with urllib.request.urlopen(f"https://tnmaccess.nationalmap.gov/api/v1/products?{q}", timeout=60) as r:
                    items = json.load(r)["items"]
                break
            except Exception as e:                      # the TNM API is flaky; back off and retry
                print(f"  TNM attempt {attempt + 1} failed: {e}", flush=True)
                time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError("TNM API kept failing")
        pick = [it for it in items if PROJECT in it["title"]] or items
        for it in pick:
            found[it["downloadURL"]] = round(it["sizeInBytes"] / 1e6)
    return found


def main():
    src, stem = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else (OUT / "sites.json", "corridors")
    max_km = float(sys.argv[3]) if len(sys.argv) > 3 else MAX_KM
    min_km = float(sys.argv[4]) if len(sys.argv) > 4 else 0
    sites = json.loads(Path(src).read_text())
    pairs = []
    for i, a in enumerate(sites):
        for b in sites[i + 1:]:
            d, brg = km_bearing(a, b)
            if min_km <= d <= max_km:
                chac = a["era"].startswith("chacoan") and b["era"].startswith("chacoan")
                pairs.append(dict(a=a["name"], b=b["name"], km=round(d, 1), bearing_deg=round(brg, 1),
                                  both_chacoan=chac, tiles=tiles_along(a, b)))
                print(f"{a['name']} - {b['name']}: {d:.1f} km, {len(pairs[-1]['tiles'])} tiles, "
                      f"{sum(pairs[-1]['tiles'].values())} MB", flush=True)
    (OUT / f"{stem}.json").write_text(json.dumps(pairs, indent=1))

    pm = []
    for s in sites:
        pm.append(f"""<Placemark><name>{s['name']}</name><description>era: {s['era']}; source: {s['source']}</description>
<styleUrl>#site</styleUrl><Point><coordinates>{s['lon']},{s['lat']},0</coordinates></Point></Placemark>""")
    by = {s["name"]: s for s in sites}
    for p in pairs:
        a, b = by[p["a"]], by[p["b"]]
        pm.append(f"""<Placemark><name>{p['a']} – {p['b']}</name><description>{p['km']} km, bearing {p['bearing_deg']}°,
{'both Chacoan-era' if p['both_chacoan'] else 'includes a later site'}</description>
<styleUrl>#{'chac' if p['both_chacoan'] else 'other'}</styleUrl><LineString><tessellate>1</tessellate>
<coordinates>{a['lon']},{a['lat']},0 {b['lon']},{b['lat']},0</coordinates></LineString></Placemark>""")
    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document><name>Site-to-site corridors</name>
<Style id="site"><IconStyle><scale>1.1</scale></IconStyle></Style>
<Style id="chac"><LineStyle><color>ff2020ff</color><width>3</width></LineStyle></Style>
<Style id="other"><LineStyle><color>ff00a5ff</color><width>2</width></LineStyle></Style>
{chr(10).join(pm)}
</Document></kml>
"""
    (OUT / f"{stem}.kml").write_text(kml)
    allt = {}
    for p in pairs:
        allt.update(p["tiles"])
    print(f"all corridors: {len(allt)} unique tiles, {sum(allt.values())} MB", flush=True)


if __name__ == "__main__":
    main()
