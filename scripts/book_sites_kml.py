"""Combine digitised book-map dots into one KML + JSON for Google Earth.

Source: Lipe, "Social Power in the Central Mesa Verde Region", in Varien & Wilshusen
(eds.), Seeking the Center Place (2002), Figs. 10.1 (AD 1150-1225) and 10.2 (AD 1225-1290),
photographed pages; georeferenced by scripts/georef_book_map.py (town markers + Four Corners).
Positions are approximate (checked against known sites: typically 0.1-2 km).

Outputs: output/book/book_sites.kml, book_sites.json
"""
import json
from pathlib import Path

OUT = Path("output/book")
FIGS = {"fig10_1": "AD 1150-1225 (Lipe Fig. 10.1)", "fig10_2": "AD 1225-1290 (Lipe Fig. 10.2)"}
SIZES = [(26, "50-150 structures", 0.7), (42, "151-250 structures", 1.0), (1e9, ">250 structures", 1.4)]


def size_class(d):
    for lim, label, scale in SIZES:
        if d < lim:
            return label, scale


def main():
    allsites, folders = [], []
    for fig, period in FIGS.items():
        data = json.loads((OUT / f"{fig}_dots.json").read_text())
        pms = []
        for k, d in enumerate(sorted(data["dots"], key=lambda d: -d["lat"])):
            label, scale = size_class(d["diameter_px"])
            sid = f"{fig}-{k:02d}"
            allsites.append(dict(id=sid, period=period, size=label, lat=d["lat"], lon=d["lon"],
                                 approx_error_km=2))
            pms.append(f"""<Placemark><name>{sid}</name><description>{period}; {label}; position ±~2 km</description>
<Style><IconStyle><scale>{scale}</scale><color>{'ff0080ff' if fig == 'fig10_1' else 'ffff8000'}</color></IconStyle></Style>
<Point><coordinates>{d['lon']},{d['lat']},0</coordinates></Point></Placemark>""")
        folders.append(f"<Folder><name>{period}</name>{''.join(pms)}</Folder>")
    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document><name>Large sites, central Mesa Verde (Lipe 2002)</name>
{''.join(folders)}
</Document></kml>
"""
    (OUT / "book_sites.kml").write_text(kml)
    (OUT / "book_sites.json").write_text(json.dumps(allsites, indent=1))
    print(f"{len(allsites)} sites written")


if __name__ == "__main__":
    main()
