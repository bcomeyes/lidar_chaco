"""Look up public coordinates for named sites (Lipe Tables 10.1/10.2 + known Chacoan great houses).

Sources, in order: Wikipedia (MediaWiki API: exact title, then full-text search; the page
must carry coordinates inside the region box), then Wikidata (label search + P625).
A hit is kept only if it falls inside BOX (central Mesa Verde region + Aztec/Chimney Rock).
Many Crow Canyon site names have no public location; those stay unmatched.

Input : output/book/names/table_names.json
Output: output/book/names/geocoded.json  (name -> lat, lon, source, page)
"""
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path("output/book/names")
BOX = (36.6, -110.2, 38.1, -107.0)        # lat0, lon0, lat1, lon1
UA = {"User-Agent": "lidar_chaco research script (archaeology site lookup)"}


def get(url, params):
    for attempt in range(4):
        try:
            req = urllib.request.Request(url + "?" + urllib.parse.urlencode(params), headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception as e:
            print("  retry:", e, flush=True)
            time.sleep(5 * (attempt + 1))
    return {}


def inside(lat, lon):
    return BOX[0] < lat < BOX[2] and BOX[1] < lon < BOX[3]


def wikipedia(name):
    api = "https://en.wikipedia.org/w/api.php"
    titles = [name] + [t["title"] for t in get(api, dict(action="query", list="search", srsearch=f'"{name}"',
                                                             srlimit=3, format="json")).get("query", {}).get("search", [])]
    for t in titles:
        q = get(api, dict(action="query", titles=t, prop="coordinates", redirects=1, format="json"))
        for p in q.get("query", {}).get("pages", {}).values():
            for c in p.get("coordinates", []):
                if inside(c["lat"], c["lon"]):
                    return dict(lat=c["lat"], lon=c["lon"], source="Wikipedia", page=p["title"])
    return None


def wikidata(name):
    api = "https://www.wikidata.org/w/api.php"
    hits = get(api, dict(action="wbsearchentities", search=name, language="en", limit=5, format="json")).get("search", [])
    for h in hits:
        e = get(api, dict(action="wbgetentities", ids=h["id"], props="claims|labels", format="json"))
        ent = e.get("entities", {}).get(h["id"], {})
        for cl in ent.get("claims", {}).get("P625", []):
            v = cl["mainsnak"].get("datavalue", {}).get("value")
            if v and inside(v["latitude"], v["longitude"]):
                return dict(lat=v["latitude"], lon=v["longitude"], source="Wikidata", page=h["id"] + " " + h.get("label", ""))
    return None


def main():
    lists = json.loads((OUT / "table_names.json").read_text())
    names = list(dict.fromkeys(n for v in lists.values() for n in v))
    cache = OUT / "geocoded.json"
    done = json.loads(cache.read_text()) if cache.exists() else {}
    for n in names:
        if n in done:
            continue
        hit = None
        for q in [n] + ([n.split("/")[0]] if "/" in n else []) + ([n.replace(" Cluster", "")] if "Cluster" in n else []):
            hit = wikipedia(q) or wikidata(q)
            if hit:
                break
        done[n] = hit
        print(f"{n}: {hit}", flush=True)
        cache.write_text(json.dumps(done, indent=1))
        time.sleep(0.5)
    print(f"{sum(v is not None for v in done.values())} of {len(done)} names located")


if __name__ == "__main__":
    main()
