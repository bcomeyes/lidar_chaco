"""Download the 1 m DEM tiles a corridor file lists, into one cache folder.

Usage: python scripts/download_tiles.py output/corridors/great_house_corridors.json
Skips tiles already cached (here or in any output/3dep/ subfolder), resumes partial
files, fetches STREAMS files at once, stops before STOP_AT and whenever free disk would drop under FLOOR_GB.

Output: output/3dep/tiles/<tile>.tif
"""
import datetime as dt
import json
import shutil
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CACHE = Path("output/3dep/tiles")
STOP_AT = dt.time(20, 15)
FLOOR_GB = 60
STREAMS = 3


def main():
    pairs = json.loads(Path(sys.argv[1]).read_text())
    tiles = {}
    for p in pairs:
        tiles.update(p["tiles"])
    CACHE.mkdir(parents=True, exist_ok=True)
    have = {f.name for f in Path("output/3dep").rglob("*.tif")}
    todo = [(u, mb) for u, mb in sorted(tiles.items()) if u.split("/")[-1] not in have]
    print(f"{len(tiles)} tiles listed, {len(todo)} to fetch, {sum(m for _, m in todo)} MB", flush=True)
    def fetch(job):
        k, (url, mb) = job
        if dt.datetime.now().time() >= STOP_AT:
            print("stop time reached", flush=True)
            return
        free_gb = shutil.disk_usage(Path.home()).free / 1e9
        if free_gb - mb / 1e3 < FLOOR_GB:
            print(f"disk floor: {free_gb:.0f} GB free, skipping", flush=True)
            return
        name = url.split("/")[-1]
        part = CACHE / (name + ".part")
        for attempt in range(5):
            try:
                start = part.stat().st_size if part.exists() else 0
                req = urllib.request.Request(url, headers={"Range": f"bytes={start}-"} if start else {})
                t0 = time.time()
                with urllib.request.urlopen(req, timeout=120) as r, open(part, "ab" if start else "wb") as f:
                    if start and r.status != 206:
                        f.truncate(0)
                    shutil.copyfileobj(r, f, 1 << 22)
                part.rename(CACHE / name)
                print(f"[{k}/{len(todo)}] {name} {mb} MB in {time.time() - t0:.0f} s", flush=True)
                return
            except Exception as e:
                print(f"[{k}/{len(todo)}] {name} attempt {attempt + 1} failed: {e}", flush=True)
                time.sleep(10 * (attempt + 1))

    with ThreadPoolExecutor(STREAMS) as ex:
        list(ex.map(fetch, enumerate(todo, 1)))
    print("all done", flush=True)


if __name__ == "__main__":
    main()
