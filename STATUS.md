# STATUS — Chaco roads LiDAR

_Last updated: 2026-11-11_

## Goal
Detect Chacoan roads and features in open-source LiDAR, then build ML models on it
(eventually Four Corners scale on 1 m 3DEP).

## Done (POC, July 2026)
- NCALM 2016 Chaco 0.5 m bare-earth DTM, 13 km² subset (Pueblo Bonito + Pueblo Alto
  mesa), in `output/output_be.tif` (EPSG:6342). Original delivery archive
  `output/rasters_NM16Dorshow.tar.gz` sits beside it.
- Notebook `pueblo_bonito_lidar_poc.ipynb`, 9 sections, all cells run, no errors:
  load → slope → 5-sun hillshade sweep → multidirectional composite → Local Relief
  Model → fusion (sunken × shadowed) → validation vs Friedman, Sofaer & Weiner 2017 Fig. 11.
- Result: Great North Road corridor detected continuously in the fusion layer; southern
  descent segments detected; modern bladed arc suppressed by fusion.
- Validation so far is **by eye** against the paper's figure.

## Step 1 done (2026-10-25): Fig. 11 ground truth + scoring
- `scripts/fig11_ground_truth.py` georeferences Fig. 11 to our hillshade (2.70 m/px,
  matches the scale bar; rotation 1.5°). Tile check: panel a 120 tiles RMS 0.88 m,
  panel b 64 tiles RMS 0.69 m, a-vs-b ≤ 0.47 m. Output `output/ground_truth/fig11_roads.tif`
  (0.5 m DTM grid; 0 outside, 1 figure area, 2 marked road; 17 segments ≥ 100 m²).
- `scripts/score_layers.py` → `scores.json`, `scores_segments.csv`. AUC (0.5 = chance):
  global composite 0.425, LRM 0.541, fusion 0.516; local (vs 15–40 m surroundings)
  LRM 0.576, fusion 0.574. Shift test peaks at 0 m in both axes → registration right,
  road signal real but weak.
- **Correction to the POC:** the bright continuous line called the "Great North Road"
  in Sections 8–9 and the README does not follow any marked segment. It runs ~28° E of N
  and curves; the paper says the road leaves Alto due north (within 0.25°). It matches
  the modern track visible in Fig. 11a. The marked due-north line (from Alto's NW corner)
  is barely visible in our layers. README/Section 9 claims need correcting (not yet edited).
- Overlays: `output/ground_truth/overlay_fusion_fig11*.jpg`.
- README + notebook Section 9 corrected (originals in `backups/`).

## Tuning + 1 m test (2026-10-25)
- `scripts/tune_layers.py`: LRM scales, slope damping, straight-line filter. Tuned on 9
  roads, checked on 8 held back. Best settings overfit (held-back 0.55 vs original 0.57),
  so the original settings stay. Filter settings are not the bottleneck.
- `scripts/road_profiles.py`: averaged cross-sections across the 8 straight marked roads.
  **Corrected 2026-10-29:** measured against the road's own shoulders (4–10 m out), 7 of 8
  dip only 1.5–4.8 cm (segment 21: 22 cm; segment 9 is a low hump, not a dip). The earlier
  "4–7 cm" line overstated it. (`depth_cm` in road_profiles.json uses a linear-trend
  baseline and is not a usable dip measure.) **At 1 m the dips are unchanged** (within 0.4 cm). Caveat: our 1 m is
  averaged 0.5 m data, kinder than a real, sparser 1 m survey.
- Conclusion: roads are real but very shallow (~5 cm) against bumpy ground; they show
  when many cross-sections are averaged along a straight line, not pixel by pixel.

## Running
Nothing. On a break from 2026-11-11; next step proposed, not approved: thin the 2024 QL1
Pueblo Alto point cloud to QL2 density, build a real 1 m DEM, score vs Fig. 11.

Previous: (Chacoan-era run ended cleanly 19:22.)

## Bigger site list from the book photos (2026-10-27)
- Varien Fig. 8.11 (IMG_0100) placed by matching its AD 1150-1225 dots to Lipe Fig. 10.1's
  (39/41 agree; homography median 480 m). Fig. 8.7 (IMG_0099, AD 1050-1150, Chacoan era)
  placed by matching its printed terrain to Fig. 8.11 with SIFT (local shift ~5 px ≈ 0.25 km).
  `scripts/georef_varien.py` → `output/book/varien_fig8_7_dots.json` (31 centres),
  `varien_fig8_11_dots.json` (58). 30 m regional DEM in `output/dem30/` (only used for checks).
- Check vs known great houses (Fig. 8.7): Goodman Point 0.6 km, Edge of the Cedars 0.9,
  Lowry 1.0, Yucca House 1.1, Bluff 1.4, Sand Canyon 1.5, Far View 1.8, Escalante 2.1.
- `scripts/build_site_list.py` merges Varien + Lipe + OSM into places (dots within 2 km =
  same community): `output/sites/places.json/.kml`. Names: OSM + Wikipedia/Wikidata
  lookups of the Lipe table names (`scripts/geocode_site_names.py`, slow, few hits —
  most Crow Canyon sites have no public location). Figs 8.8 (same period as 8.11) and
  8.12 (persistent communities) not used.

## Local corridors (2026-10-27) — did not confirm the great-house signal
- 157 pairs of OSM sites 1–15 km apart, 1024 km, 1727 lines (28 new tiles, 5.7 GB;
  `output/corridor_test_local/`, log `logs/2026-10-27_corridor_test/local.log`).
- Headline numbers looked strong (aligned 57%, p = 0.001; all dips 63%) but corridors
  overlap heavily: 84 of 157 are between Mesa Verde cliff dwellings a few km apart.
  Grouping by area (25 groups): aligned 55%, p = 0.18 (chance); all dips 61%, p = 0.04.
  Outside Mesa Verde: aligned 52% (chance), all dips 59%.
- Reading: the aligned-road signal does not hold up beyond the 9 great-house lines. The
  weak "all dips" excess, mostly in Mesa Verde, is more likely terrain (mesa tops vs
  canyon crossings) and park trails than roads. OSM visitor sites (cliff dwellings) are
  a poor test set: not Chacoan, not road-linked.

## Corridor test, great houses (2026-10-26) — no signal yet
- `scripts/corridor_test.py`: ±200 m band along each site-to-site line, detector run at the
  line's own bearing; score = straight road-like stretches (>=150 m) per km. Controls per
  corridor: same line shifted ±1/2/3 km, rotated ±10/20°. 9 corridors, 99 lines, 19 min.
- Result: real lines beat 51% of their controls (chance 50%, p = 0.48). Only
  Far View–Yucca House beats all 10 (201 vs 80 m/km), but its 17 stretches are short
  (150–340 m) at scattered offsets, not one aligned road; 1 of 9 at the top is expected
  by chance. Outputs `output/corridor_test/summary.txt`, `results.json`, `hits.kml`
  (166 stretches on real lines). Tiles: `output/3dep/tiles/` (44, 8.9 GB).
- Not yet tried: farmland mask (unplowed strips only), scoring stretches that line up at
  the same offset over km (a real road should), Chacoan-era sites from book maps.

## Aligned-pieces test (2026-10-27) — first positive signal, weak
- `scripts/aligned_test.py` on the rerun of `corridor_test.py` (now saves all pieces
  >= 100 m per line; v1 outputs in `backups/corridor_test_v1/`). Pieces chain if gap
  <= 2 km and offsets agree within 6 m + 3 m/km; chain = >= 2 pieces spanning >= 1 km.
- Real lines beat 70% of controls (chance 50%), p = 0.031. 4 corridors beat 10/10
  (Lowry–Far View, Lowry–Yucca, Bluff–Edge of the Cedars, Far View–Yucca); Far View–Escalante 0/10.
- Checks (all held): mapped OSM roads/tracks/rail removed (`scripts/mapped_roads.py`) →
  71%, p = 0.023; same-bearing (shifted) controls only → 72% (not the field grid);
  first/last 2 km trimmed → 70% (not the sites themselves); 5 km trimmed → 64%, p = 0.09.
- Caveats: 9 corridors, controls overlap between corridors, second test design tried
  (v1 was null). Chains are short: max 4 pieces, 15–25% of their span. No single
  chain is convincing on its own. Candidates: `output/corridor_test/aligned_chains.kml`.
- Logs: `logs/2026-10-27_corridor_test/`.

## Road detector (2026-10-26)
- `scripts/road_detector.py`, runs on 1 m (0.5 m averaged 2x2). Looks for a 5 m-wide
  dip vs 4–10 m shoulders, averaged 100 m along 36 bearings, local z-score.
- v1: median per-road AUC 0.51 — deep valleys/drainages swamped the 5 cm roads.
  Outputs kept in `backups/detector_v1/`.
- v2: each slice capped at 15 cm, mean dips > 25 cm rejected. **Median 0.60, 10/17 roads
  above 0.6, 3 above 0.7** (old best LRM: 0.57, 6/17). 16 of 17 roads improved.
  Caveat: v2 rules were chosen after viewing v1.
- Weakness: background is full of short false streaks (random bumps lined up over
  ~100 m). Real roads run longer and straighter → next lever.
- Outputs: `output/detector/` (road_z.tif, road_bearing.tif, road_dip_cm.tif, overlays).

## Road chaining (2026-10-26)
- `scripts/chain_roads.py`: joins aligned detector pieces (≤10° apart, ≤6 m sideways,
  gaps ≤40 m bridged) into lines. Roads are never judged on where they go; both ends of
  every line are saved as "ended here" points (user: some Chacoan roads just stop, e.g.
  the Great North Road at the Kutz Canyon badlands).
- Inside the Fig. 11 area: of detected lines ≥300 m, 81% lie on marked roads (≥200 m: 71%,
  ≥100 m: 45%). But only 8–22% of marked road length is found (precision good, recall low).
- Some "misses" may be unmarked real roads or modern roads (canyon floor).
- Outputs: `output/chains/` road_chains.geojson, road_ends.geojson (EPSG:6342), overlay.

## Road following (2026-10-26) — did not help
- `scripts/follow_roads.py`: extends lines ≥200 m step by step while a 30 m stretch ahead
  beats 90% of random ground. Result: road found 12% → 13%, precision 71% → 65%.
  Ends: 48 "faded", 10 "edge". Outputs in `output/follow/`.
- Why (window test, `logs/2026-10-26_follow_roads/window_test.log`): a stretch of marked
  road has a median dip of only 2.6–3.9 cm, about what ordinary ground shows. Even over
  150 m, only 28% of road stretches beat the 90% noise bar (52% beat the 75% bar).
- Meaning: from elevation alone, roughly half or more of each marked road is not
  separable from ordinary ground in this data. The paper's marks also draw on aerial
  photos and fieldwork. A realistic LiDAR-only map catches the clearest stretches with
  good precision, not complete roads.

## Real LiDAR available (USGS 3DEP index, checked 2026-10-26; nothing downloaded)
QL2 = ≥2 laser pulses/m², 1 m DEM. QL1 = ≥8 pulses/m². Our NCALM Chaco data: 25.6 pts/m².
| Area | Survey | Quality | Collected | DEM |
|---|---|---|---|---|
| Chaco (Pueblo Alto) | CO_CONMGaps_1_D24 | QL1 | 2024-09 | 0.5 m, pending; point cloud out |
| Montezuma Co (Cortez, Sand Canyon, Cross Canyon), Dove Creek, Durango, Hesperus, Pagosa Springs | CO_Southwest_NRCS_B1_2018 | QL2 | 2018-10/11 | 1 m |
| Rico (Dolores Co) | CO_Southwest_NRCS_B2_2018 | QL2 | 2018-11 → 2019-09 | 1 m |
| Ignacio, Chimney Rock (Archuleta Co) | CO_Southwest_NRCS_TL_2018 | QL2 | 2018-10/11 | 1 m |
| Blanding UT | UT_StatewideSouth_3_2020 | QL2 | 2020 | 1 m |
| Monticello UT | UT_Southern_B2_2018 | QL2 | 2018 | 1 m |
| Navajo Nation NM (Shiprock, Crownpoint) | NM_NorthWest_Navajo_A_TL_2018 | QL2 | 2018-11 | 1 m |
| Navajo Nation AZ (Chinle, Kayenta) | AZ_NavajoNation_1_D22 | QL2 | 2022 | 1 m |
| Aztec Airport Mesa Road (paper's case study) | NM_Animas_2014 | QL2 | 2014-10 | 1 m |
1 m DEM tiles are 10×10 km, ~0.3 GB each. Point-cloud tiles ~1 km², 35–60 MB.

## First runs on real QL2 1 m data (2026-10-26)
Tiles in `output/3dep/` (~1.3 GB): aztec (NM_Animas_2014), lowry, dolores
(CO_Southwest_NRCS_2018), hovenweep_merged (CO 2018 + two UT 2020 surveys filling the
state-line gap; surveys agree within 5–10 cm). `scripts/run_area.py NAME TILE LON LAT`
runs detector v2 + chaining on a 4×4 km window → `output/areas/NAME/` (overlay.jpg).
| Area | lines ≥150 m | ≥300 m | What dominates |
|---|---|---|---|
| Aztec airport | 43 | 6 | runways, roads, town grid, pipelines |
| Lowry Pueblo | 127 | 35 | farmland: furrows, fences, section roads |
| Dolores (S of Dove Creek) | 100 | 17 | farmland, county roads, downslope rills |
| Hovenweep | 20 | 0 | quiet — undisturbed canyon country |
- Aztec Airport Mesa Road (paper Fig. 5) could not be confirmed: ~100 m long, 20 m wide
  between low banks (not the 5 m dip the detector targets), in heavily graded ground.
- No answer key in Lowry/Dolores/Hovenweep, so these are not scored.
- Lessons: modern straight lines (farm, roads, survey grid) are the main false alarm on
  real data; undisturbed ground is quiet. Many short lines run downslope (erosion rills).

## Site locations (2026-10-26)
- Public sites: `output/corridors/sites_osm.json` (52 from OpenStreetMap, 10 great houses),
  `corridors.kml` (first 7-site version). Great-house corridors approved 2026-10-26 17:00:
  9 pairs <= 55 km (`great_house_corridors.json/.kml`; 4 only if <= 35 km). Local corridors
  (157, ~7 GB) not started.
- Book maps (user's photos, Google Drive IMG_0098–0107): Varien & Wilshusen eds., *Seeking
  the Center Place* (2002). Lipe Figs. 10.1/10.2 georeferenced from town markers
  (`scripts/georef_book_map.py`; leave-one-out 0.2–1.7 km) → 91 site dots with size class,
  `output/book/book_sites.kml`. Checked vs known sites: 0.1–2.3 km. Dots are unnamed;
  Tables 10.1/10.2 list ~75 names. Varien Figs. 8.7–8.12 (incl. AD 1050–1150) are relief
  maps without towns — not yet georeferenced. Pink highlighter on the pages is not meaningful.

## Long-term goal (user's "true north")
Chaco at 0.5 m → prove at 1 m → Cortez / Cross Canyon / Sand Canyon (Montezuma County)
→ full-area 1 m map.

## Next (README roadmap, in order)
1. ~~Georeference Fig. 11(b) → match scoring~~ done; see above.
2. Straightness/orientation features to separate roads from drainages.
3. ML detector trained on documented segments → Four Corners scale (1 m 3DEP).
4. Sibling project: reuse Sections 1–3 for NC coastal elevation screening.

## Broken / unclear
- Section 9 says the straight-vs-meandering discriminator "separates road candidates
  from drainages in every layer", but no such code exists — it is a visual judgement.
  Roadmap item 2 is what would make it real.
- Paper now in `refs/` (open-access copy from solsticeproject.org, DOI 10.1017/aap.2017.25);
  Fig. 11 extracted as `refs/fig11_pueblo_alto_landscape.jpg` (1334×695, both panels).
  Fig. 11 has a scale bar (400 m) and north arrow but no coordinates, so it must be
  georeferenced by matching landmarks to our hillshade.
- `output/`, `backups/` and `refs/` are gitignored (large rasters; paper copy not redistributed
  in the public repo). `output/` holds the only local copy of the DTM (146 MB) and the
  original archive (142 MB).
- Ground truth is a raster corridor (~10–13 m wide, as drawn), not vector centrelines.

## Constraints
Never run between 21:00 and 07:00 (pecha nightly pull); every job ends by 20:30; keep
≥ 60 GB free (119 GB free on 2026-10-25). 4 cores, 15 GB RAM, no GPU.

## Chacoan-era corridors result (2026-10-27, `output/corridor_test_chacoan/`)
Verdict: weak, not confirmed. 119 corridors between the 31 AD 1050-1150 centres.
- Aligned-pieces score: real lines beat 54% of controls (p = 0.069); mapped roads removed 55% (p = 0.042);
  grouped by area (109 groups) 55%, Wilcoxon p = 0.052. All-dips score: 52% (p = 0.24).
- By position quality (aligned): both ends exact 58% (19 corridors, p = 0.14); one end exact 55%;
  both ends book dots 52% (chance). Trend in the right direction, too few exact corridors to say.
- 15 corridors beat all 10 controls with mapped roads removed (list in chat 2026-10-27; e.g. Hawkins-Yucca
  443 m/km vs 84 controls). Candidates for eyeballing in `aligned_chains.kml`.
- Analysis script (scratch, not a project tool): grouping/split done ad hoc.

## Corridor views for the eye (2026-10-28, `output/corridor_views/index.html`)
- `scripts/corridor_views.py`: for the 15 corridors that beat 10/10 controls, 1 km x ±200 m windows along
  each aligned chain: side-lit relief (sun 8 deg, at right angles to the line, both sides), openness (20 m),
  local relief (±10 cm, 5 m), mapped OSM roads in yellow, detector pieces bracketed in red. 317 images.
- First look by Claude (273 pieces cropped into sheets): no convincing road. Hawkins-Yucca House (top score,
  443 m/km) runs along the Cortez airport runway (KCEZ 15 m off the line); its pieces are the runway's
  ditches/edges (OSM tags runways as aeroway, so the mapped-road check missed them). Most other pieces sit on
  slope breaks, drainages and terrace/canal edges, with no distinct linear feature between the brackets.
- So the p ~ 0.05 lean from the Chacoan run is probably not roads. Detector needs a stricter filter (flat
  ground, mask aeroways/canals/water) before corridor stats mean much.
- User's eye check (2026-10-29, ~80% of images): same verdict. Pieces are natural features or modern roads;
  nothing convincing. Corridor search closed for now.
- Next agreed (not started): (1) KML of all book pages + PBS map (IMG_0172) as overlays with tie points
  (cities, state lines, Four Corners); (2) Lowry roads from PBS drone frames IMG_0175-0180 in
  output/book/chacopictures, oriented by Abajo Peak = west-northwest, as an answer key.

## Map overlays KML (2026-10-29, `output/overlays/maps.kml`, `scripts/book_overlays.py`)
- 7 overlays (switchable, semi-transparent): Varien Figs 8.7, 8.8 (IMG_0098), 8.11, 8.12 (IMG_0101); Lipe 10.1, 10.2;
  PBS documentary map (IMG_0172). Plus dots read from each, and tie points.
- 8.8 / 8.12 placed by dot pattern (35 / 46 pairs, median 300 / 360 m); terrain (SIFT) matching failed on them.
- PBS map: 5 town squares + Four Corners, fit 0.1-0.7 km (1 px = 0.4 km); 161 dots read; covers south of ~37.4 N only
  (Lowry, Edge of the Cedars are above its frame). Overlay pairs checked by blending: rivers, Ute Mtn, towns line up.

## Lowry roads from PBS drone frames (2026-10-29, `output/lowry_roads/`, `scripts/lowry_drone_roads.py`)
- Camera fitted to the true skyline (30 m DEM). The mountain is **Sleeping Ute** (159 deg, 36 km), not Abajo:
  skyline error 0.4-0.7 px facing SE vs 2-2.3 px for the best westward fit. Frames face 135-159 deg; footage is at sunrise.
- Road A (long straight road) bearing 160.3 / 162.6 deg from two frames -> 161.4 deg, pointing at Sleeping Ute.
- Detector along Road A (0-3 km): 35% of length has a z>=2 hit within 10 m of the line vs 8% at other offsets;
  9 control bearings (+-5..20 deg, and 71/251/341) 2-13% with no excess. 1012 m of pieces on the line (0.1-1.7 km)
  vs <= 240 m for any control. First positive answer-key result.
- By eye (`road_A_view.png`): km 0-1 a shallow swale leaves the pueblo along the line, partly mixed with a natural
  drainage; km 1-2 crosses a plowed field with several straight marks (an X of lines; age unknown); km 2-3 nothing.
- Other roads (B, C, D) traced but less consistent between frames; rough only.

## Known Chaco roads in the corridor views (2026-10-29, `output/known_road_views/`)
- `scripts/known_road_views.py`: the 17 Fig. 11 segments drawn in the same panels as the corridor
  views (1 m pixels, ±200 m), plus a fifth panel with the published roads painted red. 19 images.
  v1 (red ticks) in `backups/known_road_views_v1/`: ticks were wrong for 9 bent/branched segments.
- First read (v1) was wrong: I said the roads were nearly invisible. The user saw many visible
  roads, and the red-outline panel confirms it: several published roads are sharp straight lines
  (seg 6's diagonal, seg 10's lines to the ring feature), some look raised rather than sunken.
  Others (e.g. seg 3, Great North Road leaving Alto; seg 6's horizontal mark) show little.
- Meaning: where roads survive, the views and the eye do show them. Detector targets 5 m dips
  only; raised/bermed roads may be missed.

## Detector v3, learned from Chaco roads; Lowry retried (2026-10-29, `output/detector_v3/`)
- `scripts/learn_detector.py`: learns the 100 m-averaged cross-section (±15 m, values + squares so
  dips and ridges both count) by logistic regression. 4945 road stretches on 17 segments; 19775
  non-road (random ground, road turned 60-90 deg, road shifted 20-40 m). 5 folds split by segment.
- Held-out AUC 0.615 vs v2 0.566 (road found at 5% false alarms 7% vs 6%): better, still weak.
  Average road profile: flat centre with ~2-4 cm higher edges at 6-9 m (bermed bed?); 41% of
  stretches have a raised centre.
- Lowry Road A (161.4 deg, 0-3 km): on-line 25% vs 7% background; per km 56% / 18% / 0%.
  9 controls 7-19% (156.4 deg, 5 deg off and overlapping near the pueblo, 19%). Road A beats 9/9.
  Pieces: 96-247 m, 535-680 m, 1071-1183 m. v3 no longer picks up the X marks in the field (km 1-2).
- Log `logs/2026-10-29_detector_v3/run.log`; view `output/detector_v3/road_A_view_v3.png`.

## Lowry area scan, blind, unplowed only (2026-10-30, `output/area_scan_lowry/`) — did not bear fruit
- `scripts/area_scan.py`: detector v3 over 7 x 7 km around Lowry, 36 bearings, NLCD 2021 cropland
  (buffered 30 m), developed land and OSM roads masked (62% scanned). Hit = top 1% of scores;
  segment = held >= 150 m along one bearing. 539 segments (19 >= 300 m), 110 km total.
- Documentary roads found blind: none convincingly. A ray 0-3 km: 0 m (the earlier 25% on-line
  signal was 100-150 m pieces under a looser bar). C right 175 m (beats 11/11 turned copies),
  B oval 87 m (10/11), A trace 195 m (9/11); B left/right, C left, D: 0 m. Traces are rough.
- Radial: 5 segments point at Lowry vs median 6 for random points: Lowry is not a hub.
- By eye: most segments sit on canyon rims and drainages. NLCD "shrub" includes old fields.
- Meaning: at 1 m QL2 the road signal is too weak to find roads without being told where to look.
- Log `logs/2026-10-30_area_scan/run.log`. NLCD clip `output/nlcd/lowry_nlcd2021.tif` (300 KB).
