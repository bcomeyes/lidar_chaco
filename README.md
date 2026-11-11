<!-- AUTO-GENERATED from the header cell of pueblo_bonito_lidar_poc.ipynb - edit there, not here. -->

# Chaco Roads LiDAR — Pueblo Bonito / Pueblo Alto Proof of Concept

**Status: POC runs end to end; road detection is weak so far** (corrected October 2026).
This notebook processes the NCALM 2016 Chaco Canyon 0.5m bare-earth DTM and tries to
detect the Chacoan roads around Pueblo Alto using three detection layers.

## What this does
1. Loads a 13 km² subset of the **NCALM 2016 Chaco DTM** (0.5m cell size) covering
   Pueblo Bonito, Chetro Ketl, and the Pueblo Alto mesa landscape.
2. Generates **Weiner-style hillshades** at five sun positions (the "manipulable light
   source" technique from Friedman, Sofaer & Weiner 2017).
3. Builds three detection layers:
   - **Multidirectional composite** — mean of four low-angle hillshades
   - **Local Relief Model (LRM)** — small-scale deviations from smoothed terrain
   - **Fusion** — pixelwise agreement of the two (crude ensemble; ML baseline)
4. Validates against Figure 11 of Friedman et al. 2017 (marked road segments,
   Pueblo Alto Landscape).

## Results so far
- Pueblo Alto & New Alto resolved to individual-room level
- East–west masonry wall through the Alto complex: detected (LRM, raised feature)
- **Correction (Oct 2026):** the bright continuous line we first called the Great North
  Road is a **modern track**, not the ancient road. It curves and runs ~28° east of north;
  the paper says the Great North Road leaves Pueblo Alto due north. Checked by laying the
  paper's Figure 11 road map exactly over our data (`scripts/fig11_ground_truth.py`).
- The roads the paper marks are present in our layers but **faint**: scored against the
  paper's map they are only slightly better than chance (`scripts/score_layers.py`).
- The modern arc (bladed track/fence) is dimmed by fusion but **not removed**.

## Roadmap
- ~~Georeference Figure 11(b) segments onto our GeoTIFFs → match scoring~~ done (Oct 2026)
- Tune the detection layers against the Figure 11 map so the faint roads stand out
- Straightness/orientation features → mathematically separate roads from drainages
- ML detector trained on Weiner's documented segments → Four Corners scale (1m 3DEP)
- Sibling project: reuse Sections 1–3 skeleton for NC coastal elevation screening

## Setup
```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Data: order the DTM subset from the
[NCALM 2016 Chaco dataset](https://portal.opentopography.org/datasetMetadata?otCollectionID=OT.042019.6342.1)
(details in Section 2) and place the GeoTIFF at `output/output_be.tif`.
