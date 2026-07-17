<!-- AUTO-GENERATED from the header cell of pueblo_bonito_lidar_poc.ipynb - edit there, not here. -->

# Chaco Roads LiDAR — Pueblo Bonito / Pueblo Alto Proof of Concept

**Status: POC validated** (July 2026). This notebook processes the NCALM 2016 Chaco Canyon
0.5m bare-earth DTM and detects known Chacoan road segments — including the head of the
**Great North Road** at Pueblo Alto — using three complementary detection layers.

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
- **Great North Road corridor: continuous detection across the full frame** (fusion layer)
- Southern descent segments toward the Chetro Ketl stairways: detected
- East–west masonry wall through the Alto complex: detected (LRM, raised feature)
- Modern arc feature (bladed track/fence): detected and *rejected* by fusion —
  paired ridge/trench signature, suppressed by the sunken×shadowed product

## Roadmap
- Georeference Figure 11(b) segments onto our GeoTIFFs → rigorous match scoring
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
