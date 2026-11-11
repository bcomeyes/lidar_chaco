#!/bin/bash
# Chacoan-era corridors: download new tiles, then detector + aligned score + mapped-road check.
# Each stage stops itself (download 20:15, detector skips lines after 20:25).
cd /home/matt/Documents/lidar_chaco
if ps -eo args | grep -q run_images_nightl[y]; then echo "pull running, not starting"; exit 0; fi
export CORRIDOR_OUT=output/corridor_test_chacoan CORRIDORS=output/corridors/chacoan_corridors.json PYTHONPATH=scripts
mkdir -p $CORRIDOR_OUT
echo "--- download $(date +%T)"
.venv/bin/python scripts/download_tiles.py $CORRIDORS || exit 1
echo "--- detector $(date +%T)"
.venv/bin/python scripts/corridor_test.py $CORRIDORS 2>&1 | grep -v "Deprecation\|alternative\|arr = s.read" || exit 1
echo "--- aligned $(date +%T)"
.venv/bin/python scripts/aligned_test.py || exit 1
echo "--- mapped roads $(date +%T)"
.venv/bin/python scripts/mapped_roads.py
echo "--- end $(date +%T)"
