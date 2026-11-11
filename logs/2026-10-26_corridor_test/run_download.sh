#!/bin/bash
# corridor tile download; stops itself at 20:15
cd /home/matt/Documents/lidar_chaco
if ps -eo args | grep -q run_images_nightl[y]; then echo "pull running, not starting"; exit 0; fi
echo "--- start $(date +%T)"
exec .venv/bin/python scripts/download_tiles.py output/corridors/great_house_corridors.json
