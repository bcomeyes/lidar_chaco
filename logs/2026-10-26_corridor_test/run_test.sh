#!/bin/bash
# corridor test; script itself skips lines after 20:25
cd /home/matt/Documents/lidar_chaco
if ps -eo args | grep -q run_images_nightl[y]; then echo "pull running, not starting"; exit 0; fi
echo "--- start $(date +%T)"
.venv/bin/python scripts/corridor_test.py output/corridors/great_house_corridors.json
echo "--- end $(date +%T)"
