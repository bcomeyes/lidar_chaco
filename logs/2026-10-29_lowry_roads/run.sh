#!/bin/bash
cd /home/matt/Documents/lidar_chaco
ps -eo args | grep -q run_images_nightl[y] && { echo "pull running, not starting"; exit 0; }
echo "--- start $(date +%T)"
PYTHONPATH=scripts timeout 2h .venv/bin/python scripts/lowry_drone_roads.py 2>&1 | grep -v -i "warn\|reshape\|arrs =\|self.a ="
echo "--- end $(date +%T)"
