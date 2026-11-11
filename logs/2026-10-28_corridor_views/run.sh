#!/bin/bash
cd /home/matt/Documents/lidar_chaco
ps -eo args | grep -q run_images_nightl[y] && { echo "pull running, not starting"; exit 0; }
echo "--- start $(date +%T)"
PYTHONPATH=scripts timeout --signal=TERM 3h .venv/bin/python scripts/corridor_views.py
echo "--- end $(date +%T)"
