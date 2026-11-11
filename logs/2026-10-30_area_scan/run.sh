#!/bin/bash
cd /home/matt/Documents/lidar_chaco
PYTHONPATH=scripts timeout --signal=INT 4h .venv/bin/python scripts/area_scan.py
