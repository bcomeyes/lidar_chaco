#!/bin/bash
cd /home/matt/Documents/lidar_chaco
PYTHONPATH=scripts timeout --signal=INT 3h .venv/bin/python scripts/learn_detector.py
