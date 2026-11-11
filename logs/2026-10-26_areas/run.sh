cd /home/matt/Documents/lidar_chaco
P=".venv/bin/python -W ignore::DeprecationWarning -W ignore::RuntimeWarning scripts/run_area.py"
timeout 25m $P aztec   output/3dep/aztec/USGS_one_meter_x76y409_NM_Animas_2014.tif            -108.0287 36.8367
timeout 25m $P lowry   output/3dep/lowry/USGS_1M_12_x68y417_CO_Southwest_NRCS_2018_D18.tif    -108.9186 37.5852
timeout 25m $P dolores output/3dep/dolores/USGS_1M_12_x68y418_CO_Southwest_NRCS_2018_D18.tif  -108.90   37.70
echo ALL-AREAS-DONE
