#!/bin/bash
echo "Starting live prediction service at $(date)" >> /data/model-data/logs/scout_ml.log

source ~/.bashrc
cd /data/pandaml/src/
source /data/venv/bin/activate

# Log the execution of the Python command
echo "Executing Python command at $(date)" >>  /data/model-data/logs/scout_ml.log
python -m scout_ml_package.live_prediction
