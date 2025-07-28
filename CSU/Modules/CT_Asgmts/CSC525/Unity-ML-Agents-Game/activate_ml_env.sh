#!/bin/bash
# Activate ML-Agents environment
source ml_agents_env/bin/activate

# Install compatible ML-Agents version
pip install --upgrade pip
pip install mlagents==0.28.0 torch==1.8.1 --extra-index-url https://download.pytorch.org/whl/cpu

echo "ML-Agents environment activated!"
echo "Run 'mlagents-learn config/collector_config.yaml --run-id=collector_v1' to start training"
