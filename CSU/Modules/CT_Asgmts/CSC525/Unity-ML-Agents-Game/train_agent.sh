#!/bin/bash
echo "Starting ML-Agents training..."
echo "Make sure Unity project is open and playing before running this script!"
echo ""

# Train the agent
mlagents-learn config/collector_config.yaml --run-id=collector_v1 --force

echo ""
echo "Training completed! Check the 'results' folder for trained models."
echo "Copy the .onnx file to Assets/Models/ in your Unity project."
