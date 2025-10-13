#!/bin/bash
# Environment Setup for Toxicology Hyperparameter Tuning
# CSC580 - CTA5

echo "================================================================"
echo "  Toxicology Neural Network - Environment Setup"
echo "  CSC580 - Critical Thinking Assignment 5"
echo "================================================================"
echo

# Check Python 3
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed"
    exit 1
fi

echo "[1/5] Creating virtual environment 'tox_env'..."
python3 -m venv tox_env

echo "[2/5] Activating virtual environment..."
source tox_env/bin/activate

echo "[3/5] Upgrading pip..."
pip install --upgrade pip

echo "[4/5] Installing dependencies..."
echo "  → Python version: $(python3 --version)"

# Install compatible versions for Python 3.13
pip install numpy
pip install pandas
pip install scikit-learn
pip install matplotlib

# Detect platform for TensorFlow
if [[ $(uname -m) == 'arm64' ]]; then
    echo "  → Detected Apple Silicon Mac"
    pip install tensorflow-macos
    pip install tensorflow-metal
else
    echo "  → Installing standard TensorFlow"
    pip install tensorflow
fi

pip install deepchem

echo
echo "[5/5] Verifying installation..."
python3 -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__} installed')"
python3 -c "import deepchem as dc; print(f'DeepChem {dc.__version__} installed')"

echo
echo "================================================================"
echo "  ✓ Setup Complete!"
echo "================================================================"
echo
echo "To activate environment:"
echo "  source tox_env/bin/activate"
echo
echo "To run the program:"
echo "  python3 toxicology_hyperparameter_tuning.py"
echo
echo "To deactivate:"
echo "  deactivate"
echo
