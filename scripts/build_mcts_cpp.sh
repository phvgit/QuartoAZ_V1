#!/bin/bash
# Build script for C++ MCTS module
# Run this on RunPod: bash scripts/build_mcts_cpp.sh

set -e

echo "Installing pybind11..."
pip install pybind11

echo "Building C++ MCTS module..."
cd alphaquarto/ai/mcts_cpp
python setup.py build_ext --inplace

echo "Copying module to parent directory..."
cp mcts_cpp*.so ../  2>/dev/null || cp mcts_cpp*.pyd ../  2>/dev/null || true

echo "Testing import..."
cd ../../..
python -c "from alphaquarto.ai.mcts_fast import MCTS_CPP_AVAILABLE; print(f'C++ MCTS available: {MCTS_CPP_AVAILABLE}')"

echo "Done!"
