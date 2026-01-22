#!/bin/bash
#
# Run all validation scripts using multiple virtual environments
#
# This script runs validation scripts in stages, using the appropriate
# environment for each framework:
#   - stardist_env: StarDist and DAPI-equivalent preprocessing (TensorFlow)
#   - instanseg_env: InstanSeg (PyTorch)
#   - cellpose_env: Cellpose and ConfluentFUCCI method (PyTorch)
#
# Usage:
#   ./run_validations_multi_env.sh              # Run all validations
#   ./run_validations_multi_env.sh --dry-run    # Show what would be done
#   ./run_validations_multi_env.sh --skip-snr   # Skip SNR computation
#
# Prerequisites:
#   - micromamba installed and available in PATH
#   - Three environments created: stardist_env, instanseg_env, cellpose_env
#
# To create the environments:
#   micromamba create -n stardist_env -f requirements_stardist.txt
#   micromamba create -n instanseg_env -f requirements_instanseg_cellpose_sam.txt
#   micromamba create -n cellpose_env -f requirements_instanseg_cellpose_sam.txt
#

set -e

# ============================================================
# Configuration - Edit these to match your environment names
# ============================================================
STARDIST_ENV="kerasEnv"
INSTANSEG_ENV="instanseg_env"
CELLPOSE_ENV="cellpose_env"

# Use micromamba or conda
CONDA_CMD="conda"

# ============================================================
# Script setup
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Results file for accumulating across environments
RESULTS_FILE="$SCRIPT_DIR/validation_results.json"

# Pass through arguments (like --dry-run, --skip-snr)
EXTRA_ARGS="$@"

echo "============================================================"
echo "Multi-Environment Validation Runner"
echo "============================================================"
echo ""
echo "Working directory: $SCRIPT_DIR"
echo "Results file: $RESULTS_FILE"
echo "Extra arguments: $EXTRA_ARGS"
echo ""

# Clean up previous results if starting fresh
if [[ ! "$EXTRA_ARGS" == *"--from-json"* ]]; then
    rm -f "$RESULTS_FILE"
fi

# ============================================================
# Stage 1: StarDist environment (TensorFlow)
# Runs: StarDist 1/2/3 channel, DAPI-equivalent preprocessing
# Also computes SNR (uses stardist/skimage)
# ============================================================
echo ""
echo "============================================================"
echo "Stage 1: StarDist environment ($STARDIST_ENV)"
echo "============================================================"
echo ""

# Activate environment and run
eval "$($CONDA_CMD shell.bash hook)"
$CONDA_CMD activate $STARDIST_ENV

if [ -f "$RESULTS_FILE" ]; then
    python update_latex_table.py --framework stardist --from-json "$RESULTS_FILE" --save-json "$RESULTS_FILE" $EXTRA_ARGS
else
    python update_latex_table.py --framework stardist --save-json "$RESULTS_FILE" $EXTRA_ARGS
fi

echo ""
echo "StarDist validation complete."

# ============================================================
# Stage 2: InstanSeg environment (PyTorch)
# Runs: InstanSeg 1/2/3 channel
# ============================================================
echo ""
echo "============================================================"
echo "Stage 2: InstanSeg environment ($INSTANSEG_ENV)"
echo "============================================================"
echo ""

# deactivate other environment and activate new
deactivate
# $CONDA_CMD activate $INSTANSEG_ENV
source ~/instanseg_venv_new/bin/activate

python update_latex_table.py --framework instanseg --skip-snr --from-json "$RESULTS_FILE" --save-json "$RESULTS_FILE" $EXTRA_ARGS

echo ""
echo "InstanSeg validation complete."

# ============================================================
# Stage 3: Cellpose environment (PyTorch)
# Runs: Cellpose and ConfluentFUCCI method
# ============================================================
echo ""
echo "============================================================"
echo "Stage 3: Cellpose environment ($CELLPOSE_ENV)"
echo "============================================================"
echo ""

deactivate
# $CONDA_CMD activate $CELLPOSE_ENV
source ~/cellpose_venv/bin/activate

# Run cellpose scripts
python update_latex_table.py --framework cellpose --skip-snr --from-json "$RESULTS_FILE" --save-json "$RESULTS_FILE" $EXTRA_ARGS

# Run confluentfucci scripts (uses same environment)
python update_latex_table.py --framework confluentfucci --skip-snr --from-json "$RESULTS_FILE" --save-json "$RESULTS_FILE" $EXTRA_ARGS

echo ""
echo "Cellpose validation complete."

# ============================================================
# Final: Update the LaTeX table
# ============================================================
echo ""
echo "============================================================"
echo "Updating LaTeX table with all results"
echo "============================================================"
echo ""

deactivate
# Use stardist env for final update (has all required packages for table update)
$CONDA_CMD activate $STARDIST_ENV

python update_latex_table.py --from-json "$RESULTS_FILE" --skip-snr --skip-accuracy

echo ""
echo "============================================================"
echo "Complete!"
echo "============================================================"
echo ""
echo "Results saved to: $RESULTS_FILE"
echo "LaTeX table updated: latex_tables/table_segmentation.tex"
echo ""
