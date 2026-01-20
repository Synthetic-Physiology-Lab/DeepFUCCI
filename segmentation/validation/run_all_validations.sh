#!/bin/bash
#
# Run all validation scripts and update the LaTeX table
#
# Usage:
#   ./run_all_validations.sh              # Run all and update table
#   ./run_all_validations.sh --dry-run    # Show what would be done
#   ./run_all_validations.sh --save-json results.json  # Save results to JSON
#   ./run_all_validations.sh --env-config environments.json  # Use different Python envs
#   ./run_all_validations.sh --framework stardist  # Only run StarDist scripts
#
# Environment Configuration:
#   Different scripts require different virtual environments:
#   - stardist: TensorFlow-based (StarDist, DAPI-equivalent preprocessing)
#   - instanseg: PyTorch-based (InstanSeg)
#   - cellpose: PyTorch-based (Cellpose)
#   - confluentfucci: Specific Cellpose version
#
#   Create environments.json from environments.json.template and specify paths
#   to the Python interpreters for each framework.
#
#   Alternatively, run framework-by-framework:
#     ./run_all_validations.sh --framework stardist --save-json results_stardist.json
#     # activate instanseg environment
#     ./run_all_validations.sh --framework instanseg --from-json results_stardist.json --save-json results_combined.json
#     # etc.
#
# This script runs the Python update_latex_table.py which:
# 1. Executes all validation scripts in the appropriate directories
# 2. Parses the accuracy values at IoU=0.5 from the output
# 3. Updates latex_tables/table_segmentation.tex with the new values
#
# Table columns (from comment in table_segmentation.tex):
#   SNR     = Signal-to-Noise Ratio (computed from snr_utils.py)
#   \rom{1} = DAPIeq, raw, StarDist
#   \rom{2} = DAPIeq, post, StarDist
#   \rom{3} = DAPIeq, post, Cellpose
#   \rom{4} = DAPIeq, Cellpose Denoise (cyto3_denoise)
#   \rom{5} = ConfluentFUCCI
#   \rom{6} = InstanSeg 1Ch
#   \rom{7} = InstanSeg 2Ch
#   \rom{8} = InstanSeg 3Ch
#   1-CH    = This work (StarDist 1 channel)
#   2-CH    = This work (StarDist 2 channels)
#   3-CH    = This work (StarDist 3 channels)
#
# Dataset rows:
#   Validation    -> ValidationData/, DAPI_equivalent/, InstanSeg/, ConfluentFUCCI/
#   HT1080, 20x   -> (needs separate 20x scripts)
#   HT1080, 40x   -> HT1080_extra_dataset/
#   Han et al.    -> eDetectHaCaTFUCCI/
#   ConfluentFUCCI -> ConfluentFUCCI/
#   CellMAPTracer -> CellMAPtracer/validation/
#   Cotton et al. -> CottonEtAl2024/validation/
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "Running all validations and updating LaTeX table"
echo "============================================================"
echo ""
echo "Working directory: $SCRIPT_DIR"
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python not found in PATH"
    exit 1
fi

# Run the Python script with all arguments passed through
python update_latex_table.py "$@"

echo ""
echo "============================================================"
echo "Complete!"
echo "============================================================"
