#!/bin/bash
# Run SNR analysis scripts for network comparison

set -e  # Exit on error

cd "$(dirname "$0")"

# ============================================
# Virtual environment activation (fill in)
# ============================================
# Examples:
#   conda activate stardist_env
#   source /path/to/venv/bin/activate
#   micromamba activate stardist_env

# TODO: Uncomment and modify the line below
# conda activate _______________

# ============================================
# Run analysis
# ============================================

echo "=== Step 1: SNR Distribution Analysis ==="
python snr_histogram.py

echo ""
echo "=== Step 2: Accuracy vs SNR Analysis ==="
python snr_accuracy_analysis.py

echo ""
echo "=== Analysis complete ==="
echo "Outputs:"
echo "  - snr_histogram_all_datasets.pdf/png"
echo "  - snr_histogram_per_channel.pdf/png"
echo "  - snr_statistics.json"
echo "  - snr_accuracy_by_dataset.pdf/png"
echo "  - snr_accuracy_comparison.pdf/png"
echo "  - snr_accuracy_results.json"
