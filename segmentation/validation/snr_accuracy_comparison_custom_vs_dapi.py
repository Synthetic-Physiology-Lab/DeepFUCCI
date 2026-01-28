"""
Compare SNR accuracy between custom-trained networks and DAPI-equivalent methods.

Compares:
- Custom-trained 2CH or 3CH network
- DAPI-equivalent with preprocessing
- DAPI-equivalent without preprocessing

On datasets: Validation, HT1080_20x, HT1080_40x
"""

import json
import matplotlib.pyplot as plt
import numpy as np

# Load results
with open("snr_accuracy_results.json") as f:
    custom_results = json.load(f)

with open("snr_accuracy_dapieq_results.json") as f:
    dapi_results = json.load(f)

# Datasets to compare
DATASETS = ["Validation", "HT1080_20x", "HT1080_40x"]
DATASET_LABELS = {
    "Validation": "Validation",
    "HT1080_20x": "HT1080 20x",
    "HT1080_40x": "HT1080 40x",
}

# Methods to compare
# Custom network: prefer 2CH since it uses same channels as DAPI-eq (cyan+magenta)
# If 3CH is desired, it can be changed here
CUSTOM_MODEL_KEY = "2 CH (cyan+magenta)"
CUSTOM_LABEL = "Custom 2CH"

# Create figure with 3 subplots (one per dataset)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Colors and markers for methods
method_styles = {
    "custom": {"color": "#2ecc71", "marker": "o", "label": CUSTOM_LABEL},
    "dapi_raw": {"color": "#e74c3c", "marker": "s", "label": "DAPI-eq (no preproc.)"},
    "dapi_postproc": {"color": "#3498db", "marker": "^", "label": "DAPI-eq (preproc.)"},
}

for ax_idx, dataset in enumerate(DATASETS):
    ax = axes[ax_idx]

    # Get data for custom model
    if dataset in custom_results and CUSTOM_MODEL_KEY in custom_results[dataset]:
        custom_data = custom_results[dataset][CUSTOM_MODEL_KEY]
        if custom_data["bin_centers"]:
            ax.plot(
                custom_data["bin_centers"],
                custom_data["recalls_per_bin"],
                color=method_styles["custom"]["color"],
                marker=method_styles["custom"]["marker"],
                label=f"{method_styles['custom']['label']} (R={custom_data['overall_recall']:.2f})",
                linewidth=2,
                markersize=6,
            )

    # Get data for DAPI raw (no preprocessing)
    if dataset in dapi_results["raw"]["datasets"]:
        dapi_raw = dapi_results["raw"]["datasets"][dataset]
        if dapi_raw["bin_centers"]:
            ax.plot(
                dapi_raw["bin_centers"],
                dapi_raw["recalls_per_bin"],
                color=method_styles["dapi_raw"]["color"],
                marker=method_styles["dapi_raw"]["marker"],
                label=f"{method_styles['dapi_raw']['label']} (R={dapi_raw['overall_recall']:.2f})",
                linewidth=2,
                markersize=6,
            )

    # Get data for DAPI postprocessed
    if dataset in dapi_results["postprocessed"]["datasets"]:
        dapi_postproc = dapi_results["postprocessed"]["datasets"][dataset]
        if dapi_postproc["bin_centers"]:
            ax.plot(
                dapi_postproc["bin_centers"],
                dapi_postproc["recalls_per_bin"],
                color=method_styles["dapi_postproc"]["color"],
                marker=method_styles["dapi_postproc"]["marker"],
                label=f"{method_styles['dapi_postproc']['label']} (R={dapi_postproc['overall_recall']:.2f})",
                linewidth=2,
                markersize=6,
            )

    ax.set_xlabel("Signal-to-Noise Ratio (SNR)")
    ax.set_ylabel("Detection Rate (Recall)")
    ax.set_title(DATASET_LABELS[dataset])
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig("snr_accuracy_custom_vs_dapi.pdf")
plt.savefig("snr_accuracy_custom_vs_dapi.png", dpi=150)
print("Saved: snr_accuracy_custom_vs_dapi.pdf/png")
plt.show()

# Print summary table
print("\n" + "=" * 80)
print("Summary: Custom 2CH vs DAPI-equivalent Methods")
print("=" * 80)
print(f"{'Dataset':<15} {'Custom 2CH':>15} {'DAPI (raw)':>15} {'DAPI (preproc)':>15}")
print("-" * 60)

for dataset in DATASETS:
    custom_recall = "N/A"
    dapi_raw_recall = "N/A"
    dapi_postproc_recall = "N/A"

    if dataset in custom_results and CUSTOM_MODEL_KEY in custom_results[dataset]:
        custom_recall = f"{custom_results[dataset][CUSTOM_MODEL_KEY]['overall_recall']:.3f}"

    if dataset in dapi_results["raw"]["datasets"]:
        dapi_raw_recall = f"{dapi_results['raw']['datasets'][dataset]['overall_recall']:.3f}"

    if dataset in dapi_results["postprocessed"]["datasets"]:
        dapi_postproc_recall = f"{dapi_results['postprocessed']['datasets'][dataset]['overall_recall']:.3f}"

    print(f"{DATASET_LABELS[dataset]:<15} {custom_recall:>15} {dapi_raw_recall:>15} {dapi_postproc_recall:>15}")

print("=" * 80)
