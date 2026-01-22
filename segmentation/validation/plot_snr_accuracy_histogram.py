"""
Plot SNR vs Detection Rate histogram for 1CH, 2CH, 3CH models.

Usage:
    python plot_snr_accuracy_histogram.py
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load results
with open("snr_accuracy_results.json") as f:
    results = json.load(f)

# Focus on Validation dataset
dataset = "Validation"
data = results[dataset]

# Extract data for each model
models = ["1 CH (tubulin)", "2 CH (cyan+magenta)", "3 CH (all)"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # Blue, Orange, Green
labels = ["1-CH", "2-CH", "3-CH"]

snr_centers = data[models[0]]["bin_centers"]
n_bins = len(snr_centers)

# Create figure
fig, ax = plt.subplots(figsize=(12, 6))

# Bar width and positions
bar_width = 0.25
x = np.arange(n_bins)

# Plot bars for each model
for i, (model, color, label) in enumerate(zip(models, colors, labels)):
    recalls = data[model]["recalls_per_bin"]
    offset = (i - 1) * bar_width
    bars = ax.bar(x + offset, recalls, bar_width, label=label, color=color, alpha=0.8)

# Customize plot
ax.set_xlabel("SNR Bin Center", fontsize=12)
ax.set_ylabel("Detection Rate (Recall)", fontsize=12)
ax.set_title(f"Detection Rate vs SNR - {dataset} Dataset (IoU=0.5)", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels([f"{s:.1f}" for s in snr_centers], rotation=45, ha="right")
ax.set_ylim(0, 1.05)
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)

# Add overall recall annotation
for i, (model, label) in enumerate(zip(models, labels)):
    overall = data[model]["overall_recall"]
    ax.text(
        0.98,
        0.95 - i * 0.05,
        f"{label}: {overall:.1%}",
        transform=ax.transAxes,
        ha="right",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

plt.tight_layout()
plt.savefig("snr_accuracy_histogram_validation.pdf", dpi=300, bbox_inches="tight")
plt.savefig("snr_accuracy_histogram_validation.png", dpi=300, bbox_inches="tight")
print("Saved: snr_accuracy_histogram_validation.pdf/png")

# Also create a line plot version
fig2, ax2 = plt.subplots(figsize=(10, 6))

for model, color, label in zip(models, colors, labels):
    recalls = data[model]["recalls_per_bin"]
    overall = data[model]["overall_recall"]
    ax2.plot(
        snr_centers,
        recalls,
        marker="o",
        color=color,
        linewidth=2,
        markersize=8,
        label=f"{label} (overall: {overall:.1%})",
    )

ax2.set_xlabel("Signal-to-Noise Ratio (SNR)", fontsize=12)
ax2.set_ylabel("Detection Rate (Recall)", fontsize=12)
ax2.set_title(f"Detection Rate vs SNR - {dataset} Dataset (IoU=0.5)", fontsize=14)
ax2.set_ylim(0, 1.05)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("snr_accuracy_lineplot_validation.pdf", dpi=300, bbox_inches="tight")
plt.savefig("snr_accuracy_lineplot_validation.png", dpi=300, bbox_inches="tight")
print("Saved: snr_accuracy_lineplot_validation.pdf/png")

# Export to CSV for external plotting
df = pd.DataFrame(
    {
        "SNR_bin_center": snr_centers,
        "detection_rate_1CH": data[models[0]]["recalls_per_bin"],
        "detection_rate_2CH": data[models[1]]["recalls_per_bin"],
        "detection_rate_3CH": data[models[2]]["recalls_per_bin"],
        "counts_per_bin": data[models[0]]["counts_per_bin"],
    }
)
df.to_csv("snr_accuracy_validation_detailed.csv", index=False)
print("Saved: snr_accuracy_validation_detailed.csv")

# Print summary
print("\n" + "=" * 60)
print("Summary - Validation Dataset")
print("=" * 60)
print(f"{'Model':<25} {'Overall Recall':>15} {'Mean SNR':>10} {'Std SNR':>10}")
print("-" * 60)
for model, label in zip(models, labels):
    print(
        f"{label:<25} {data[model]['overall_recall']:>15.1%} "
        f"{data[model]['snr_mean']:>10.2f} {data[model]['snr_std']:>10.2f}"
    )
print("=" * 60)

plt.show()
