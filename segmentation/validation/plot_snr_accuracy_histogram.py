"""
Plot SNR vs Detection Rate histogram for 1CH, 2CH, 3CH models.

Includes:
- Recall (detection rate) binned by SNR
- Overall metrics: Recall, Precision, Accuracy (Jaccard)

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

# =============================================================================
# Figure 1: Bar chart - Recall vs SNR bins
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

bar_width = 0.25
x = np.arange(n_bins)

for i, (model, color, label) in enumerate(zip(models, colors, labels)):
    recalls = data[model]["recalls_per_bin"]
    offset = (i - 1) * bar_width
    ax.bar(x + offset, recalls, bar_width, label=label, color=color, alpha=0.8)

ax.set_xlabel("SNR Bin Center", fontsize=12)
ax.set_ylabel("Recall (Detection Rate)", fontsize=12)
ax.set_title(f"Recall vs SNR - {dataset} Dataset (IoU=0.5)", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels([f"{s:.1f}" for s in snr_centers], rotation=45, ha="right")
ax.set_ylim(0, 1.05)
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)

# Add overall metrics annotation
for i, (model, label) in enumerate(zip(models, labels)):
    recall = data[model]["overall_recall"]
    precision = data[model]["overall_precision"]
    accuracy = data[model]["overall_accuracy"]
    ax.text(
        0.98,
        0.95 - i * 0.08,
        f"{label}: Acc={accuracy:.1%}, R={recall:.1%}, P={precision:.1%}",
        transform=ax.transAxes,
        ha="right",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

plt.tight_layout()
plt.savefig("snr_recall_histogram_validation.pdf", dpi=300, bbox_inches="tight")
plt.savefig("snr_recall_histogram_validation.png", dpi=300, bbox_inches="tight")
print("Saved: snr_recall_histogram_validation.pdf/png")

# =============================================================================
# Figure 2: Line plot - Recall vs SNR
# =============================================================================
fig2, ax2 = plt.subplots(figsize=(10, 6))

for model, color, label in zip(models, colors, labels):
    recalls = data[model]["recalls_per_bin"]
    accuracy = data[model]["overall_accuracy"]
    ax2.plot(
        snr_centers,
        recalls,
        marker="o",
        color=color,
        linewidth=2,
        markersize=8,
        label=f"{label} (Acc: {accuracy:.1%})",
    )

ax2.set_xlabel("Signal-to-Noise Ratio (SNR)", fontsize=12)
ax2.set_ylabel("Recall (Detection Rate)", fontsize=12)
ax2.set_title(f"Recall vs SNR - {dataset} Dataset (IoU=0.5)", fontsize=14)
ax2.set_ylim(0, 1.05)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("snr_recall_lineplot_validation.pdf", dpi=300, bbox_inches="tight")
plt.savefig("snr_recall_lineplot_validation.png", dpi=300, bbox_inches="tight")
print("Saved: snr_recall_lineplot_validation.pdf/png")

# =============================================================================
# Figure 3: Overall metrics comparison bar chart
# =============================================================================
fig3, ax3 = plt.subplots(figsize=(10, 6))

metrics = ["Recall", "Precision", "Accuracy"]
metric_keys = ["overall_recall", "overall_precision", "overall_accuracy"]
x_metrics = np.arange(len(metrics))
bar_width = 0.25

for i, (model, color, label) in enumerate(zip(models, colors, labels)):
    values = [data[model][k] for k in metric_keys]
    offset = (i - 1) * bar_width
    bars = ax3.bar(x_metrics + offset, values, bar_width, label=label, color=color, alpha=0.8)
    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

ax3.set_xlabel("Metric", fontsize=12)
ax3.set_ylabel("Score", fontsize=12)
ax3.set_title(f"Detection Metrics Comparison - {dataset} Dataset (IoU=0.5)", fontsize=14)
ax3.set_xticks(x_metrics)
ax3.set_xticklabels(metrics, fontsize=11)
ax3.set_ylim(0, 1.15)
ax3.legend(fontsize=11, loc="lower right")
ax3.grid(axis="y", alpha=0.3)

# Add formula annotations
ax3.text(0.02, 0.98, "Recall = TP/(TP+FN)", transform=ax3.transAxes, fontsize=8, va="top")
ax3.text(0.02, 0.93, "Precision = TP/(TP+FP)", transform=ax3.transAxes, fontsize=8, va="top")
ax3.text(0.02, 0.88, "Accuracy = TP/(TP+FN+FP)", transform=ax3.transAxes, fontsize=8, va="top")

plt.tight_layout()
plt.savefig("metrics_comparison_validation.pdf", dpi=300, bbox_inches="tight")
plt.savefig("metrics_comparison_validation.png", dpi=300, bbox_inches="tight")
print("Saved: metrics_comparison_validation.pdf/png")

# =============================================================================
# Export to CSV
# =============================================================================
df = pd.DataFrame(
    {
        "SNR_bin_center": snr_centers,
        "recall_1CH": data[models[0]]["recalls_per_bin"],
        "recall_2CH": data[models[1]]["recalls_per_bin"],
        "recall_3CH": data[models[2]]["recalls_per_bin"],
        "counts_per_bin": data[models[0]]["counts_per_bin"],
    }
)
df.to_csv("snr_recall_validation_detailed.csv", index=False)
print("Saved: snr_recall_validation_detailed.csv")

# Export overall metrics
metrics_df = pd.DataFrame(
    {
        "Model": labels,
        "TP": [data[m]["tp"] for m in models],
        "FN": [data[m]["fn"] for m in models],
        "FP": [data[m]["fp"] for m in models],
        "Recall": [data[m]["overall_recall"] for m in models],
        "Precision": [data[m]["overall_precision"] for m in models],
        "Accuracy": [data[m]["overall_accuracy"] for m in models],
        "SNR_mean": [data[m]["snr_mean"] for m in models],
        "SNR_std": [data[m]["snr_std"] for m in models],
    }
)
metrics_df.to_csv("metrics_summary_validation.csv", index=False)
print("Saved: metrics_summary_validation.csv")

# =============================================================================
# Print summary
# =============================================================================
print("\n" + "=" * 80)
print(f"Summary - {dataset} Dataset")
print("=" * 80)
print(f"{'Model':<10} {'TP':>6} {'FN':>6} {'FP':>6} {'Recall':>10} {'Precision':>10} {'Accuracy':>10}")
print("-" * 80)
for model, label in zip(models, labels):
    d = data[model]
    print(
        f"{label:<10} {d['tp']:>6} {d['fn']:>6} {d['fp']:>6} "
        f"{d['overall_recall']:>10.1%} {d['overall_precision']:>10.1%} {d['overall_accuracy']:>10.1%}"
    )
print("=" * 80)

plt.show()
