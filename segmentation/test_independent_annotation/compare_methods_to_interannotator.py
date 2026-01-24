"""
Compare segmentation method accuracies against inter-annotator agreement.

This script loads validation results and inter-annotator agreement data,
then generates comparison tables in LaTeX and CSV format.

Usage:
    python compare_methods_to_interannotator.py

Output:
    - interannotator_comparison.csv
    - interannotator_comparison.tex
"""

import json
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================

VALIDATION_RESULTS = "../validation/validation_results.json"

# Inter-annotator agreement rates (from find_disagreeing_labels.py)
INTERANNOTATOR_AGREEMENT = {
    "ht1080_20x": 0.805,  # 650/807 = 80.5%
    "ht1080_40x": 0.898,  # 679/756 = 89.8%
}

# Methods to include in comparison (in order)
METHODS = [
    ("Inter-annotator", "interannotator"),
    ("DAPIeq raw StarDist", "dapieq_raw_stardist"),
    ("DAPIeq post StarDist", "dapieq_post_stardist"),
    ("DAPIeq Cellpose", "dapieq_post_cellpose"),
    ("DAPIeq Cellpose Denoise", "dapieq_cellpose_denoise"),
    ("ConfluentFUCCI", "confluentfucci_method"),
    ("InstanSeg 1CH", "instanseg_1ch"),
    ("InstanSeg 2CH", "instanseg_2ch"),
    ("InstanSeg 3CH", "instanseg_3ch"),
    ("StarDist 1CH", "stardist_1ch"),
    ("StarDist 2CH", "stardist_2ch"),
    ("StarDist 3CH", "stardist_3ch"),
]

# Datasets to include
DATASETS = [
    ("HT1080 20x", "ht1080_20x"),
    ("HT1080 40x", "ht1080_40x"),
]


# =============================================================================
# Main
# =============================================================================


def main():
    # Load validation results
    with open(VALIDATION_RESULTS) as fp:
        validation_results = json.load(fp)

    # Add inter-annotator agreement to results
    for dataset_key, agreement in INTERANNOTATOR_AGREEMENT.items():
        if dataset_key in validation_results:
            validation_results[dataset_key]["interannotator"] = agreement

    # Build comparison table
    print("=" * 80)
    print("Method Accuracy vs Inter-Annotator Agreement")
    print("=" * 80)

    # Print header
    header = ["Method"] + [name for name, _ in DATASETS]
    print(f"\n{header[0]:<30} " + " ".join(f"{h:>12}" for h in header[1:]))
    print("-" * 60)

    # Collect data for CSV/LaTeX
    table_data = []

    for method_name, method_key in METHODS:
        row = [method_name]
        for dataset_name, dataset_key in DATASETS:
            if dataset_key in validation_results:
                value = validation_results[dataset_key].get(method_key)
                if value is not None:
                    row.append(f"{value:.3f}")
                else:
                    row.append("n/a")
            else:
                row.append("n/a")

        table_data.append(row)
        print(f"{row[0]:<30} " + " ".join(f"{v:>12}" for v in row[1:]))

    # Save CSV
    csv_path = "interannotator_comparison.csv"
    with open(csv_path, "w") as fp:
        fp.write(",".join(header) + "\n")
        for row in table_data:
            fp.write(",".join(row) + "\n")
    print(f"\nSaved: {csv_path}")

    # Generate LaTeX table
    latex_path = "interannotator_comparison.tex"
    with open(latex_path, "w") as fp:
        fp.write("\\documentclass{standalone}\n")
        fp.write("\\usepackage{booktabs}\n")
        fp.write("\\usepackage{xcolor}\n")
        fp.write("\n")
        fp.write("\\begin{document}\n")
        fp.write("\n")

        # Table
        n_cols = len(DATASETS) + 1
        fp.write(f"\\begin{{tabular}}{{l{'r' * len(DATASETS)}}}\n")
        fp.write("\\toprule\n")

        # Header
        fp.write("\\textbf{Method}")
        for name, _ in DATASETS:
            fp.write(f" & \\textbf{{{name}}}")
        fp.write(" \\\\\n")
        fp.write("\\midrule\n")

        # Data rows
        for row in table_data:
            method_name = row[0]
            values = row[1:]

            # Bold inter-annotator row
            if "Inter-annotator" in method_name:
                fp.write(f"\\textit{{{method_name}}}")
            else:
                fp.write(method_name)

            for i, v in enumerate(values):
                if v != "n/a":
                    # Check if this is the best value for this dataset
                    dataset_key = DATASETS[i][1]
                    all_values = []
                    for _, mk in METHODS:
                        if mk != "interannotator" and dataset_key in validation_results:
                            val = validation_results[dataset_key].get(mk)
                            if val is not None:
                                all_values.append(val)

                    current_val = float(v) if v != "n/a" else 0
                    is_best = current_val == max(all_values) if all_values else False

                    if is_best and "Inter-annotator" not in method_name:
                        fp.write(f" & \\textbf{{{v}}}")
                    else:
                        fp.write(f" & {v}")
                else:
                    fp.write(" & n/a")

            fp.write(" \\\\\n")

            # Add line after inter-annotator
            if "Inter-annotator" in method_name:
                fp.write("\\midrule\n")

        fp.write("\\bottomrule\n")
        fp.write("\\end{tabular}\n")
        fp.write("\n")
        fp.write("\\end{document}\n")

    print(f"Saved: {latex_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    for dataset_name, dataset_key in DATASETS:
        if dataset_key not in validation_results:
            continue

        interannotator = INTERANNOTATOR_AGREEMENT.get(dataset_key, 0)
        print(f"\n{dataset_name} (Inter-annotator: {interannotator:.1%}):")

        methods_above = []
        methods_below = []

        for method_name, method_key in METHODS:
            if method_key == "interannotator":
                continue
            value = validation_results[dataset_key].get(method_key)
            if value is not None:
                if value >= interannotator:
                    methods_above.append((method_name, value))
                else:
                    methods_below.append((method_name, value))

        print(f"  Methods >= inter-annotator: {len(methods_above)}")
        for name, val in sorted(methods_above, key=lambda x: -x[1]):
            print(f"    {name}: {val:.1%}")

        print(f"  Methods < inter-annotator: {len(methods_below)}")


if __name__ == "__main__":
    main()
