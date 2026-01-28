"""
Compare segmentation method accuracies against inter-annotator agreement.

This script loads validation results and inter-annotator agreement data,
then generates comparison tables in LaTeX and CSV format.

It now also includes validation against independent annotator masks to test
whether models learned generalizable features or annotation-specific biases.

Usage:
    python compare_methods_to_interannotator.py

Output:
    - interannotator_comparison.csv
    - interannotator_comparison.tex
    - interannotator_comparison_extended.csv (with independent annotator validation)
"""

import json
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================

VALIDATION_RESULTS = "../validation/validation_results.json"
VALIDATION_INDEPENDENT_RESULTS = "validation_independent_results.json"

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


def load_independent_validation():
    """Load validation results against independent annotator masks."""
    independent_path = Path(VALIDATION_INDEPENDENT_RESULTS)
    if not independent_path.exists():
        return None

    with open(independent_path) as fp:
        return json.load(fp)


def generate_extended_comparison():
    """
    Generate extended comparison showing accuracy against both annotators.

    This tests whether models learned generalizable features or just
    matched the primary annotator's style.
    """
    # Load both result sets
    with open(VALIDATION_RESULTS) as fp:
        validation_results = json.load(fp)

    independent_results = load_independent_validation()
    if independent_results is None:
        print("\nNote: Run validate_against_independent.py first to generate")
        print("      independent annotator validation results.")
        return

    print("\n" + "=" * 80)
    print("Extended Comparison: Accuracy vs Primary and Independent Annotators")
    print("=" * 80)
    print()
    print("This tests whether models learned generalizable nuclear features")
    print("or annotation-specific biases from the primary annotator.")
    print()

    # Prepare extended comparison table
    lines = [
        "Method,Dataset,vs_Primary,vs_Independent,Difference,Interpretation"
    ]

    # All methods to include in extended comparison
    all_methods = [
        ("StarDist 1CH", "stardist_1ch"),
        ("StarDist 2CH", "stardist_2ch"),
        ("StarDist 3CH", "stardist_3ch"),
        ("DAPIeq raw", "dapieq_raw"),
        ("DAPIeq post", "dapieq_post"),
    ]

    for dataset_name, dataset_key in DATASETS:
        print(f"\n{dataset_name}:")
        print(f"  Inter-annotator agreement: {INTERANNOTATOR_AGREEMENT[dataset_key]:.1%}")
        print()

        # Get results from independent validation
        if dataset_key not in independent_results.get("per_dataset", {}):
            print(f"  No independent validation data for {dataset_key}")
            continue

        dataset_data = independent_results["per_dataset"][dataset_key]

        print(f"  {'Method':<20} {'vs Primary':>12} {'vs Independent':>15} {'Diff':>8}")
        print(f"  {'-'*58}")

        for method_name, method_key in all_methods:
            vs_primary = dataset_data.get("vs_primary", {}).get(method_key, {})
            vs_independent = dataset_data.get("vs_independent", {}).get(method_key, {})

            if vs_primary and vs_independent:
                acc_primary = vs_primary.get("accuracy", 0)
                acc_independent = vs_independent.get("accuracy", 0)
                diff = acc_primary - acc_independent

                # Interpretation
                interannotator = INTERANNOTATOR_AGREEMENT[dataset_key]
                if abs(diff) < 0.02:
                    interpretation = "Generalizes well"
                elif diff > 0.05:
                    interpretation = "May have annotation bias"
                elif acc_independent >= interannotator:
                    interpretation = "Robust across annotators"
                else:
                    interpretation = "Needs investigation"

                print(f"  {method_name:<20} {acc_primary:>12.1%} {acc_independent:>15.1%} {diff:>+8.1%}")

                lines.append(
                    f"{method_name},{dataset_name},{acc_primary:.3f},{acc_independent:.3f},{diff:+.3f},{interpretation}"
                )

        # Add inter-annotator baseline
        interannotator = INTERANNOTATOR_AGREEMENT[dataset_key]
        print(f"  {'Inter-annotator':<20} {interannotator:>12.1%} {interannotator:>15.1%} {0:>+8.1%}")
        lines.append(
            f"Inter-annotator,{dataset_name},{interannotator:.3f},{interannotator:.3f},0.000,Baseline"
        )

    # Save extended CSV
    csv_path = "interannotator_comparison_extended.csv"
    with open(csv_path, "w") as fp:
        fp.write("\n".join(lines))
    print(f"\nExtended comparison saved to: {csv_path}")

    # Print summary interpretation
    print("\n" + "=" * 80)
    print("Interpretation Guide")
    print("=" * 80)
    print()
    print("1. If vs_Independent ≈ vs_Primary (diff < 2%):")
    print("   → Model learned generalizable nuclear features")
    print()
    print("2. If vs_Independent < vs_Primary (diff > 5%):")
    print("   → Model may have learned primary annotator's style/biases")
    print()
    print("3. If vs_Independent > inter-annotator agreement:")
    print("   → Model is robust and performs well across annotation styles")
    print()
    print("4. Key insight: If the best models (StarDist 2CH/3CH) maintain high")
    print("   accuracy against independent annotations, they're learning real")
    print("   nuclear boundaries, not just mimicking training annotations.")


if __name__ == "__main__":
    main()

    # Generate extended comparison if independent validation exists
    generate_extended_comparison()
