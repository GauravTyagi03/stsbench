import os
import numpy as np
from collections import defaultdict

# Human-readable model names
MODEL_NAME_MAP = {
    "mean": "Mean",
    "shuffled": "Shuffled",
    "linear": "Linear",
    "cnn": "CNN Decoder",
    "diffusion": "Diffusion",
}

# Order of appearance in table
MODEL_ORDER = [
    "mean",
    "shuffled",
    "linear",
    "cnn",
    "diffusion",
]

# Metrics to report
METRICS = ["PSNR","LPIPS"]


def load_results(log_dir):
    results = defaultdict(lambda: defaultdict(dict))
    for fname in os.listdir(log_dir):
        if fname.endswith(".npy") and ("dorsal" in fname or "ventral" in fname):
            parts = fname.replace(".npy", "").split("_")
            dataset = parts[0]
            metric = parts[-1]
            model = "_".join(parts[1:-1])
            value = np.load(os.path.join(log_dir, fname)).mean() # load logged metric & compute mean
            results[dataset][model][metric] = value
    return results


def format_latex_table(dataset_label, dataset_key, data):
    lines = []
    lines.append("\\begin{table}")
    lines.append(f"  \\caption{{Reconstruction Results ({dataset_label} Stream). Best value for each metric is bolded (max for PSNR, min for LPIPS).}}")
    lines.append(f"  \\label{{tab:recon-{dataset_key}}}")
    lines.append("  \\centering")
    lines.append("  \\begin{tabular}{lcc}")
    lines.append("    \\toprule")
    lines.append("Model & PSNR & SSIM & LPIPS \\\\")
    lines.append("    \\midrule")

    # compute max per metric to determine what to bold in the table
    metric_max = {metric: float("-inf") for metric in METRICS}
    for model in MODEL_ORDER:
        if model in data:
            for metric in METRICS:
                val = data[model].get(metric, float("-inf"))
                if metric == "LPIPS":
                    if val < metric_max[metric] or metric_max[metric] == float("-inf"):
                        metric_max[metric] = val
                else:
                    if val > metric_max[metric]:
                        metric_max[metric] = val

    # add rows to table
    for model in MODEL_ORDER:
        if model in data:
            row = [MODEL_NAME_MAP.get(model, model)]
            for metric in METRICS:
                val = data[model].get(metric, None)
                if val is None:
                    row.append("N/A")
                else:
                    is_best = (
                        val == metric_max[metric]
                        if metric != "LPIPS"
                        else val == metric_max[metric]
                    )
                    val_str = f"{val:.3f}"
                    row.append(f"$\\mathbf{{{val_str}}}$" if is_best else val_str)
            lines.append(" & ".join(row) + " \\\\")

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def main():
    log_dir = "../reconstruction/logs/"
    results = load_results(log_dir)

    for dataset_key, label in [("dorsal", "Dorsal"), ("ventral", "Ventral")]:
        table = format_latex_table(label, dataset_key, results.get(dataset_key, {}))
        print("\n" + table + "\n")


if __name__ == "__main__":
    main()
