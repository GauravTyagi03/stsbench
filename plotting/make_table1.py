import os
import numpy as np
from glob import glob
from collections import defaultdict
import argparse

model_name_map = {
    'gabor3d': '3D Gabor',
    'r3d18': '3D ResNet - Kinetics',
    'dorsalnet': '3D ResNet - Self Motion',
    'resnet18': '2D ResNet - ImageNet',
    'simple3d1': '3D CNN-1', 
    'simple3d3': '3D CNN-3', 
    'simple3d5': '3D CNN-5',
    'simple3d7': '3D CNN-7',
}

model_training_map = {
    'gabor3d': 'aHand-tuned',
    'r3d18': 'bPretrained',
    'dorsalnet': 'bPretrained',
    'resnet18': 'bPretrained',
    'simple3d1': 'cEnd-to-end', 
    'simple3d3': 'cEnd-to-end', 
    'simple3d5': 'cEnd-to-end',
    'simple3d7': 'cEnd-to-end',
}    

layer_name_map = {
    'gabor3d': {'layer1': 'Simple', 'layer2': 'Complex'},
    'r3d18': {f'layer{i}': f'Layer {i}' for i in range(1, 5)},
    'dorsalnet': {'s1': 'Layer 1', 'res0': 'Layer 2', 'res1': 'Layer 3', 'res2': 'Layer 4'},
    'resnet18': {f'layer{i}': f'Layer {i}' for i in range(1, 5)},
    'simple3d1': {f'none': f'Layer 1'},
    'simple3d3': {f'none': f'Layer 3'},
    'simple3d5': {f'none': f'Layer 5'},
    'simple3d7': {f'none': f'Layer 7'},
}

def parse_filename(filename):
    base = os.path.basename(filename).replace(".npy", "")
    parts = base.split("_")
    if len(parts) < 5:
        raise ValueError(f"Unexpected filename format: {filename}")
    dv = parts[0]
    model = parts[2]
    size = int(parts[3])
    layer = parts[4]
    return dv, model, size, layer

def compute_stats(data):
    mean = np.nanmean(data ** 2) ## this squares the correlation for average (Pearson corr) for each neuron to yield R^2, then takes mean over neurons
    stderr = np.nanstd(data ** 2, ddof=1) / np.sqrt(len(data) - np.sum(np.isnan(data)))
    return mean, stderr

def format_table_latex_main(sorted_entries, max_entry):
    lines = []
    lines.append("\\begin{table}")
    lines.append("  \\caption{Encoding Model Results. Coefficient of determination results are reported as mean $\\pm$ standard error of the mean across neurons in the dataset. The coefficient of determination of the best fitting model is bolded.}")
    lines.append("  \\label{tab:encoding-supplement}")
    lines.append("  \\centering")
    lines.append("  \\setlength\\extrarowheight{-3pt}")
    lines.append("  \\begin{tabular}{llll}")
    lines.append("    \\toprule")
    lines.append("Training Scheme & Model Name & Layer & $R^2$ \\\\")
    lines.append("    \\midrule")

    last_model = None
    last_layer = None
    last_train = None
    entries = list(sorted_entries.items())

    for idx, ((train_disp, model_disp, layer_disp), rows) in enumerate(entries):
        if len(rows) > 1:
            max_row = 0
            max_res = 0
            for i in range(len(rows)):
                if rows[i][1] > max_res:
                    max_res = rows[i][1]
                    max_row = i
            rows = [rows[max_row]]

        # Determine if this is a new model
        new_train = train_disp != last_train
        new_model = model_disp != last_model or new_train
        new_layer = layer_disp != last_layer or new_model or new_train

        if new_train and last_train is not None:
            lines.append("    \\midrule")
        elif new_model and last_model is not None:
            lines.append("    \\cmidrule{2-4}")
        elif new_layer and last_layer is not None:
            lines.append("    \\cmidrule{3-4}")

        for i, (size, mean, stderr) in enumerate(rows):
            is_global_max = (
                mean == max_entry[0]
                and train_disp == max_entry[1]
                and model_disp == max_entry[2]
                and layer_disp == max_entry[3]
                and size == max_entry[4]
            )
            value_str = f"${mean:.3f} \\pm {stderr:.3f}$"
            if is_global_max:
                value_str = f"$\\mathbf{{{mean:.3f} \\pm {stderr:.3f}}}$"

            # First row in a new model prints model name
            train_cell = train_disp[1:] if i == 0 and new_train else ""
            model_cell = model_disp if i == 0 and new_model else ""
            # First row in a new layer prints layer name
            layer_cell = layer_disp if i == 0 else ""

            lines.append(f" {train_cell} &   {model_cell} & {layer_cell} & {value_str} \\\\")

        last_train = train_disp
        last_model = model_disp
        last_layer = layer_disp

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)
    
def format_table_latex_sup(sorted_entries, max_entry):
    lines = []
    lines.append("\\begin{table}")
    lines.append("  \\caption{Encoding Model Results. Coefficient of determination results are reported as mean $\\pm$ standard error of the mean across neurons in the dataset. The coefficient of determination of the best fitting model is bolded.}")
    lines.append("  \\label{tab:encoding-supplement}")
    lines.append("  \\centering")
    lines.append("  \\setlength\\extrarowheight{-3pt}")
    lines.append("  \\begin{tabular}{lllll}")
    lines.append("    \\toprule")
    lines.append("Training Scheme & Model Name & Layer & Input Size (px) & R^2 \\\\")
    lines.append("    \\midrule")

    last_model = None
    last_layer = None
    last_train = None
    entries = list(sorted_entries.items())

    for idx, ((train_disp, model_disp, layer_disp), rows) in enumerate(entries):
        rows.sort(key=lambda x: x[0])  # sort by input size

        # Determine if this is a new model
        new_train = train_disp != last_train
        new_model = model_disp != last_model or new_train
        new_layer = layer_disp != last_layer or new_model or new_train

        if new_train and last_train is not None:
            lines.append("    \\midrule")
        elif new_model and last_model is not None:
            lines.append("    \\cmidrule{2-5}")
        elif new_layer and last_layer is not None:
            lines.append("    \\cmidrule{3-5}")

        for i, (size, mean, stderr) in enumerate(rows):
            is_global_max = (
                mean == max_entry[0]
                and train_disp == max_entry[1]
                and model_disp == max_entry[2]
                and layer_disp == max_entry[3]
                and size == max_entry[4]
            )
            value_str = f"${mean:.3f} \\pm {stderr:.3f}$"
            if is_global_max:
                value_str = f"$\\mathbf{{{mean:.3f} \\pm {stderr:.3f}}}$"

            # First row in a new model prints model name
            train_cell = train_disp[1:] if i == 0 and new_train else ""
            model_cell = model_disp if i == 0 and new_model else ""
            # First row in a new layer prints layer name
            layer_cell = layer_disp if i == 0 else ""

            lines.append(f" {train_cell} &   {model_cell} & {layer_cell} & {size} & {value_str} \\\\")

            # Insert cmidrule between input sizes of the same layer
            if i < len(rows) - 1:
                lines.append("    \\cmidrule{4-5}")
        last_train = train_disp
        last_model = model_disp
        last_layer = layer_disp

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def main(log_dir="../encoding/logs/", dset='dorsal'):
    files = glob(os.path.join(log_dir, "*_test.npy"))
    results = defaultdict(list)  # key: (model_disp, layer_disp) -> list of (size, mean, stderr)

    max_mean = -np.inf
    max_entry = (None, None, None, None, None)  # (mean, model_disp, layer_disp, size)

    for f in files:
        dv, model, size, layer = parse_filename(f)
        if dv == dset and model in model_name_map.keys():
            data = np.load(f)
            mean, stderr = compute_stats(data)
    
            model_disp = model_name_map.get(model, model)
            layer_disp = layer_name_map.get(model, {}).get(layer, layer)
            train_disp = model_training_map.get(model, model)
    
            results[(train_disp, model_disp, layer_disp)].append((size, mean, stderr))
    
            if mean > max_mean:
                max_mean = mean
                max_entry = (mean, train_disp, model_disp, layer_disp, size)

    # Sort entries alphabetically by model, then by numeric layer
    def layer_sort_key(layer_name):
        try:
            return int(layer_name.split()[-1])
        except:
            return 99

    sorted_results = dict(sorted(results.items(), key=lambda x: (x[0][0], x[0][1], layer_sort_key(x[0][2]))))

    latex_output = format_table_latex_sup(sorted_results, max_entry)
    file = open("./tables/tableS1.tex", 'w')
    file.write(latex_output)
    file.close()

    latex_output = format_table_latex_main(sorted_results, max_entry)
    file = open("./tables/table1.tex", 'w')
    file.write(latex_output)
    file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dset', type=str, required=False, default='dorsal',
                        help='Directory containing model subdirectories.')

    args = parser.parse_args()

    main(dset=args.dset)




