import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def plot_reconstruction_grid(directory, model_names, image_ids):
    # Map internal model names to nicer display names
    model_name_map = {
        "diffusion": "Diffusion",
        "linear": "Linear",
        "cnn": "CNN"
    }
    
    num_models = len(model_names)
    num_rows = num_models + 1  # +1 for ground truth
    num_cols = len(image_ids)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(2*num_cols, 2*num_rows))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    # Always make axes 2D for consistency
    if num_rows == 1:
        axes = [axes]
    if num_cols == 1:
        axes = [[ax] for ax in axes]

    # --- Plot ground truth on top row ---
    for col_idx, img_id in enumerate(image_ids):
        img_path = os.path.join(directory, model_names[0], f"{img_id}_true.png")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Ground truth image not found: {img_path}")
        img = mpimg.imread(img_path)
        ax = axes[0][col_idx]
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])

        if col_idx == 0:
            ax.set_ylabel("Ground truth", fontsize=16, fontweight='bold', rotation=90, labelpad=20, va='center')

    # --- Plot model reconstructions below ---
    for row_idx, model_name in enumerate(model_names):
        for col_idx, img_id in enumerate(image_ids):
            img_path = os.path.join(directory, model_name, f"{img_id}_pred.png")
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")
            img = mpimg.imread(img_path)
            ax = axes[row_idx + 1][col_idx]  # +1 because ground truth is at 0
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])

            if col_idx == 0:
                label = model_name_map.get(model_name, model_name)
                ax.set_ylabel(label, fontsize=16, fontweight='bold', rotation=90, labelpad=20, va='center')
    if "dorsal" in directory:
        plt.savefig("figures/dorsal_recon.png")
    elif "ventral" in directory: 
        plt.savefig(f"figures/ventral_recon_{image_ids[0]}.png")
        
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot grid of reconstructed images.")
    parser.add_argument('--directory', type=str, required=True,
                        help='Directory containing model subdirectories.')
    parser.add_argument('--model_names', type=str, nargs='+', required=True,
                        help='List of model names.')
    parser.add_argument('--image_ids', type=int, nargs='+', required=True,
                        help='List of image IDs to plot.')

    args = parser.parse_args()

    plot_reconstruction_grid(args.directory, args.model_names, args.image_ids)

if __name__ == "__main__":
    main()