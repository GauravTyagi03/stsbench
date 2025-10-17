import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import numpy as np

def create_recon_gif(directory, image_ids, output_path="recon_transition.gif",
                     duration=100, pause_seconds=2):
    if 'dorsal' in directory: 
        output_path = "../assets/dorsal_reconstructions.gif"
        val = "Dorsal Stream"
    else:
        output_path = "../assets/ventral_reconstructions.gif"
        val = "Ventral Stream"
        
    num_cols = len(image_ids)
    fig, axes = plt.subplots(1, num_cols, figsize=(2 * num_cols, 2))
    if num_cols == 1:
        axes = [axes]

    pause_frames = pause_seconds * 1000 // duration  # e.g., 1000 ms / 100 ms = 10 frames

    # Load images
    gt_images = []
    pred_images = []

    for img_id in image_ids:
        gt_path = os.path.join(directory, "diffusion", f"{img_id}_true.png")
        pred_path = os.path.join(directory, "diffusion", f"{img_id}_pred.png")
        if not os.path.exists(gt_path) or not os.path.exists(pred_path):
            raise FileNotFoundError(f"Missing image(s) for ID {img_id}")

        gt = mpimg.imread(gt_path).astype(np.float32)
        pred = mpimg.imread(pred_path).astype(np.float32)

        if gt.max() > 1.0:
            gt /= 255.0
        if pred.max() > 1.0:
            pred /= 255.0

        gt_images.append(gt)
        pred_images.append(pred)

    # Create blended transition frames
    frames = []

    def render_row(blended_images, save_path, title=None):
        fig, axes = plt.subplots(1, num_cols, figsize=(1.5 * num_cols, 2))
        if num_cols == 1:
            axes = [axes]
        for i, img in enumerate(blended_images):
            axes[i].imshow(np.clip(img, 0, 1))
            axes[i].axis('off')
        if title:
            fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=100)
        plt.close()

    # Fade GT → Diffusion
    blend_steps = np.linspace(0, 1, 10)

    for alpha in blend_steps:
        blended = [(1 - alpha) * gt + alpha * pred for gt, pred in zip(gt_images, pred_images)]
        render_row(blended, "frame.png", title="...")
        frames.append(Image.open("frame.png").convert("RGB"))

    # Pause on diffusion with title
    for _ in range(pause_frames):
        render_row(pred_images, "frame.png", title=val + " Reconstruction")
        frames.append(Image.open("frame.png").convert("RGB"))

    # Fade Diffusion → GT
    for alpha in blend_steps[::-1]:
        blended = [(1 - alpha) * gt + alpha * pred for gt, pred in zip(gt_images, pred_images)]
        render_row(blended, "frame.png", title="...")
        frames.append(Image.open("frame.png").convert("RGB"))

    # Pause on ground truth with title
    for _ in range(pause_frames):
        render_row(gt_images, "frame.png", title="Ground Truth")
        frames.append(Image.open("frame.png").convert("RGB"))

    # Save GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    )
    os.remove("./frame.png")

    print(f"GIF saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Create GT <-> Diffusion reconstruction GIF.")
    parser.add_argument('--directory', type=str, required=True,
                        help='Directory containing model subdirectories.')
    parser.add_argument('--model_names', type=str, nargs='+', required=True,
                        help='List of model names (only diffusion is used).')
    parser.add_argument('--image_ids', type=int, nargs='+', required=True,
                        help='List of image IDs to visualize.')

    args = parser.parse_args()

    if "diffusion" not in args.model_names:
        raise ValueError("This script only supports the 'diffusion' model for now.")

    create_recon_gif(args.directory, args.image_ids)

if __name__ == "__main__":
    main()

