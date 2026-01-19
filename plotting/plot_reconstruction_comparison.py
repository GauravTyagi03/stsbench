import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import re


def plot_reconstruction_comparison(img_directory, stream_type='ventral', output_path=None):
    """
    Create side-by-side comparison images of true vs reconstructed images.
    
    Args:
        img_directory: Directory containing pred and true images
        stream_type: 'ventral' or 'dorsal'
        output_path: Path to save the output image (default: img_directory/comparison.png)
    """
    # Find all image pairs
    pred_files = sorted(glob.glob(os.path.join(img_directory, '*_pred.png')))
    true_files = sorted(glob.glob(os.path.join(img_directory, '*_true.png')))
    
    # Extract image IDs from filenames
    image_ids = []
    for pred_file in pred_files:
        match = re.search(r'(\d+)_pred\.png', os.path.basename(pred_file))
        if match:
            img_id = match.group(1)
            true_file = os.path.join(img_directory, f'{img_id}_true.png')
            if os.path.exists(true_file):
                image_ids.append(img_id)
    
    if not image_ids:
        raise ValueError(f"No matching image pairs found in {img_directory}")
    
    # Sort image IDs numerically
    image_ids = sorted(image_ids, key=int)
    
    num_images = len(image_ids)
    # Arrange horizontally: true images on top row, reconstructions on bottom row
    num_rows = 2  # Top row: true, bottom row: reconstruction
    num_cols = num_images
    
    # Create figure with appropriate size (wider for horizontal layout)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(3*num_cols, 6))
    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    
    # Handle single column case
    if num_cols == 1:
        axes = [[axes[0]], [axes[1]]]
    
    stream_label = stream_type.capitalize() + " Stream"
    
    # Plot each image pair
    for col_idx, img_id in enumerate(image_ids):
        true_path = os.path.join(img_directory, f'{img_id}_true.png')
        pred_path = os.path.join(img_directory, f'{img_id}_pred.png')
        
        if not os.path.exists(true_path):
            raise FileNotFoundError(f"True image not found: {true_path}")
        if not os.path.exists(pred_path):
            raise FileNotFoundError(f"Predicted image not found: {pred_path}")
        
        # Load images
        true_img = mpimg.imread(true_path)
        pred_img = mpimg.imread(pred_path)
        
        # Plot true image (top row)
        ax_true = axes[0][col_idx]
        ax_true.imshow(true_img)
        ax_true.set_xticks([])
        ax_true.set_yticks([])
        ax_true.set_title(f'Image {img_id}: True', fontsize=12, fontweight='bold', pad=10)
        
        # Plot predicted image (bottom row)
        ax_pred = axes[1][col_idx]
        ax_pred.imshow(pred_img)
        ax_pred.set_xticks([])
        ax_pred.set_yticks([])
        ax_pred.set_title(f'Image {img_id}: Recon', fontsize=12, fontweight='bold', pad=10)
    
    # Add overall title
    fig.suptitle(f'{stream_label} Reconstructions', fontsize=16, fontweight='bold', y=0.995)
    
    # Determine output path
    if output_path is None:
        output_path = os.path.join(img_directory, f'{stream_type}_comparison.png')
    
    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Comparison image saved to: {output_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Create side-by-side comparison of true vs reconstructed images."
    )
    parser.add_argument(
        '--img_directory',
        type=str,
        default='reconstruction/img',
        help='Directory containing pred and true images (default: reconstruction/img, relative to project root)'
    )
    parser.add_argument(
        '--ventral',
        action='store_true',
        help='Label as ventral stream reconstructions (default if neither flag is specified)'
    )
    parser.add_argument(
        '--dorsal',
        action='store_true',
        help='Label as dorsal stream reconstructions (overrides --ventral)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for the comparison image (default: img_directory/{stream_type}_comparison.png)'
    )
    
    args = parser.parse_args()
    
    # Determine stream type
    stream_type = 'dorsal' if args.dorsal else 'ventral'
    
    # Resolve directory path
    if os.path.isabs(args.img_directory):
        img_dir = args.img_directory
    else:
        # Resolve relative to current working directory (typically project root)
        img_dir = os.path.abspath(args.img_directory)
    
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    
    # Resolve output path
    output = args.output
    if output is not None and not os.path.isabs(output):
        # If relative output path, resolve relative to current working directory
        output = os.path.abspath(output)
    
    plot_reconstruction_comparison(img_dir, stream_type, output)


if __name__ == "__main__":
    main()
