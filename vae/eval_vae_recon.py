"""
Evaluate VAE-conditioned reconstruction quality (PSNR + LPIPS).
================================================================
Drop-in equivalent of reconstruction/eval.py but takes an explicit
--image_dir so it works with any output folder regardless of how the
images were generated.

Usage:
    python eval_vae_recon.py \
        --image_dir /path/to/folder/containing/{i}_pred.png \
        --run_name  vae_z128_wide_nb2_ts_k3_skip \
        --output_dir /oak/.../stsbench/vae/logs/eval
"""

import argparse
import os
import random

import numpy as np
from PIL import Image
import PIL
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

eval_tf = transforms.ToTensor()
lpips_fn = lpips.LPIPS(net='alex').to(device)


def load_pil(path):
    return Image.open(path).convert('RGB').resize((150, 150))


def compute_metrics(gen_pil, true_pil):
    assert isinstance(gen_pil,  PIL.Image.Image)
    assert isinstance(true_pil, PIL.Image.Image)

    g = eval_tf(gen_pil)
    t = eval_tf(true_pil)

    g8 = (g.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    t8 = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    ps = psnr(t8, g8, data_range=255)

    lp = lpips_fn(
        2 * (g.unsqueeze(0).to(device) - 0.5),
        2 * (t.unsqueeze(0).to(device) - 0.5),
    ).item()

    return ps, lp


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate VAE-conditioned reconstructions (PSNR + LPIPS)'
    )
    parser.add_argument('--image_dir',  required=True, type=str,
                        help='Directory containing {i}_pred.png and {i}_true.png')
    parser.add_argument('--run_name',   required=True, type=str,
                        help='Label used for output .npy file names')
    parser.add_argument('--output_dir', required=True, type=str,
                        help='Directory to write metric .npy files and summary')
    parser.add_argument('--n',          default=100, type=int,
                        help='Number of test samples (default: 100 for ventral)')
    parser.add_argument('--k',          default=5,   type=int,
                        help='Shuffled bootstrap replicates per sample (default: 5)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    gen_fns  = [f'{i}_pred.png' for i in range(args.n)]
    true_fns = [f'{i}_true.png' for i in range(args.n)]

    # verify files exist before loading
    missing = [f for f in gen_fns + true_fns
               if not os.path.exists(os.path.join(args.image_dir, f))]
    if missing:
        raise FileNotFoundError(
            f'{len(missing)} image files not found in {args.image_dir}.\n'
            f'First missing: {missing[0]}'
        )

    print(f'Loading {args.n} pairs from: {args.image_dir}')
    gen_imgs  = [load_pil(os.path.join(args.image_dir, f)) for f in gen_fns]
    true_imgs = [load_pil(os.path.join(args.image_dir, f)) for f in true_fns]

    null_img = Image.new('RGB', (150, 150), (128, 128, 128))

    true_m, shuf_m, null_m = [], [], []
    random.seed(42)
    for i in range(args.n):
        true_m.append(compute_metrics(gen_imgs[i], true_imgs[i]))
        cands = [j for j in range(args.n) if j != i]
        picks = random.sample(cands, args.k)
        shuf_m.append([compute_metrics(true_imgs[i], true_imgs[j]) for j in picks])
        null_m.append(compute_metrics(true_imgs[i], null_img))

    true_m = np.array(true_m)         # (n, 2)
    shuf_m = np.array(shuf_m)         # (n, k, 2)
    null_m = np.array(null_m)         # (n, 2)

    shuf_mean = shuf_m.mean(axis=1)   # (n, 2)
    shuf_sem  = shuf_m.std(axis=1, ddof=1) / np.sqrt(args.n - 1)

    metric_names = ['PSNR', 'LPIPS']
    summary_lines = [f'Run: {args.run_name}', f'Image dir: {args.image_dir}', '']
    for idx, name in enumerate(metric_names):
        t_avg = true_m[:, idx].mean()
        s_avg = shuf_mean[:, idx].mean()
        s_sem = shuf_sem[:, idx].mean()
        m_avg = null_m[:, idx].mean()
        line = f'{name} – True: {t_avg:.4f} | Shuffled: {s_avg:.4f} ± {s_sem:.4f} | Null: {m_avg:.4f}'
        print(line)
        summary_lines.append(line)

        np.save(os.path.join(args.output_dir, f'{args.run_name}_{name}.npy'),        true_m[:, idx])
        np.save(os.path.join(args.output_dir, f'{args.run_name}_shuffled_{name}.npy'), shuf_mean[:, idx])
        np.save(os.path.join(args.output_dir, f'{args.run_name}_null_{name}.npy'),    null_m[:, idx])

    summary_path = os.path.join(args.output_dir, f'{args.run_name}_summary.txt')
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines) + '\n')
    print(f'\nSummary written to: {summary_path}')


if __name__ == '__main__':
    main()
