import os
import numpy as np
import random
import argparse
from PIL import Image
import PIL

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
import lpips

from skimage.metrics import peak_signal_noise_ratio as psnr

from utils import load_config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

eval_tf = transforms.ToTensor()

lpips_fn = lpips.LPIPS(net='alex').to(device)

# Args
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./configs/dorsal_stream_diffusion.yaml')
args = parser.parse_args()
config = load_config(args.config)
name = config['name']
model_name = config['model_name']
image_dir = f"./logs/{name}/{model_name}/"
model_name = image_dir.rstrip('/').split('/')[-1]
print("Using image directory:", image_dir)

# Test dataset size
if 'dorsal_stream' in image_dir:
    dset, n = 'dorsal', 40
elif 'ventral_stream' in image_dir:
    dset, n = 'ventral', 100
else:
    raise ValueError('unknown dataset')

gen_fns  = [f"{i}_pred.png" for i in range(n)]
true_fns = [f"{i}_true.png" for i in range(n)]

def load_pil(path):
    return Image.open(path).convert('RGB').resize((150, 150)) # warning - note this is manually specified, careful if you want to change!

def compute_metrics(gen_pil, true_pil):
    assert isinstance(gen_pil, PIL.Image.Image), "gen_pil must be a PIL.Image"
    assert isinstance(true_pil, PIL.Image.Image), "true_pil must be a PIL.Image"

    g = eval_tf(gen_pil); t = eval_tf(true_pil) # convert PIL images to tensors 

    assert g.ndim == 3 and g.shape[0] == 3, f"gen image must be [3, H, W], got {g.shape}"
    assert t.ndim == 3 and t.shape[0] == 3, f"true image must be [3, H, W], got {t.shape}"
    assert (0 <= g).all() and (g <= 1).all(), "gen image values must be in [0, 1]"
    assert (0 <= t).all() and (t <= 1).all(), "true image values must be in [0, 1]"

    # see skimage for docs on image data types and channel ordering - https://scikit-image.org/docs/0.25.x/user_guide/data_types.html
    g8 = (g.permute(1,2,0).numpy()*255).astype(np.uint8)
    t8 = (t.permute(1,2,0).numpy()*255).astype(np.uint8)
    ps = psnr(t8, g8, data_range=255)

    # see lpips library for docs on image range - https://github.com/richzhang/PerceptualSimilarity/tree/master
    lp = lpips_fn(2*(g.unsqueeze(0).to(device)-0.5),
                  2*(t.unsqueeze(0).to(device)-0.5)
    ).item()
    
    return ps, lp

# Load images
gen_imgs  = [load_pil(os.path.join(image_dir,f)) for f in gen_fns]
true_imgs = [load_pil(os.path.join(image_dir,f)) for f in true_fns]

null_metrics = []
null_img = Image.new('RGB', (150, 150), (128, 128, 128)) # this is our 'mean' control

# compute metrics - our shuffles are bootstrapped over 5 replicates, this is seeded for replicability
true_m, shuf_m = [], []
random.seed(42)
k=5
for i in range(n):
    true_m.append(compute_metrics(gen_imgs[i], true_imgs[i]))
    cands = [j for j in range(n) if j!=i]
    picks = random.sample(cands, k)
    sm = [compute_metrics(true_imgs[i], true_imgs[j]) for j in picks]
    shuf_m.append(sm)
    null_metrics.append(compute_metrics(true_imgs[i], null_img))

true_m = np.array(true_m)         # (n,2)
shuf_m = np.array(shuf_m)         # (n,k,2)
null_metrics = np.array(null_metrics)

shuf_mean = shuf_m.mean(axis=1)   # (n,2)
shuf_sem  = shuf_m.std(axis=1, ddof=1)/np.sqrt(n-1)

# log and print eval results 
out = "./logs/"; os.makedirs(out, exist_ok=True)
metrics = ["PSNR","LPIPS"]
for idx,name in enumerate(metrics):
    t_avg = true_m[:,idx].mean()
    s_avg = shuf_mean[:,idx].mean()
    s_sem = shuf_sem[:,idx].mean()
    m_avg = null_metrics[:, idx].mean()
    print(f"{name} – True: {t_avg:.4f} | Shuffled: {s_avg:.4f} ± {s_sem:.4f} Mean: {m_avg:.4f}")
    np.save(os.path.join(out,f"{dset}_{model_name}_{name}.npy"), true_m[:,idx])
    np.save(os.path.join(out,f"{dset}_shuffled_{name}.npy"), shuf_mean[:,idx])
    np.save(os.path.join(out,f"{dset}_mean_{name}.npy"), null_metrics[:,idx])

