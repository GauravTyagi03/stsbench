import os
import numpy as np
import random
import argparse
from PIL import Image
import PIL
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
import lpips
from itertools import product
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.stats import circmean

from utils import load_config
from glob import glob
from collections import defaultdict

import os
import cv2
import numpy as np
import random

from tqdm import tqdm 

def create_mp4_videos_from_images(directory_path):
    fps = 25 
    frame_count = 5

    # Collect all pred and true frames
    image_files = glob(os.path.join(directory_path, '*_*_*.png'))
    sequences = defaultdict(lambda: {'pred': [], 'true': []})

    for filepath in image_files:
        filename = os.path.basename(filepath)
        parts = filename.split('_')
        if len(parts) != 3:
            continue
        video_num, _, frame_type_ext = parts
        frame_type, _ = frame_type_ext.split('.')
        if frame_type not in ['pred', 'true']:
            continue
        try:
            frame_index = int(parts[1])
        except ValueError:
            continue
        sequences[video_num][frame_type].append((frame_index, filepath))

    for video_num, frames_dict in sequences.items():
        for kind in ['pred', 'true']:
            frames = sorted(frames_dict[kind], key=lambda x: x[0])
            if len(frames) != frame_count:
                continue  # Skip incomplete sequences

            # Read the first image to get frame size
            first_img = cv2.imread(frames[0][1])
            if first_img is None:
                continue
            height, width, _ = first_img.shape
            video_filename = os.path.join(directory_path, f"{video_num}_{kind}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))

            for _, img_path in frames:
                img = cv2.imread(img_path)
                if img is not None:
                    out.write(img)

            out.release()

def compute_optic_flow_mean(video_path):
    # computes optic flow for entire video, and returns entire flow field
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    flows = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flows.append(np.mean(flow, axis=(0, 1)))
        prev_gray = gray
    cap.release()
    
    return np.mean(flows, axis=0), np.sqrt(np.sum(np.square(np.mean(flows, axis=0)))), flows

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

eval_tf = transforms.ToTensor()

lpips_fn = lpips.LPIPS(net='alex').to(device)

# Args
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./configs/dorsal_stream_diffusion_video.yaml')
args = parser.parse_args()
config = load_config(args.config)
name = config['name']
model_name = config['model_name']

neuron_ablation = config.get("neuron_ablation", False)
baseline = config.get("baseline", False)
modality = config.get("modality", "image")

if neuron_ablation and baseline:
    image_dir = f"./logs/{name}/{model_name}_neuron_ablation/"
elif baseline:
    image_dir = f"./logs/{name}/{model_name}/"
    model_name = image_dir.rstrip('/').split('/')[-1]
else:
    image_dir = config['train_params']['output_dir']
    
print("Using image directory:", image_dir)
# creates mp4 videos
create_mp4_videos_from_images(image_dir)

# Test dataset size
if 'dorsal_stream' in image_dir:
    dset, n = 'dorsal', 40
elif 'ventral_stream' in image_dir:
    dset, n = 'ventral', 100
else:
    raise ValueError('unknown dataset')

gen_fns  = [f"{i}_{j}_pred.png" for (i, j) in product(range(n), range(5))]
true_fns = [f"{i}_{j}_true.png" for (i, j) in product(range(n), range(5))]

def load_pil(path):
    return Image.open(path).convert('RGB').resize((150, 150)) # warning - note this is manually specified, careful if you want to change!

# computes LPIPS AND PSNR
# lpips uses color
def compute_lpips(gen_pil, true_pil):
    assert isinstance(gen_pil, PIL.Image.Image), "gen_pil must be a PIL.Image"
    assert isinstance(true_pil, PIL.Image.Image), "true_pil must be a PIL.Image"

    g = eval_tf(gen_pil); t = eval_tf(true_pil) # convert PIL images to tensors 

    assert g.ndim == 3 and g.shape[0] == 3, f"gen image must be [3, H, W], got {g.shape}"
    assert t.ndim == 3 and t.shape[0] == 3, f"true image must be [3, H, W], got {t.shape}"
    assert (0 <= g).all() and (g <= 1).all(), "gen image values must be in [0, 1]"
    assert (0 <= t).all() and (t <= 1).all(), "true image values must be in [0, 1]"

    # see lpips library for docs on image range - https://github.com/richzhang/PerceptualSimilarity/tree/master
    lp = lpips_fn(2*(g.unsqueeze(0).to(device)-0.5),
                  2*(t.unsqueeze(0).to(device)-0.5)
    ).item()
    
    return lp

# psnr uses grayscale
def compute_psnr(gen_pil, true_pil):
    assert isinstance(gen_pil, PIL.Image.Image), "gen_pil must be a PIL.Image"
    assert isinstance(true_pil, PIL.Image.Image), "true_pil must be a PIL.Image"

    g = eval_tf(gen_pil); t = eval_tf(true_pil) # convert PIL images to tensors 

    #assert g.ndim == 3 and g.shape[0] == 1, f"gen image must be [3, H, W], got {g.shape}"
    #assert t.ndim == 3 and t.shape[0] == 1, f"true image must be [3, H, W], got {t.shape}"
    assert (0 <= g).all() and (g <= 1).all(), "gen image values must be in [0, 1]"
    assert (0 <= t).all() and (t <= 1).all(), "true image values must be in [0, 1]"

    # see skimage for docs on image data types and channel ordering - https://scikit-image.org/docs/0.25.x/user_guide/data_types.html
    g8 = (g.permute(1,2,0).numpy()*255).astype(np.uint8)
    t8 = (t.permute(1,2,0).numpy()*255).astype(np.uint8)
    ps = psnr(t8, g8, data_range=255)
    
    return ps

# load images
gen_imgs  = [load_pil(os.path.join(image_dir,f)) for f in gen_fns]
true_imgs = [load_pil(os.path.join(image_dir,f)) for f in true_fns]

# video metrics ---
# diffusion model 
# shuffle null model
# mean null model 

# vid metrics for diffusion model
cos_sim = []
true_ofs = []
cos_sim_each = []
true_ofs_framewise = []
for train_id in range(40): 
    of, mag, flows = compute_optic_flow_mean(os.path.join(image_dir, f"{train_id}_pred.mp4"))
    pred_of = of/mag

    # framewise version
    frame_ofs_pred = []
    for i in range(len(flows)):
        frame_of = flows[i]/np.sqrt(np.sum(np.square(flows[i])))
        frame_ofs_pred.append(frame_of)
        
    of, mag, flows = compute_optic_flow_mean(os.path.join(image_dir, f"{train_id}_true.mp4"))
    true_of = of/mag
    true_ofs.append(true_of)

    # framewise version
    frame_ofs_true = []
    for i in range(len(flows)):
        frame_of = flows[i]/np.sqrt(np.sum(np.square(flows[i])))
        frame_ofs_true.append(frame_of)
        true_ofs_framewise.append(frame_of)

    cos_sim.append(np.dot(pred_of, true_of)) # cosine similarity of true and predited optic flow

    for i in range(len(flows)):
        cos_sim_each.append(np.dot(frame_ofs_pred[i], frame_ofs_true[i]))
        
print("cosine sim real: " + str(np.mean(cos_sim)))
print("cosine sim real framewise: " + str(np.mean(cos_sim_each)))

# vide metrics for shuffle model --- 25 random shuffles
true_ofs = np.array(true_ofs)
cos_sim_circ_shuf = []
np.random.seed(42)
for i in range(25):
    for train_id in range(40):
        idx_val = np.random.randint(40)
        while idx_val == train_id:
            idx_val = np.random.randint(40)
        cos_sim_circ_shuf.append(np.dot(true_ofs[idx_val], true_ofs[train_id]))
print("cosine sim shuf: " + str(np.mean(cos_sim_circ_shuf)))

true_ofs_framewise = np.array(true_ofs_framewise)
cos_sim_circ_shuf_framewise = []
np.random.seed(42)
for i in range(25):
    for train_id in range(len(cos_sim_each)):
        idx_val = np.random.randint(len(cos_sim_each))
        while idx_val == train_id:
            idx_val = np.random.randint(len(cos_sim_each))
        cos_sim_circ_shuf.append(np.dot(true_ofs_framewise[idx_val], true_ofs_framewise[train_id]))
print("cosine sim shuf framewise: " + str(np.mean(cos_sim_circ_shuf)))

# video metrics for the mean null model --- 

cos_sim_circ_mean = []

print("starting train set optic flow calculation with 1000 videso to get an estimate of mean...")
train_ofs = []
train_ofs_framewise = []
dorsal_stream_paths = glob(os.path.join("../dataset/dorsal_stream/", '*.mp4'))
video_ids = np.random.choice(len(dorsal_stream_paths), 500, replace=False)
dorsal_stream_paths = np.array(dorsal_stream_paths)
for filepath in tqdm(dorsal_stream_paths[video_ids]):
    of, mag, flows = compute_optic_flow_mean(filepath)
    norm_of = of / mag
    if not np.isnan(norm_of).any() and not np.isinf(norm_of).any():
        train_ofs.append(norm_of)

    for i in range(len(flows)):
        frame_of = flows[i]/np.sqrt(np.sum(np.square(flows[i])))
        if not np.isnan(frame_of).any() and not np.isinf(frame_of).any():
            train_ofs_framewise.append(frame_of)
        
train_ofs = np.array(train_ofs)
train_ofs_framewise = np.array(train_ofs_framewise)

# compute circular mean of our train set 
cm_value = (circmean((np.arctan2(train_ofs[:, 0], train_ofs[:, 1]))))
mean_ofs = np.array([np.sin(cm_value), np.cos(cm_value)])

cm_value_framewise = (circmean((np.arctan2(train_ofs_framewise[:, 0], train_ofs_framewise[:, 1]))))
mean_ofs_framewise = np.array([np.sin(cm_value_framewise), np.cos(cm_value_framewise)])

for train_id in range(40):
    cos_sim_circ_mean.append(np.dot(mean_ofs, true_ofs[train_id]))
    
print("cosine sim mean: " + str(np.mean(cos_sim_circ_mean)))

cos_sim_circ_mean = []
for train_id in range(len(true_ofs_framewise)):
    cos_sim_circ_mean.append(np.dot(mean_ofs_framewise, true_ofs_framewise[train_id]))
    
print("cosine sim mean framewise: " + str(np.mean(cos_sim_circ_mean)))

# now we compute the image metrics which are just averaged across frames 

null_metrics = []
null_metrics_gray = []
null_img = Image.new('RGB', (150, 150), (128, 128, 128)) # this is our 'mean' control

# compute metrics - our shuffles are bootstrapped over 5 replicates, this is seeded for replicability
true_m, shuf_m = [], []
random.seed(42)
k=5
for i in range(len(gen_imgs)):
    """
    true_m.append((compute_psnr(gen_imgs[i].convert('L'), true_imgs[i].convert('L')), compute_lpips(gen_imgs[i], true_imgs[i])))
    cands = [j for j in range(n) if j!=i]
    picks = random.sample(cands, k)
    sm = [(compute_psnr(true_imgs[i].convert('L'), true_imgs[j].convert('L')), compute_lpips(true_imgs[i], true_imgs[j])) for j in picks]
    shuf_m.append(sm)
    null_metrics.append((compute_psnr(true_imgs[i].convert('L'), null_img.convert('L')), compute_lpips(true_imgs[i], null_img)))
    """
    true_m.append((compute_psnr(gen_imgs[i], true_imgs[i]), compute_lpips(gen_imgs[i], true_imgs[i])))
    cands = [j for j in range(n) if j!=i]
    picks = random.sample(cands, k)
    sm = [(compute_psnr(true_imgs[i], true_imgs[j]), compute_lpips(true_imgs[i], true_imgs[j])) for j in picks]
    shuf_m.append(sm)
    null_metrics.append((compute_psnr(true_imgs[i], null_img), compute_lpips(true_imgs[i], null_img)))

true_m = np.array(true_m)         # (n,2)
shuf_m = np.array(shuf_m)         # (n,k,2)
null_metrics = np.array(null_metrics)

shuf_mean = shuf_m.mean(axis=1)   # (n,2)
shuf_sem  = shuf_m.std(axis=1, ddof=1)/np.sqrt(n-1)

# log and print eval results 
out = "./logs/revisions/"; os.makedirs(out, exist_ok=True)
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

