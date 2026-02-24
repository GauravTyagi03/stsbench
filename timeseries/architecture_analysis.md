# Temporal Binning Architecture Analysis

## Background

The current model takes time-averaged neural data as a single vector per trial. The proposed change replaces that scalar-per-neuron summary with a sequence of time-binned neural states, letting the U-Net's cross-attention mechanism attend over temporal structure.

---

## Why the Architecture Change Makes Sense

### The current model throws away temporal information

Each neuron's activity is collapsed to a single number (the trial-average), regardless of *when* during the trial that neuron fired. Visual cortex has a well-known temporal response profile: a latency period of ~50ms, a sharp onset peak, a sustained period, and adaptation. Two different stimuli can produce nearly identical trial-averaged responses while having very different temporal dynamics. The current conditioning vector is blind to this.

### Cross-attention is already designed for variable-length sequences

The U-Net's cross-attention layers — in `DownBlock`, `MidBlock`, and `UpBlockUnet` — accept a context tensor of shape `(B, T, context_dim)` and produce queries from the spatial image features, attending to all T tokens. With T=1, each spatial patch attends to a single, flat brain state. With T=15-20, each patch can dynamically weight different temporal windows of neural activity. This is exactly the mechanism that lets text-conditioned diffusion models pick out relevant words for each image region — here, we're letting each image patch pick out relevant *temporal epochs* of neural activity.

Critically, the cross-attention blocks in `blocks.py` already handle arbitrary T without modification. The assertion on line 138 only checks `context.shape[-1] == self.context_dim`, not the sequence length. `nn.Linear` and `nn.MultiheadAttention` both operate over the last dimension, so the sequence axis passes through transparently.

### The existing bottleneck already handles temporal context correctly

The optional `bottleneck_layer` in `unet_cond_base.py` is an `nn.Linear(neural_embed_dim, bottleneck_dim)` applied to `context_hidden_states`. Since `nn.Linear` broadcasts over batch and sequence dimensions, it maps `(B, 1, N) → (B, 1, D)` today and will map `(B, T, N) → (B, T, D)` after the change — no modification needed.

### Positional encoding is necessary

`nn.MultiheadAttention` is permutation-invariant: swapping the order of the T context tokens produces the same output. Without positional encoding, the model cannot distinguish bin 3 from bin 17. Adding sinusoidal positional encoding to the T dimension gives the model a free and well-studied mechanism to infer temporal order, consistent with standard transformer practice.

---

## Issues and Modifications Worth Considering

### 1. Signal-to-noise per bin degrades as T increases

Trial-averaging suppresses noise. Within a single time bin of width `w` ms, each electrode has fewer total measurements to average over. If `w` is too small (many bins), each bin is noisy; too large (few bins), and you lose temporal resolution. The 15–20 bin range over a ~600ms response window corresponds to ~30-40ms per bin, which aligns well with known neural response timescales and keeps per-bin SNR reasonable.

**Recommendation:** Start at T=15–20. Expose `num_bins` as a config hyperparameter and sweep it on a validation metric. Non-uniform binning (finer resolution in the 50–200ms response window, coarser elsewhere) is a further refinement worth trying.

### 2. Whether to include pre-stimulus (baseline) bins

The normalization pipeline includes ~100ms of pre-stimulus baseline. Including 2–3 baseline bins could help the model distinguish driven response from background noise, since the cross-attention will learn that pre-stimulus bins carry little image-relevant information. Alternatively, these can be dropped to reduce T.

### 3. Positional encoding placement

The plan calls for adding 1D sinusoidal PE over the T dimension. This should be added *after* projecting N → d_model (i.e., project to the embedding space first, then add PE). Adding PE directly to the raw neural dimension N (~1000-2244) is less principled because the neural feature space has no semantic relationship to standard embedding distances that PE assumes.

Concretely: `(B, T, N) → Linear → (B, T, d_model) → +PE → (B, T, d_model)`.

### 4. Lightweight cross-bin interaction (optional improvement)

After adding PE, a single Transformer encoder layer over the T dimension would allow bins to interact — e.g., modeling that "bin 5 *and* bin 12 are both active" is informative. This is optional but could improve expressiveness. One layer is enough; deeper stacks are likely overkill for T=15-20 and risk overfitting.

### 5. Classifier-free guidance dropout must drop the full sequence

The current training loop zeros out `neural_condition` with probability 0.1 for CFG. This logic must zero out all T bins together (not independently), so the null conditioning is a full zero tensor of shape `(B, T, N)` rather than `(B, 1, N)`. The existing `empty_neural_embed` logic needs its shape updated accordingly.

### 6. Memory and compute cost

With T=1 today, cross-attention cost is negligible. At T=20, each cross-attention operation has 20× more key-value pairs. Given the image spatial dimension is `H×W` (e.g., 16×16=256 at the deepest U-Net level), the attention matrix is `(B, 256, 20)` — trivially small. Compute increase is minimal.

### 7. Projection dimension choice

If `neural_embed_dim` stays as raw N (~2244), each block's `context_proj` must initialize with `nn.Linear(2244, out_channels)` — fine but redundant since every block separately learns the same neural-to-channel projection. Projecting to a shared `context_dim` (e.g., 256 or 512) upstream in `TemporalNeuralConditioner` is more parameter-efficient and lets the downstream cross-attention blocks receive a richer, already-mixed representation. If using the bottleneck, `bottleneck_dim` serves this purpose and no separate upstream projection is needed.

---

## Intuition for Whether This Will Be Effective

### Arguments for effectiveness

Visual cortex encodes stimuli across time, not just in aggregate. Early bins (~50–100ms post-stimulus) reflect low-level feature detection; later bins (~150–300ms) reflect higher-level object recognition and feedback. Pooling across these epochs erases category-discriminating temporal patterns. A model that can attend selectively to the "object recognition window" for semantic guidance and the "feature detection window" for texture/orientation should reconstruct more faithfully.

The cross-attention mechanism is a good fit because different spatial regions of an image likely correspond to different temporal neural patterns. A small patch corresponding to a face might correlate with sustained neural activity in face-selective neurons; a background region might correlate only with early, transient responses. Cross-attention lets the model discover these correspondences.

### Arguments for limited gains

If the downstream task (reconstruction quality as measured by perceptual metrics) is already near-ceiling with the time-averaged model, the extra temporal structure may not help further. The quality of normalized time-series data matters: if bin-wise normalization introduced artifacts or the SNR per bin is too low, the temporal information could add more noise than signal. The model may learn to mostly ignore the temporal structure and approximate the time-average behavior anyway.

**Net assessment:** Likely helpful, especially for stimuli that differ mainly in category (faces vs. objects vs. scenes) where temporal response dynamics are most discriminative. The architecture is already set up to exploit this information without fundamental changes; the main risk is empirical (low per-bin SNR) rather than architectural.

---

## Data Shape Flow

### Current (T = 1, time-averaged)

```
Raw data:            (n_timepoints, n_electrodes, n_trials)   [in HDF5]
Time-average:        (n_trials, N)                            [in dataloader]
Batch:               (B, N)
Unsqueeze:           (B, 1, N)                                [train_ddpm_cond.py:97]
Bottleneck (opt):    (B, 1, N) → (B, 1, D)                   [unet_cond_base.py:118]
context_proj/block:  (B, 1, D) → (B, 1, C_layer)            [blocks.py:139]
Cross-attn KV:       (B, 1, C_layer)
Cross-attn Q:        (B, H*W, C_layer)                        [spatial features]
Cross-attn output:   (B, H*W, C_layer) → reshape → (B, C_layer, H, W)
```

### Proposed (T bins, time-series)

```
Normalized data:     (n_timepoints, n_electrodes, n_trials)   [in HDF5]
Bin & select:        (n_trials, T, N)                         [updated dataloader]
Batch:               (B, T, N)
TemporalNeuralConditioner:
  └─ Linear (N→d):   (B, T, N) → (B, T, d_model)
  └─ + sine/cos PE:  (B, T, d_model)   [PE added on T axis]
  └─ (opt) Transformer encoder over T: (B, T, d_model)
Bottleneck (opt):    (B, T, d_model) → (B, T, D)              [same linear, no change needed]
context_proj/block:  (B, T, D) → (B, T, C_layer)             [same linear, no change needed]
Cross-attn KV:       (B, T, C_layer)                          [T tokens instead of 1]
Cross-attn Q:        (B, H*W, C_layer)                        [spatial features, unchanged]
Cross-attn output:   (B, H*W, C_layer) → reshape → (B, C_layer, H, W)
```

The U-Net blocks see a larger key-value sequence but are otherwise identical. The entire modification is contained to: (1) the data loading pipeline, (2) a new `TemporalNeuralConditioner` module applied before the U-Net, and (3) minor adjustments to the conditioning setup in the training and sampling scripts.
