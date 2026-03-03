# Differentiable Sliced Wasserstein Losses for NCA Training

## Summary

The Sliced Wasserstein Distance (SWD) has emerged as a theoretically grounded, practical replacement for Gram-matrix losses in neural texture synthesis — and by extension, Neural Cellular Automata (NCA) training. This survey covers the mathematical foundations, available PyTorch/JAX implementations, recent variants (2024–2026), and integration strategies for NCA pipelines. The SWD captures complete feature distributions rather than just second-order statistics, leading to measurably better texture quality, and can be computed with a simple sort-and-compare algorithm that is fully differentiable.

## Key Findings

1. **SWD is the emerging standard for NCA texture loss.** Multiple recent NCA papers (μNCA, Multi-Texture NCA, MeshNCA, DyNCA, Cells2Pixels) have adopted optimal-transport-based losses over Gram matrices, with the Sliced Wasserstein variant being the most common choice due to its simplicity and effectiveness.

2. **Implementation is trivially simple.** The core algorithm is ~10 lines of PyTorch: project features onto random unit vectors, sort projections, compute L2 distance between sorted values. It requires no iterative solvers, no regularization hyperparameters, and is fully differentiable through autograd.

3. **POT is the go-to library** for sliced Wasserstein in both PyTorch and JAX, with `ot.sliced_wasserstein_distance()` supporting multiple backends. For JAX-native work, OTT-JAX provides broader OT tooling but lacks a dedicated SWD function.

4. **The variant landscape has exploded (2024–2026)** — Max-Sliced, Distributional, Generalized, Energy-Based, Tree-Sliced, Streaming, and Spherical variants each address specific limitations of vanilla SWD.

5. **32 random projections suffice** for NCA texture training per the Multi-Texture NCA paper, though variance decreases with more projections (POT examples test 1–1000).

## Deep Dive

### Mathematical Formulation

The Sliced Wasserstein Distance between two d-dimensional distributions μ and ν is defined as:

```
SW_p(μ, ν) = ( ∫_{S^{d-1}} W_p^p(θ#μ, θ#ν) dσ(θ) )^{1/p}
```

where:
- S^{d-1} is the unit sphere in R^d
- θ#μ is the pushforward (projection) of μ onto direction θ
- W_p is the 1D Wasserstein-p distance
- σ is the uniform measure on the sphere

The 1D Wasserstein distance has a closed-form solution via sorting: for empirical distributions with n samples each, W_p^p = (1/n) Σ|x_sorted[i] - y_sorted[i]|^p.

In practice, the integral over the sphere is approximated with L random projections:

```
SW_p(μ, ν) ≈ ( (1/L) Σ_{i=1}^{L} W_p^p(θ_i#μ, θ_i#ν) )^{1/p}
```

### Algorithm (Pseudocode)

```python
def sliced_wasserstein_loss(features_gen, features_target, n_projections=32, p=2):
    """
    features_gen, features_target: [N, C] tensors (N spatial positions, C channels)
    """
    d = features_gen.shape[1]
    # 1. Generate random unit vectors
    theta = torch.randn(n_projections, d)
    theta = theta / theta.norm(dim=1, keepdim=True)

    # 2. Project features onto random directions
    proj_gen = features_gen @ theta.T      # [N, L]
    proj_target = features_target @ theta.T # [N, L]

    # 3. Sort along spatial dimension
    proj_gen_sorted = torch.sort(proj_gen, dim=0)[0]
    proj_target_sorted = torch.sort(proj_target, dim=0)[0]

    # 4. Compute L2 (Wasserstein-2) distance
    loss = ((proj_gen_sorted - proj_target_sorted) ** p).mean()
    return loss
```

### Why SWD Beats Gram-Matrix Loss

The Gram-matrix loss (Gatys et al., 2015) computes G = F^T F / N, capturing only second-order statistics (feature correlations). The SWD captures the full distribution, which means:

| Property | Gram Loss | Sliced Wasserstein |
|----------|-----------|-------------------|
| Captures mean | ✗ (centered) | ✓ |
| Captures variance | ✓ (diagonal) | ✓ |
| Captures higher moments | ✗ | ✓ |
| Captures full distribution shape | ✗ | ✓ |
| True metric (triangle inequality) | ✗ | ✓ |
| Handles multimodal distributions | Poorly | Well |
| Theoretical guarantees | Weak | Strong (metrizes weak convergence) |

### VGG Feature Extraction for NCA Training

The standard configuration (Heitz et al., CVPR 2021; Catrina et al., 2025) uses:

- **Network**: VGG-16 or VGG-19, pretrained on ImageNet
- **Layers**: conv1_1, conv2_1, conv3_1, conv4_1, conv5_1 (first conv of each block)
- **Modifications**:
  - Max pooling → Average pooling
  - Zero padding → Reflect padding
  - Inputs normalized with ImageNet statistics
  - Activations are scaled per layer
- **Loss**: Sum of SWD across all selected layers

```
L_total(I, I_target) = Σ_{l ∈ layers} SWD(F_l(I), F_l(I_target))
```

### Available Libraries

#### POT (Python Optimal Transport) — Best for SWD
- **Install**: `pip install pot[backend-torch]` or `pip install pot[backend-jax]`
- **API**: `ot.sliced_wasserstein_distance(xs, xt, a=None, b=None, n_projections=50, seed=None)`
- **Backends**: NumPy, PyTorch, JAX (limited), CuPy, TensorFlow
- **Also provides**: Max-Sliced, Spherical Sliced (2024-2025), Circular OT
- **Caveat**: JAX backend is slow; OTT-JAX recommended for JAX OT work

#### GeomLoss — Best for Large-Scale Sinkhorn
- **Install**: `pip install geomloss`
- **Focus**: Sinkhorn divergences (entropic regularization of Wasserstein)
- **Strengths**: Linear memory via KeOps, GPU-optimized
- **Limitation**: No direct SWD — focuses on Sinkhorn approximations
- **PyTorch only**

#### OTT-JAX — Best for JAX-Native OT
- **Install**: `pip install ott-jax`
- **Focus**: Sinkhorn, Gromov-Wasserstein, neural OT (flow matching, ICNN maps)
- **Limitation**: No dedicated SWD implementation
- **Strengths**: JIT compilation, VMAP, implicit differentiation

#### Direct Implementation (Recommended for NCA)
For NCA training, the SWD is so simple that direct implementation is often preferred over library dependencies:

```python
# Complete NCA texture loss in ~20 lines
import torch
import torchvision.models as models

vgg = models.vgg19(pretrained=True).features.eval()
layers = [1, 6, 11, 20, 29]  # conv1_1 through conv5_1 (after ReLU)

def extract_features(x):
    features = []
    for i, layer in enumerate(vgg):
        x = layer(x)
        if i in layers:
            features.append(x)
    return features

def sw_loss(gen_features, target_features, n_proj=32):
    loss = 0
    for gf, tf in zip(gen_features, target_features):
        B, C, H, W = gf.shape
        gf_flat = gf.reshape(B, C, -1).permute(0, 2, 1)  # [B, HW, C]
        tf_flat = tf.reshape(B, C, -1).permute(0, 2, 1)

        theta = torch.randn(n_proj, C, device=gf.device)
        theta = theta / theta.norm(dim=1, keepdim=True)

        proj_g = (gf_flat @ theta.T).sort(dim=1)[0]
        proj_t = (tf_flat @ theta.T).sort(dim=1)[0]
        loss += ((proj_g - proj_t) ** 2).mean()
    return loss
```

### SWD Variants (2024–2026)

| Variant | Key Idea | Use Case |
|---------|----------|----------|
| **Max-Sliced** (Deshpande et al.) | Only the most informative projection direction | When one direction dominates |
| **Distributional** (Nguyen et al., ICLR 2021) | Optimal distribution over projections | Better than random when d is large |
| **Generalized** (Kolouri et al.) | Non-linear projections via generalized Radon transform | Capturing non-linear structure |
| **Energy-Based** (Nguyen et al.) | Neural energy function to weight projections | Data-adaptive discrimination |
| **Kernel Max-Sliced** (2024) | Optimal nonlinear 1D mapping | NP-hard but strong theoretically |
| **Streaming** (Nguyen, 2025) | Quantile sketches for sequential data | Online/streaming settings |
| **Tree-Sliced** (Tran et al., 2025) | Tree metric spaces replace 1D projections | Topological/hierarchical data |
| **Spherical Sliced** (Liu et al., ICLR 2025) | For data on spheres | Directional/rotation data |

### Integration Strategy for NCA Training

#### Standard Training Pipeline
1. **Forward**: Run NCA for 32–96 steps (stochastic) to generate texture
2. **Extract**: Pass both generated and target through VGG, collect features at 5 layers
3. **Compute**: SWD at each layer with 32 random projections, sum losses
4. **Backward**: Backpropagate through NCA steps (gradient normalization recommended)
5. **Extras**: Add overflow loss to keep cell states in [-1, 1]

#### Training Configuration (from Multi-Texture NCA)
- Pool size: 1024 states
- Batch size: 8
- Evolution steps: 32–96 (random per iteration)
- ~50% stochastic cell updates per step
- 5,000 epochs (single texture), 10,000 (multi-texture)
- Adam optimizer with gradient normalization after each CA step

#### Comparison with Alternative OT Losses for NCA
- **Gram Loss**: Fastest but lowest quality; misses distribution shape
- **SWD (Heitz et al.)**: Best quality-cost tradeoff; 32 projections sufficient
- **Relaxed OT / Sinkhorn (Kolkin et al.)**: Used by Cells2Pixels (2026); iterative solver, more expensive but handles moment matching
- **OTT-Loss (μNCA)**: Sinkhorn-based on patch distributions; avoids VGG entirely, works with 68-param models but different quality profile

#### Real-Time Deployment After Training
Once trained with SWD loss, the NCA itself is tiny (hundreds to thousands of parameters) and runs as a simple compute shader:
- **WebGL**: Used by Self-Organising Textures, MeshNCA, Cells2Pixels demos
- **WebGPU**: Neural Automata Playground demonstrates WebGPU compute shaders for NCA
- **GLSL/WGSL**: μNCA models (68–588 bytes) can be implemented in a few lines of shader code

The SWD loss is only needed during training, not at inference time.

## Connections

- **Relates to prior research on NCA layer ablation**: Layer selection for VGG features matters — conv3_1 and conv4_1 contribute most to perceptual quality (see rq-1771829371565-nca-layer-ablation)
- **Relates to perceptual loss distillation**: SWD could potentially be computed on distilled lightweight networks instead of VGG, reducing training cost (see rq-1771851237047-perceptual-loss-distillation)
- **Relates to hybrid loss scheduling**: Start with Gram loss for fast exploration, switch to SWD for final quality refinement (see rq-1771851237044-hybrid-loss-schedule)
- **Theoretical foundation**: SGD convergence for SW losses has been proven (Tanguy et al., 2023), providing theoretical backing that Gram loss lacks

## Follow-up Questions

1. **Distributional SWD for NCA**: Could learning optimal projection distributions (DSW) improve NCA texture quality over random projections, especially for high-dimensional feature spaces?
2. **SWD on distilled features**: Can SWD computed on a 3-layer distilled perceptual network match SWD on VGG-19 for NCA training quality?
3. **Adaptive projection count scheduling**: Start with few projections (fast, noisy) and increase during training — does this improve convergence speed?
4. **Tree-Sliced Wasserstein for hierarchical textures**: Could TSW's tree-structured projections better capture multi-scale texture structure than standard linear projections?

## Sources

- Heitz, E., Vanhoey, K., Chambon, T., & Belcour, L. (2021). A Sliced Wasserstein Loss for Neural Texture Synthesis. CVPR 2021. https://arxiv.org/abs/2006.07229
- Catrina, S. et al. (2025). Multi-Texture Synthesis through Signal Responsive Neural Cellular Automata. Scientific Reports. https://www.nature.com/articles/s41598-025-23997-7
- Mordvintsev, A. & Niklasson, E. (2021). μNCA: Texture Generation with Ultra-Compact Neural Cellular Automata. https://arxiv.org/abs/2111.13545
- Pajouheshgar, E. et al. (2026). Neural Cellular Automata: From Cells to Pixels. https://arxiv.org/abs/2506.22899
- Tanguy, E. (2023). Convergence of SGD for Training Neural Networks with Sliced Wasserstein Losses. https://arxiv.org/abs/2307.11714
- Nguyen, K. et al. (2021). Distributional Sliced-Wasserstein and Applications to Generative Modeling. ICLR 2021. https://arxiv.org/abs/2002.07367
- Kolouri, S. et al. (2019). Generalized Sliced Wasserstein Distances. https://arxiv.org/abs/1902.00434
- Li, P. et al. (2022). Long Range Constraints for Neural Texture Synthesis Using Sliced Wasserstein Loss. https://arxiv.org/abs/2211.11137
- POT (Python Optimal Transport) Library. https://pythonot.github.io/
- OTT-JAX: Optimal Transport Tools for JAX. https://ott-jax.readthedocs.io/
- GeomLoss Library. https://www.kernel-operations.io/geomloss/
- Official SWL implementation: https://github.com/tchambon/A-Sliced-Wasserstein-Loss-for-Neural-Texture-Synthesis
- Unofficial PyTorch SWL: https://github.com/xchhuang/pytorch_sliced_wasserstein_loss
- Energy-Based SWD: https://github.com/khainb/EBSW
- Distributional SWD: https://github.com/VinAIResearch/DSW
