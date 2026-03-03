# NCA-ConvNeXt Hybrid Denoiser: Could ConvNeXt Inverted Bottleneck Blocks Serve as NCA Update Rules?

**Research ID:** rq-1740621600000-nca-convnext-hybrid
**Completed:** 2026-03-01
**Priority:** 7
**Tags:** nca, convnext, architecture, diffusion-models, hybrid-design

---

## Summary

ConvNeXt's inverted bottleneck block is a remarkably natural fit for NCA update rules, sharing the same fundamental structure: local spatial perception (depthwise conv) → channel expansion → nonlinear transformation → channel compression → residual update. Replacing the traditional NCA update rule (fixed Sobel/Laplacian perception + small MLP) with a modernized ConvNeXt-style block could yield significant improvements in expressiveness, training stability, and gradient flow — particularly for diffusion-based NCA denoisers like Diff-NCA. Key advantages include: 7×7 depthwise perception (vs. 3×3), GELU activation (smoother gradients), Layer Normalization (contractive dynamics for long-horizon stability), and Global Response Normalization (preventing feature collapse in self-supervised settings). The design is novel — no published work directly combines ConvNeXt blocks with NCA architectures — but all architectural components are well-understood, making this a high-feasibility research direction.

---

## Key Findings

### 1. Architectural Alignment: ConvNeXt Block ≈ NCA Update Rule

The standard NCA update rule (Mordvintsev et al., 2020) and the ConvNeXt block share a strikingly similar computational structure:

| Component | Traditional NCA | ConvNeXt Block | Alignment |
|-----------|----------------|----------------|-----------|
| **Perception** | 3×3 depthwise conv (fixed Sobel/Laplacian kernels) | 7×7 depthwise conv (learned kernels) | Direct analog — both gather local spatial info |
| **Channel mixing** | 1×1 conv → hidden dim | 1×1 conv → 4× expanded dim | Same operation, different scale |
| **Nonlinearity** | ReLU | GELU (single, between 1×1 convs) | Functional equivalent |
| **Projection** | 1×1 conv → state dim | 1×1 conv → original dim | Identical |
| **State update** | Residual: s ← s + Δs | Residual: x ← x + f(x) | Identical |
| **Normalization** | None / gradient norm | LayerNorm (pre-block) | ConvNeXt adds stability |

The mapping is almost one-to-one. A ConvNeXt block applied per-cell is structurally equivalent to an NCA update rule with:
- Learned rather than fixed perception kernels
- Wider local neighborhood (7×7 vs 3×3)
- More expressive channel mixing via inverted bottleneck
- Better-conditioned training dynamics via normalization

### 2. The Inverted Bottleneck as Enhanced State Transformation

In standard NCAs, the update rule typically uses a small MLP with ~8,000 parameters that maps a 48-dimensional perception vector (16 channels × 3 filter types) to a 16-dimensional state update. This is a narrow → narrow architecture.

The ConvNeXt inverted bottleneck uses a **narrow → wide → narrow** pattern:
- Input: C channels (e.g., 96)
- Expansion: 4C channels (e.g., 384) via 1×1 conv
- GELU nonlinearity
- Compression: C channels via 1×1 conv

This is significant for NCAs because:
- **Richer intermediate representations**: The 4× expansion creates a high-dimensional space where complex state transformations can be learned, before compressing back to the cell state dimension
- **Controlled expressiveness**: Unlike simply making the MLP wider (which increases parameter count quadratically), the inverted bottleneck provides expressiveness efficiently — the depthwise conv is O(C × k²) and the expansion is O(C × 4C), both linear in C
- **Analogous to biological signaling**: Cells in biological systems often amplify signals internally before making discrete state decisions — the inverted bottleneck mirrors this pattern

### 3. 7×7 Depthwise Perception: Broader Context Without Global Communication

A fundamental limitation of NCAs is local information propagation — each cell can only "see" its immediate neighbors, requiring O(N) steps to communicate across an N×N grid. ConvNeXt's 7×7 depthwise convolution directly addresses this:

- **3×3 kernel** (standard NCA): 9 cells per perception step, receptive field grows by 2 per step
- **7×7 kernel** (ConvNeXt-style): 49 cells per perception step, receptive field grows by 6 per step

This means a ConvNeXt-NCA would need **~3× fewer steps** to achieve the same effective receptive field as a standard NCA, directly reducing inference time. For the Diff-NCA architecture (which uses 20 steps with 3×3 perception), a 7×7 kernel could potentially achieve similar results in ~7 steps.

However, there's a tradeoff: 7×7 depthwise convolutions have been identified as a throughput bottleneck due to memory access patterns (InceptionNeXt, CVPR 2024). For NCA applications where step count matters more than per-step throughput, this is likely a favorable tradeoff.

### 4. Normalization: The Key to Long-Horizon Stability

Training instability is a well-documented challenge in NCAs. The original Growing NCA work observed "sudden jumps of the loss value in the later stages of training." Several normalization strategies from ConvNeXt are directly applicable:

**Layer Normalization (LayerNorm)**:
- Already proven effective in NCA contexts — the ViTCA paper (NeurIPS 2022) showed that LayerNorm creates contractive dynamics conducive to fixed-point convergence
- ConvNeXt uses a single LayerNorm before each block, which in NCA terms normalizes the cell state before the update rule runs
- This stabilizes the iterative dynamics over long rollout horizons

**Global Response Normalization (GRN, from ConvNeXt V2)**:
- Prevents feature collapse — dead or saturated channels in the cell state vector
- Enhances inter-channel competition via L2-norm pooling and divisive normalization
- Particularly relevant for NCA because:
  - NCAs are known to develop redundant hidden channels during training
  - Feature collapse in cell states means wasted representational capacity
  - GRN's channel competition could lead to more diverse, information-rich cell states

**Group Normalization**:
- Already used in Diff-NCA (hidden depth of 512 with group norm)
- Complementary to LayerNorm — normalizes across channel groups rather than all channels

### 5. GELU Activation: Smoother Dynamics for Continuous State Spaces

NCA state vectors are continuous — each channel holds a real number representing chemical concentration, signaling potential, or learned hidden features. GELU offers advantages over ReLU in this context:

- **No dying neurons**: ReLU can permanently zero out channels in the state vector, effectively reducing the cell's representational capacity. GELU preserves gradient flow for all values
- **Smooth gradients**: GELU's smooth transition avoids the discontinuous gradient at zero that can cause oscillations in iterative NCA dynamics
- **Non-monotonicity**: GELU's slight dip below zero for small negative values adds expressiveness — the update rule can learn more nuanced state transitions
- **Sparse single usage**: ConvNeXt uses only one GELU per block (between the 1×1 convs). Minimal activation functions reduce the risk of gradient vanishing over many NCA steps

### 6. Concrete Architecture Proposal: ConvNeXt-NCA

Based on the analysis, a ConvNeXt-NCA update rule would look like:

```
Input: cell state s ∈ R^C (e.g., C=96)

1. LayerNorm(s)                           # Normalize state
2. DWConv7x7(s) → z ∈ R^C                # Perceive 7×7 neighborhood (depthwise, learned)
3. Conv1x1(z) → h ∈ R^{4C}               # Expand to hidden dim (384)
4. GELU(h)                                # Nonlinearity
5. GRN(h)                                 # Global Response Normalization (optional)
6. Conv1x1(h) → Δs ∈ R^C                 # Compress back to state dim
7. Stochastic mask m ~ Bernoulli(p=0.5)   # Fire rate (standard NCA)
8. s ← s + m ⊙ Δs                        # Residual state update
```

**Parameter count estimate** (C=96):
- DWConv 7×7: 96 × 49 = 4,704
- Conv1x1 (expand): 96 × 384 = 36,864
- Conv1x1 (compress): 384 × 96 = 36,864
- LayerNorm: 192 (scale + bias)
- GRN: 768 (γ + β for 384 channels)
- **Total: ~79,400 parameters per update rule**

For comparison:
- Original Growing NCA: ~8,000 parameters
- Diff-NCA: ~208,000 total (including multiple components)
- FourierDiff-NCA: ~887,000 total

The ConvNeXt-NCA update rule at 79K params is ~10× larger than the original NCA but much smaller than full Diff-NCA, while offering substantially richer dynamics.

### 7. Advantages for Diffusion Denoising Specifically

When used as a denoiser in a diffusion framework (replacing U-Net), the ConvNeXt-NCA offers:

**vs. U-Net**:
- **Parameter efficiency**: 79K per update rule vs. millions in U-Net
- **Scale invariance**: Same model works at any resolution (NCA property)
- **Parallelism**: Every cell updates simultaneously (vs. U-Net's sequential encoder-decoder)
- **Inpainting/super-res for free**: NCA's local nature handles variable-size inputs

**vs. Standard Diff-NCA**:
- **Faster convergence**: Wider receptive field (7×7) means fewer steps needed
- **Better gradient flow**: GELU + LayerNorm prevent training instability
- **Richer dynamics**: Inverted bottleneck provides more expressive per-step transformations
- **Feature diversity**: GRN prevents channel collapse in the hidden state

**vs. FourierDiff-NCA**:
- **Simpler architecture**: No need for separate Fourier-domain processing
- **Fully local**: No global operations (Fourier transform is global by nature)
- **Lower overhead**: Avoids FFT/IFFT computation

### 8. Challenges and Open Questions

**Throughput vs. step count tradeoff**: 7×7 depthwise convolutions are slower per step than 3×3. Does the reduced step count compensate? InceptionNeXt's parallel branch decomposition (splitting 7×7 into smaller parallel kernels) could help.

**GRN in iterative settings**: GRN was designed for single-pass networks. In an iterative NCA, the global L2-norm statistics change at every step. Does this help (dynamic normalization) or hurt (unstable statistics)?

**Optimal channel count**: Standard NCAs use 16 channels; ConvNeXt models start at 96. What's the sweet spot for NCA-ConvNeXt hybrid? The inverted bottleneck's 4× expansion means even 32 base channels would expand to 128 in the hidden layer.

**Conditioning mechanism**: For diffusion denoising, the model needs timestep conditioning. Diff-NCA uses sinusoidal positional encoding. How should this integrate with the ConvNeXt block — via FiLM conditioning? AdaLN (Adaptive Layer Normalization, as in DiT)?

**Stochastic fire rate interaction**: NCAs use stochastic cell updates (50% fire rate). Does this interact well with LayerNorm's statistics, which assume all positions are active?

**Training cost**: The 10× increase in parameters per update rule, combined with 7×7 convolutions, may significantly increase training time even if fewer steps are needed.

---

## Deep Dive: Why This Hasn't Been Done Yet

Despite the natural alignment, no published work combines ConvNeXt blocks with NCA architectures. Likely reasons:

1. **Community separation**: NCA research (Google Brain / self-organizing systems) and ConvNeXt research (Meta AI) have different communities with little overlap
2. **NCA minimalism culture**: The NCA community values tiny models (~8K params) as a feature, not a limitation. A 79K-param update rule feels "bloated" by NCA standards
3. **Recency**: ConvNeXt V2 (Jan 2023) and Diff-NCA (Jan 2024) are both recent. The specific combination hasn't had time to emerge
4. **Different priorities**: NCA research focuses on emergence and self-organization; ConvNeXt research focuses on classification performance

This gap represents a clear opportunity for novel contribution.

---

## Connections to Prior Research

- **NCA-Diffusion Hybrid (rq-1772162368871)**: Established that NCAs can replace U-Nets in diffusion. ConvNeXt-NCA would be a modernized update rule for these hybrids
- **Minimal LPIPS Proxy (rq-1771762675169)**: If perceptual loss can be made lightweight, training ConvNeXt-NCAs becomes more practical
- **Architecture-Agnostic Distillation (rq-1740621600000)**: ConvNeXt-NCA could be a distillation target — distill a U-Net diffusion model into a ConvNeXt-NCA denoiser
- **Layer Ablation Studies**: The inverted bottleneck's channel expansion factor (4×) and kernel size (7×7) are both ablation-worthy hyperparameters

---

## Follow-up Questions

1. **Empirical validation**: What FID/LPIPS scores does a ConvNeXt-NCA denoiser achieve on CelebA/CIFAR-10 vs. Diff-NCA and FourierDiff-NCA?
2. **Adaptive kernel sizes**: Could the NCA learn different kernel sizes for different steps (small kernels early, large kernels late)?
3. **ConvNeXt-NCA for texture synthesis**: Does the architecture improve texture NCA training stability and quality?
4. **DiT-style conditioning**: Can Adaptive Layer Normalization (from Diffusion Transformers) be used for timestep conditioning in ConvNeXt-NCA?
5. **Grouped inverted bottleneck**: Could channel groups in the expansion use different kernel sizes (InceptionNeXt-style) for multi-scale perception?

---

## Sources

- [ConvNeXt: A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545) — Liu et al., 2022
- [ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders](https://arxiv.org/abs/2301.00808) — Woo et al., 2023
- [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/) — Mordvintsev et al., 2020
- [Parameter-efficient diffusion with neural cellular automata](https://www.nature.com/articles/s44335-025-00026-4) — Kalkhof et al., 2025
- [Frequency-Time Diffusion with Neural Cellular Automata](https://arxiv.org/abs/2401.06291) — 2024
- [Attention-based Neural Cellular Automata (ViTCA)](https://proceedings.neurips.cc/paper_files/paper/2022/file/361e5112d2eca09513bbd266e4b2d2be-Paper-Conference.pdf) — Tesfaldet et al., NeurIPS 2022
- [InceptionNeXt: When Inception Meets ConvNeXt](https://arxiv.org/abs/2303.16900) — Yu et al., CVPR 2024
- [Cellular automata as convolutional neural networks](https://arxiv.org/abs/1809.02942) — Gilpin, 2019
- [Neural Cellular Automata: From Cells to Pixels](https://arxiv.org/abs/2506.22899) — Review, 2025
- [A Path to Universal Neural Cellular Automata](https://arxiv.org/abs/2505.13058) — Béna, 2025
- [Neural cellular automata: Applications to biology and beyond classical AI](https://arxiv.org/abs/2509.11131) — Hartl et al., Physics of Life Reviews, 2026
- [Isotropic Neural Cellular Automata](https://google-research.github.io/self-organising-systems/isonca/) — Google Research
- [ConvNeXt Backbone: Modernizing CNNs](https://www.emergentmind.com/topics/convnext-backbone)
- [MedSegDiffNCA: Diffusion Models With Neural Cellular Automata](https://arxiv.org/abs/2501.02447) — 2025
