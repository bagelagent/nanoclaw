# Systematic Layer Ablation for Texture NCAs

**Research ID:** rq-1771829371565-nca-layer-ablation
**Research Date:** February 27, 2026
**Topic:** Test VGG/SqueezeNet layers and measure perceptual quality vs FLOP reduction

---

## Summary

Layer selection in perceptual loss functions for Neural Cellular Automata (NCA) texture synthesis represents a critical tradeoff between computational efficiency and perceptual quality. Research shows that VGG architectures (particularly VGG-16 and VGG-19) dominate texture synthesis quality, with multi-layer Gram matrix approaches using conv1 through conv5 layers producing the best results. However, SqueezeNet offers a compelling lightweight alternative with 50× fewer parameters and 10× fewer FLOPs than comparable networks, achieving similar perceptual similarity metrics (70.07% accuracy in LPIPS 2AFC tests vs 68.65% for VGG) while sacrificing some fine-grained texture quality.

The key insight is that **layer choice matters as much as architecture choice** - selecting optimal extraction layers can dramatically impact both quality and efficiency, with early layers (conv1-conv3) capturing fine details at lower computational cost, while deeper layers (conv4-conv5) capture broader semantic patterns at higher cost.

---

## Key Findings

### 1. VGG Layer-Wise Computational Costs

VGG-16's computational burden is heavily concentrated in later layers:

| Layer | MegaFLOPs | Input Dims |
|-------|-----------|------------|
| block1_conv1 | 173.4 | 224×224×3 |
| block1_conv2 | 3,699.4 | 224×224×64 |
| block2_conv1 | 1,849.7 | 112×112×64 |
| block2_conv2 | 3,699.4 | 112×112×128 |
| block3_conv1-3 | 1,849.7 - 3,699.4 | 56×56×256 |
| block4_conv1-3 | 1,849.7 - 3,699.4 | 28×28×512 |
| block5_conv1-3 | Similar | 14×14×512 |

**Key observation:** The first conv layer of each block has ~50% fewer FLOPs than subsequent layers due to reduced input depth after pooling. The very first layer (conv1_1) is dramatically cheaper at only 173 MegaFLOPs.

### 2. SqueezeNet Architecture and Efficiency

SqueezeNet achieves dramatic parameter reduction through its Fire Module design:

- **Squeeze layer:** Uses only 1×1 filters to reduce depth
- **Expand layer:** Mix of 1×1 and 3×3 filters to restore expressiveness
- **Parameter reduction:** 50× fewer parameters than AlexNet (~2.8 MB vs ~58.9 MB for VGG)
- **FLOP reduction:** 10× fewer floating point operations
- **Accuracy tradeoff:** Can achieve AlexNet-level accuracy but with limitations on fine-grained texture

**Fire Module efficiency:** By reducing depth before 3×3 convolutions, computational cost drops since 3×3 filters require 9× the computation of 1×1 filters.

**Optimal 3×3 filter ratio:** Research shows top-5 accuracy plateaus at 85.6% with 50% 3×3 filters - more 3×3 filters increase model size without improving accuracy on ImageNet.

### 3. Layer Selection for Texture Synthesis

#### Standard NCA Practice (Distill.pub 2021)
- **Layers used:** block1_conv1, block2_conv1, block3_conv1, block4_conv1, block5_conv1 (first layer of each VGG block)
- **Rationale:** Captures texture at multiple scales from fine to coarse
- **Loss function:** L₂ distance between Gram matrices of activations
- **VGG frozen:** Gradients only flow through NCA, keeping VGG weights fixed

#### Multi-layer Style Transfer Best Practices
- **Optimal configuration:** Using all 5 layers (conv1-conv5) produces best texture quality
- **Content loss layers:** conv1-conv3 allow near-perfect reconstruction; quality degrades in conv4-conv5 as network focuses on broader semantics
- **Style loss layers:** conv4 performs best for style transfer individually, but combining all layers yields optimal results
- **Standard style loss layers:** conv1_1, conv2_1, conv3_1, conv4_1, conv5_1 (VGG-19)
- **Alternative configuration:** conv2_2, conv3_4, conv4_4, conv5_2 for texture loss

#### Quality vs Efficiency Tradeoffs
- **Single layer (e.g., conv4_2):** Fast but limited expressiveness
- **Early layers (conv1-conv3):** Capture fine details, lower computational cost
- **Deep layers (conv4-conv5):** Capture semantic patterns, higher cost
- **Multi-layer Gram matrices:** Best quality but multiplicative cost increase

### 4. LPIPS Layer Configuration and Performance

The Learned Perceptual Image Patch Similarity (LPIPS) metric provides empirical comparison across architectures:

| Network | Layers | Channels | Model Size | 2AFC Accuracy | Best Use Case |
|---------|--------|----------|------------|---------------|---------------|
| VGG-16 | 5 | [64,128,256,512,512] | 58.9 MB | 68.65% | Training/optimization ("traditional perceptual loss") |
| SqueezeNet | 7 | [64,128,256,384,384,512,512] | 2.8 MB | 70.07% | Resource-constrained scenarios |
| AlexNet | 5 | [64,192,384,256,256] | 9.1 MB | ~69% | Forward evaluation (fastest, best as forward metric) |

**Key insight:** SqueezeNet actually achieves slightly higher accuracy in human perceptual similarity tests (70.07% vs 68.65%) despite being 21× smaller than VGG, suggesting that parameter efficiency doesn't necessarily sacrifice perceptual quality.

### 5. Batch Normalization Impact

A critical finding for layer ablation studies: **VGG-16 with batch normalization performed in the bottom four on most benchmarks**, while VGG-16 without batch norm placed in the top three. This suggests batch norm interferes with perceptual feature extraction for texture tasks.

### 6. Recent NCA Efficiency Research (2024-2025)

#### Diff-NCA and FourierDiff-NCA (2025)
- **Diff-NCA:** Generates 512×512 images with only 336k parameters
- **FourierDiff-NCA:** 1.1M parameters, achieves FID score of 49.48 (>2× better than 4× larger UNet)
- **Application:** Parameter-efficient diffusion models using NCA architecture

#### High-Resolution NCA with Implicit Decoders (2025)
- **Architecture:** NCA on coarse grid + lightweight implicit decoder
- **Loss function:** VGG-based texture loss with relaxed optimal transport
- **Multi-resolution:** Same model renders at arbitrary resolution
- **Layer extraction:** Features from pretrained VGG16 at multiple layers

### 7. SqueezeNet Limitations for Texture Tasks

Despite efficiency gains, SqueezeNet has documented weaknesses:

- **Fine-grained features:** Struggles with accuracy (84.48% F1 score in some tasks)
- **Texture quality:** Produces poorer quality than VGG-19 in texture synthesis
- **Robustness:** "Extremely compact but often misses defects" - recommended only for highly resource-constrained scenarios
- **Comparison to AlexNet:** Less robust than AlexNet for texture recreation, though still more efficient

---

## Deep Dive: Optimal Layer Selection Strategy for NCAs

### The Layer Ablation Framework

A systematic ablation study for NCA texture synthesis should test:

1. **Single-layer baselines:**
   - Early (conv1_1, conv2_1): Test fine detail capture
   - Mid (conv3_1, conv4_1): Test texture pattern capture
   - Deep (conv5_1): Test semantic/coarse pattern capture

2. **Multi-layer combinations:**
   - Adjacent pairs (conv1-2, conv2-3, etc.): Test hierarchical benefits
   - Standard full-stack (conv1-5): Current best practice baseline
   - Selective combinations: Remove expensive layers, measure quality degradation

3. **Architecture comparison:**
   - VGG-16 vs VGG-19: Depth vs efficiency
   - VGG vs SqueezeNet: Quality vs parameters/FLOPs
   - With vs without batch norm: Feature extraction quality

### Metrics for Evaluation

**Perceptual Quality:**
- LPIPS (VGG, SqueezeNet, AlexNet variants)
- DISTS (Deep Image Structure and Texture Similarity)
- FID (Fréchet Inception Distance)
- KID (Kernel Inception Distance)
- SSIM (Structural Similarity Index)
- Human evaluation (2AFC tests)

**Computational Efficiency:**
- Total FLOPs for forward pass
- GPU memory footprint
- Training time per iteration
- Inference time (for real-time applications)
- Parameter count of combined NCA + loss network

**Training Dynamics:**
- Convergence speed (iterations to acceptable quality)
- Stability (gradient norms, loss variance)
- Generalization (performance on held-out textures)

### Expected Tradeoff Curves

Based on current research, we would expect to find:

1. **Quality-FLOP frontier:**
   - VGG-19 conv1-5: Highest quality, highest cost (baseline)
   - VGG-16 conv1-4: ~95% quality, ~70% cost
   - VGG-16 conv1-3: ~85% quality, ~40% cost
   - SqueezeNet 7-layer: ~90% quality, ~10% cost

2. **Quality plateau zones:**
   - Adding conv5 to conv1-4 may provide minimal benefit
   - First 3 layers may capture most texture information
   - Diminishing returns after 3-4 layers

3. **Critical layers:**
   - conv3/conv4 likely contain most texture-critical features
   - Early layers (conv1-2) may be redundant for coarse textures
   - Deep layers (conv5) may be unnecessary for non-semantic textures

### Proposed Experimental Protocol

#### Phase 1: Single-Layer Ablation
For each layer in {conv1_1, conv2_1, conv3_1, conv4_1, conv5_1}:
1. Train identical NCA architecture
2. Use only that layer for Gram matrix loss
3. Measure convergence time, final LPIPS, FLOPs
4. Identify which layer is most "information-dense" for texture

#### Phase 2: Additive Layer Combination
Starting from best single layer from Phase 1:
1. Add one layer at a time (both earlier and later)
2. Measure marginal quality improvement vs marginal FLOP cost
3. Plot quality-efficiency frontier
4. Identify "elbow points" where adding more layers yields diminishing returns

#### Phase 3: Architecture Comparison
For optimal layer combinations from Phase 2:
1. Compare VGG-16 vs VGG-19
2. Compare VGG vs SqueezeNet at equivalent layers
3. Test with/without batch norm
4. Measure transfer learning: train on texture A, test on texture B

#### Phase 4: NCA-Specific Optimization
Based on results from Phases 1-3:
1. Test distilled SqueezeNet (with PCA dimensionality reduction)
2. Test MILO (pseudo-MOS training for layer compression)
3. Compare to specialized lightweight losses (sliced Wasserstein)
4. Evaluate hybrid approaches (SqueezeNet during exploration, VGG for fine-tuning)

---

## Connections to Existing Knowledge

### Related Research Topics in Queue

This ablation study connects to several other research topics:

1. **Hybrid loss scheduling (rq-1771851237044):** Could use layer ablation results to inform when to switch from SqueezeNet to VGG during training
2. **Distilled SqueezeNet benchmarks (rq-1771851531076):** Layer ablation provides baseline for comparing distillation approaches
3. **MILO adaptation (rq-1771851531077):** Understanding which layers matter most informs pseudo-MOS training targets
4. **Multi-scale CLIP+VGG (rq-1770914400002):** Could apply layer ablation findings to hierarchical NCAs
5. **Differentiable sliced Wasserstein (rq-1771851237046):** Alternative loss functions could replace expensive multi-layer VGG losses

### Broader ML Context

**Perceptual loss evolution:**
- Original Gatys et al. (2015): Texture synthesis via Gram matrices
- Johnson et al. (2016): Real-time style transfer via perceptual loss
- Zhang et al. (2018): "Unreasonable effectiveness" of deep features (LPIPS)
- Current trend: Lightweight alternatives (SqueezeNet, MobileNet) for edge deployment

**NCA-specific considerations:**
- NCAs are already lightweight (~100k-1M parameters)
- Loss network often dominates computational cost
- Real-time texture synthesis requires <40ms per frame (25 Hz)
- Mobile/edge deployment demands <10MB models

---

## Follow-up Questions

1. **Can we distill VGG layer knowledge into a tiny student network?**
   - Train SqueezeNet-like network to mimic VGG Gram matrix statistics
   - Use teacher-student approach with layer-wise distillation losses
   - Potential for custom texture-optimized loss network

2. **Do different texture classes benefit from different layer combinations?**
   - Organic textures (wood, stone) vs geometric (tiles, fabric)
   - High-frequency (grass, fur) vs low-frequency (clouds, water)
   - Per-texture-class layer selection could optimize efficiency

3. **Can we learn which layers to use via meta-learning?**
   - Treat layer selection as a differentiable choice
   - Learn layer importance weights during training
   - Sparse layer selection with L1 regularization

4. **How do layer choices affect NCA robustness to damage/perturbations?**
   - Different layers might encode different robustness properties
   - Testing layer ablation under adversarial attacks
   - Resilience to initialization, damage, and noise

5. **What is the minimum viable perceptual loss for NCAs?**
   - Single conv layer + learned calibration?
   - Handcrafted texture statistics (no deep network)?
   - Fourier/wavelet domain losses vs spatial domain?

6. **Can we replace multi-layer VGG with learned lightweight alternatives?**
   - Train end-to-end: NCA + loss network jointly
   - Use NAS to discover optimal loss network architecture
   - Evaluate learned loss vs traditional VGG-based loss

---

## Sources

### Academic Papers & Research

1. [Multi-texture synthesis through signal responsive neural cellular automata](https://www.nature.com/articles/s41598-025-23997-7) - Scientific Reports, 2025
2. [DyNCA: Real-time Dynamic Texture Synthesis Using Neural Cellular Automata](https://openaccess.thecvf.com/content/CVPR2023/papers/Pajouheshgar_DyNCA_Real-Time_Dynamic_Texture_Synthesis_Using_Neural_Cellular_Automata_CVPR_2023_paper.pdf) - CVPR 2023
3. [Self-Organising Textures](https://distill.pub/selforg/2021/textures/) - Distill.pub, 2021
4. [Parameter-efficient diffusion with neural cellular automata](https://www.nature.com/articles/s44335-025-00026-4) - npj Unconventional Computing, 2025
5. [A Systematic Performance Analysis of Deep Perceptual Loss Networks](https://arxiv.org/html/2302.04032v3) - arXiv 2023
6. [The Unreasonable Effectiveness of Deep Features as a Perceptual Metric](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_The_Unreasonable_Effectiveness_CVPR_2018_paper.pdf) - CVPR 2018 (LPIPS)
7. [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://link.springer.com/chapter/10.1007/978-3-319-46475-6_43) - ECCV 2016
8. [Long Range Constraints for Neural Texture Synthesis Using Sliced Wasserstein Loss](https://arxiv.org/html/2211.11137) - arXiv 2022

### Technical Resources & Implementations

9. [LPIPS GitHub Repository](https://github.com/richzhang/PerceptualSimilarity) - Official LPIPS implementation
10. [VGG models for perceptual loss - PyTorch Image Models Issue](https://github.com/huggingface/pytorch-image-models/issues/299)
11. [Keras FLOP Estimator](https://github.com/ckyrkou/Keras_FLOP_Estimator) - Tool for estimating FLOPs
12. [Neural Cellular Automata - GitHub Implementation](https://github.com/0xekez/neural-cellular-automata)

### Architecture Comparisons & Analysis

13. [Comparative Analysis of Lightweight Deep Learning Models](https://arxiv.org/html/2505.03303v1) - arXiv 2025
14. [SqueezeNet: AlexNet-level accuracy](https://arxiv.org/pdf/1602.07360) - Original SqueezeNet paper
15. [SqueezeNet: The Key to Unlocking Edge Computing](https://medium.com/sfu-cspmp/squeezenet-the-key-to-unlocking-the-potential-of-edge-computing-c8b224d839ba)
16. [VGG16 Layer-wise FLOPS Analysis](https://www.researchgate.net/figure/The-original-and-pruned-model-FLOPs-on-each-layer-for-VGG-16-on-CIFAR-10_fig2_329196072)

### Texture Synthesis & Style Transfer

17. [Texture Synthesis Using Convolutional Neural Networks](https://www.cs.toronto.edu/~bonner/courses/2022s/csc2547/papers/discriminative/image-transformation/texture-synthesis,-gatys,-nips-2015.pdf) - Gatys et al., NIPS 2015
18. [Neural Style Transfer from Scratch](https://medium.com/@JohnyOnTheSpot/neural-style-transfer-from-scratch-a-deep-dive-using-vgg19-and-gram-matrices-in-pytorch-598b272e8fc3)
19. [Texture Synthesis with Deep Learning](https://medium.com/@ThomasArtemius/texture-synthesis-fb48fb1db182)
20. [Precomputed Real-Time Texture Synthesis with Markovian GANs](https://www.researchgate.net/publication/308277264_Precomputed_Real-Time_Texture_Synthesis_with_Markovian_Generative_Adversarial_Networks)

### Recent Applications

21. [FacadeGAN: Facade Texture Placement with GANs](https://www.mdpi.com/2076-3417/16/2/860) - 2026
22. [Inkartgan: Deep learning for ink painting](https://link.springer.com/article/10.1007/s10845-025-02674-6) - Journal of Intelligent Manufacturing, 2025
23. [FloLPIPS: Bespoke Video Quality Metric](https://arxiv.org/html/2207.08119)
24. [Deep learning model using SqueezeNet for music genre classification](https://www.nature.com/articles/s41598-025-16499-z) - Scientific Reports, 2025

---

## Research Methodology Notes

This research synthesis was conducted through:
- 10 web searches across academic databases and technical resources
- Analysis of 50+ sources including peer-reviewed papers, technical blogs, and GitHub repositories
- Cross-referencing between NCA-specific research and broader perceptual loss literature
- Extraction of quantitative metrics (FLOPs, parameter counts, accuracy scores)
- Synthesis of best practices from style transfer, texture synthesis, and NCA communities

The research reveals a clear gap: while layer selection has been extensively studied for style transfer and super-resolution, **systematic layer ablation specifically for NCA texture synthesis is underexplored**. Most NCA papers use the standard "first layer of each VGG block" approach from the original Distill.pub work without questioning whether this is optimal for different texture types, computational budgets, or application domains.

**Recommended next step:** Design and execute the 4-phase experimental protocol outlined above, starting with single-layer ablation on a diverse texture dataset (DTD, Describable Textures Dataset) to identify information-dense layers for NCA-specific texture synthesis.
