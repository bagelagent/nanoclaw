# Architecture-Agnostic Progressive Diffusion Distillation

*Research completed: 2026-03-01*
*Topic: Can we distill U-Net diffusion models into MobileNet or efficient ConvNeXt architectures?*

## Summary

Yes, architecture-agnostic diffusion distillation is not only possible but is an active and rapidly maturing area of research. The key insight is that the U-Net inductive bias is not essential to diffusion model performance, and knowledge can be transferred across fundamentally different architectures (CNN, Transformer, MLP, and hybrid designs) through carefully designed distillation objectives. The field has converged on several complementary approaches: data-free distillation from pretrained models (DKDM, CVPR 2025), operator-level grafting within architectures (Liquid AI), linear-attention replacements (EDiT), and NCA-based ultra-lightweight alternatives (Diff-NCA with only 336K parameters). The most practical path to MobileNet/ConvNeXt-based diffusion likely combines cross-architecture distillation techniques with architecture-specific efficiency optimizations.

## Key Findings

### 1. DKDM Proves Architecture-Agnostic Distillation Works (CVPR 2025)

**DKDM (Data-Free Knowledge Distillation for Diffusion Models)** is the landmark paper that directly answers the research question. Key results:

- **Cross-architecture CNN→ViT distillation**: FID 17.86 on CIFAR-10 (data-free)
- **Cross-architecture ViT→CNN distillation**: FID 6.85 on CIFAR-10 (data-free) — *significantly better*, suggesting CNNs are more effective as compressed diffusion models
- **Same-architecture distillation**: IS 8.60, FID 9.56 on CIFAR-10 (vs. teacher: IS 9.52, FID 4.45)
- **Sometimes outperforms data-trained models**: On ImageNet 32×32, DKDM achieved IS 10.50 vs. data-trained IS 9.99

The critical innovation is using **noisy intermediate samples** as the knowledge form rather than final generated images, enabling direct learning from each denoising step without expensive generation. Their dynamic iterative distillation converges faster than naive approaches.

**Implication for MobileNet/ConvNeXt**: The ViT→CNN direction working better than CNN→ViT suggests that efficient CNN architectures like MobileNet or ConvNeXt could be excellent student targets for distilling from larger diffusion teachers.

### 2. MobileDiffusion: Purpose-Built Mobile Architecture (ECCV 2024)

Google's **MobileDiffusion** demonstrates what a fully optimized mobile diffusion architecture looks like:

- **520M parameters** (vs. Stable Diffusion's ~1.29B)
- **0.5 second generation** on iPhone 15 Pro
- Architecture optimizations:
  - Separable convolutions replace standard convolutions in deep UNet layers
  - Residual blocks reduced from 22 (SD) to 12
  - UViT-style bottleneck with more transformer blocks at low resolution
  - 8-channel VAE latent space (vs. 4-channel in SD)
  - DiffusionGAN for one-step sampling

This represents the "design from scratch" approach rather than distillation, but provides a blueprint for what efficient diffusion architectures need.

### 3. SnapFusion: Distillation-Based Mobile Optimization (NeurIPS 2024)

**SnapFusion** took the opposite approach — starting from Stable Diffusion and compressing it:

- **< 2 second generation** on mobile devices
- Architecture evolving: gradually modifying pretrained UNet while preserving performance
- Removed transformer blocks at highest resolution
- Step distillation with CFG regularization: 8 steps beats SD's 50 steps
- VAE decoder compression via prompt-driven distillation

### 4. Cross-Architecture KD Techniques (RSD, OFA-KD)

The **Redundancy Suppression Distillation (RSD)** method (July 2025) provides the theoretical and practical toolkit for making cross-architecture distillation work:

- **2.34% average accuracy gain** across 15 heterogeneous architecture pairs
- Works across CNN (ResNet, MobileNet, ConvNeXt), Transformer (ViT, Swin, DeiT), and MLP models
- Key technique: suppress architecture-specific redundancy in representations, keeping only architecture-invariant knowledge
- **10× fewer extra parameters** than OFA-KD
- ConvNeXt-T → DeiT-T distillation shows 6.70% gain over OFA baseline

While RSD focuses on discriminative tasks, its principles directly apply to distilling diffusion model features.

### 5. Operator Grafting in DiT Models (Liquid AI)

**Grafting** offers a component-level approach to architecture transformation:

- Replace attention operators in DiT with local gated convolutions, sliding window attention, or Mamba-2
- Activation distillation transfers functionality using regression objectives
- Achieves FID 2.77 with 2× depth reduction (28→14 blocks)
- For high-resolution: 1.43× faster, <2% quality drop
- Requires only 12K synthetic samples for distillation

This suggests a hybrid path: rather than wholesale architecture replacement, selectively grafting efficient operators into existing diffusion architectures.

### 6. NCA-Based Ultra-Lightweight Diffusion (Nature, 2025)

**Diff-NCA** and **FourierDiff-NCA** represent the extreme efficiency frontier:

- **Diff-NCA**: 336K parameters generates 512×512 pathology images
- **FourierDiff-NCA**: 1.1M parameters, FID 49.48 on CelebA (vs. 128.2 for 4× larger UNet)
- Leverages NCA's local communication patterns for extreme parameter efficiency
- FourierDiff-NCA adds Fourier-based diffusion for early global communication

While not competitive with state-of-the-art on general image quality, these demonstrate that orders-of-magnitude parameter reduction is possible.

## Deep Dive: The Architecture Gap Problem

### Why Cross-Architecture Distillation is Hard

The fundamental challenge in distilling across architectures is **representation mismatch**:

1. **Spatial encoding differences**: CNNs encode spatial information through local receptive fields with translation equivariance. Transformers use global attention with learned positional encodings. These produce fundamentally different internal representations.

2. **Feature hierarchy mismatch**: U-Net features at corresponding depths may not align semantically with features in a MobileNet or ConvNeXt at the same depth.

3. **The convolution-attention spectrum**: As noted in the ICLR 2026 blog post on diffusion architecture evolution, "convolution is a constrained form of attention" — convolutions are "translation-equivariant, finite-support Toeplitz operators" that form a subset of self-attention kernels. This means going from attention → convolution involves information loss.

### Solutions Emerging in 2025-2026

| Approach | Mechanism | Overhead |
|----------|-----------|----------|
| **Logit-space projection** (OFA-KD) | Project features into architecture-agnostic logit space | Medium (extra projector networks) |
| **Redundancy suppression** (RSD) | Remove architecture-specific information, keep invariants | Low (10× less than OFA) |
| **Activation distillation** (Grafting) | Direct regression of activations through new operators | Low (per-operator) |
| **Time-domain knowledge** (DKDM) | Match noisy intermediate representations | Minimal |
| **Contrastive alignment** | InfoNCE objectives after spatial smoothing | Low |

### Progressive Distillation as Bridge

The original **progressive distillation** framework (Salimans & Ho, ICLR 2022) — repeatedly halving sampling steps — could be combined with architecture transformation:

1. Start: Full U-Net teacher, 1000 steps
2. Step distill: Same U-Net, 8 steps
3. Architecture distill: U-Net → MobileNet backbone, 8 steps
4. Further compress: MobileNet, 4 steps with consistency distillation

This staged approach avoids attempting to simultaneously learn a new architecture and a compressed sampling schedule, which compounds the difficulty.

### The Consistency Model Path

**Consistency Models** (Song et al., 2023; ECT at ICLR 2025) offer an alternative framing:

- Instead of distilling the multi-step process, learn a direct noise→image mapping
- Architecture-agnostic by design: any network that maps noise to images works
- **SANA-Sprint** (ICCV 2025) achieves 0.1s generation at 1024×1024 using linear diffusion transformers + consistency distillation
- A MobileNet or ConvNeXt consistency model could be trained directly from a diffusion teacher

## Connections to Previous Research

### Link to NCA Texture Synthesis Pipeline

This research directly informs the NCA-centric pipeline explored in prior queue items:

- **Diff-NCA** (336K params) already proves NCA can serve as a diffusion backbone
- Cross-architecture distillation (DKDM) could be used to distill from SD3/FLUX into NCA-based architectures
- The cascade routing system (from earlier research) could use architecture-appropriate models at each tier:
  - Tier 1 (fast): NCA or MobileNet-based lightweight generator
  - Tier 2 (quality): ConvNeXt-based distilled diffusion
  - Tier 3 (best): Full transformer-based model

### Link to Perceptual Loss Distillation

The distilled perceptual metrics explored in earlier research (SqueezeNet LPIPS, minimal LPIPS proxies) serve a dual purpose:
- As training losses for the student diffusion model during architecture distillation
- As fast quality assessment for the cascade routing system

### Link to WebGPU/Shader Research

Efficient architectures like MobileNet use depthwise separable convolutions, which map well to GPU shaders. A MobileNet-based diffusion model could potentially run in WebGPU for real-time texture synthesis in the browser.

## Follow-up Questions

1. **Progressive architecture morphing**: Can we gradually transform a U-Net into a MobileNet-style architecture through iterative operator replacement (inspired by grafting), rather than distilling all at once?

2. **NCA-ConvNeXt hybrid denoiser**: Could ConvNeXt's inverted bottleneck blocks serve as NCA update rules, combining NCA's local parallel updates with ConvNeXt's modern design?

3. **Quantization-aware architecture distillation**: Can we simultaneously distill into a smaller architecture and quantize to INT8/INT4, targeting specific mobile accelerators (Apple ANE, Qualcomm Hexagon)?

## Sources

1. Xiang et al., "DKDM: Data-Free Knowledge Distillation for Diffusion Models with Any Architecture," CVPR 2025 — https://arxiv.org/abs/2409.03550
2. Zhao et al., "MobileDiffusion: Instant Text-to-Image Generation on Mobile Devices," ECCV 2024 — https://arxiv.org/html/2311.16567v2
3. Li et al., "SnapFusion: Text-to-Image Diffusion Model on Mobile Devices within Two Seconds," NeurIPS 2024 — https://arxiv.org/abs/2306.00980
4. Zhang et al., "Cross-Architecture Distillation Made Simple with Redundancy Suppression," 2025 — https://arxiv.org/abs/2507.21844
5. Liquid AI, "Exploring Diffusion Transformer Designs via Grafting" — https://www.liquid.ai/research/exploring-diffusion-transformer-designs-via-grafting
6. Kalkhof et al., "Parameter-efficient diffusion with neural cellular automata," npj Unconventional Computing, 2025 — https://www.nature.com/articles/s44335-025-00026-4
7. Salimans & Ho, "Progressive Distillation for Fast Sampling of Diffusion Models," ICLR 2022 — https://arxiv.org/abs/2202.00512
8. Song et al., "Consistency Models," ICML 2023 — https://arxiv.org/abs/2303.01469
9. "From U-Nets to DiTs: The Architectural Evolution of Text-to-Image Diffusion Models," ICLR 2026 Blogposts — https://iclr-blogposts.github.io/2026/blog/2026/diffusion-architecture-evolution/
10. Chen et al., "SANA-Sprint: One-Step Diffusion with Continuous-Time Consistency Distillation," ICCV 2025 — https://openaccess.thecvf.com/content/ICCV2025/papers/Chen_SANA-Sprint_One-Step_Diffusion_with_Continuous-Time_Consistency_Distillation_ICCV_2025_paper.pdf
11. "EDiT: Efficient Diffusion Transformers with Linear Compressed Attention," 2025 — https://arxiv.org/html/2503.16726v1
12. QIANLONG0502/DKDM GitHub Repository — https://github.com/qianlong0502/DKDM
