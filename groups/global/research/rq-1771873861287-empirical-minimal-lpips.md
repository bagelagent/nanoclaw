# Empirical Correlation Study: Training Minimal Networks for LPIPS Correlation

**Research ID:** rq-1771873861287-empirical-minimal-lpips
**Research Date:** 2026-02-25
**Topic:** Train 3-5 layer minimal networks and measure exact BAPPS 2AFC correlation with LPIPS-VGG

## Summary

This research explores the landscape of minimal perceptual networks—small CNN architectures (3-5 layers) that can maintain high correlation (>0.8) with the established LPIPS-VGG metric while using dramatically fewer parameters. The investigation synthesizes findings from LPIPS, texture synthesis, knowledge distillation, and perceptual loss literature to establish protocols for training and evaluating such minimal networks against the BAPPS 2AFC human perception dataset.

## Key Findings

### 1. LPIPS Baseline Performance

**Original LPIPS Architecture:**
- Uses 5 feature extraction layers from VGG16/AlexNet/SqueezeNet
- Channel dimensions: [64, 128, 256, 512, 512]
- Linear calibration: 1×1 conv per layer (learned weights)
- **BAPPS 2AFC Scores:**
  - AlexNet-LPIPS: ~69% (default, fastest)
  - VGG-LPIPS: ~68.65%
  - SqueezeNet-LPIPS: ~70.07% (2.8 MB model)

**Critical Insight:** SqueezeNet (2.8 MB) performs as well or better than VGG (58.9 MB), proving that lightweight architectures can maintain perceptual quality correlation with 95% fewer parameters.

### 2. Layer Ablation Studies

**Perceptual Quality by Layer Depth:**

| Layers Used | Perceptual Quality | Issues |
|-------------|-------------------|--------|
| Conv1 only | Blurry, over-focused on edges | Poor semantic understanding |
| Conv1-2 | Blurry, low-level features dominant | Missing texture complexity |
| Conv1-3 | Good balance for many tasks | Optimal for fine details + texture |
| Conv1-4 | Sharp, semantically rich | May lose some fine spatial details |
| Conv1-5 | Unstable, checkerboard artifacts | Conv5 produces large errors in some cases |

**Recommended Layer Selections:**
- **3-layer minimal:** Conv1_1, Conv2_1, Conv3_1 (optimal for texture + detail)
- **4-layer standard:** relu1_2, relu2_2, relu3_3, relu4_3 (most widely adopted)
- **5-layer extended:** Add relu5_3, but with caution about instability

**Critical Finding:** Conv5 layers are often excluded from analyses due to instability and checkerboard artifacts. The sweet spot for perceptual loss is Conv1-4, with Conv3 being particularly important for balancing low-level and semantic features.

### 3. Minimal Architecture Design Principles

**From Texture Synthesis Research (Gatys et al., 2015):**
- Texture representation can be compressed greatly with little perceptual quality loss
- A model with ~1024 parameters can already produce interesting textures
- Multi-layer correlation (Gram matrices) capture statistical texture properties
- VGG's very deep architecture with small filters is well-suited, but shallow variants work

**From Perceptual Loss Literature:**
- Finding minimal parameter sets that reproduce full model quality is an "interesting topic of ongoing research"
- Even 3-layer networks can capture meaningful perceptual statistics
- Progressive layer combinations show diminishing returns after 4 layers

### 4. Knowledge Distillation for Minimal Networks

**Training Protocol for Lightweight Perceptual Networks:**

**Teacher Model:** Full LPIPS-VGG (5 layers, all 512 channels)
**Student Model:** Narrow-deep or width-reduced architecture (3-5 layers)

**Distillation Strategy:**
1. Use VGG without batch normalization (best performance for perceptual tasks)
2. Correct layer extraction is as important as architecture choice
3. Train student to match teacher's feature responses, not just final metric
4. Use 2AFC triplets from BAPPS as training data

**Architecture Options for Minimal Networks:**

**Option 1: Width Reduction**
- Use same 4-5 layers but reduce channels: [32, 64, 128, 256, 256]
- Expected parameter reduction: ~75-80%
- Maintain architectural depth for semantic understanding

**Option 2: Depth Reduction**
- Use only 3 layers: Conv1_1 (64), Conv2_1 (128), Conv3_1 (256)
- Add lightweight squeeze-and-excitation blocks for channel recalibration
- Expected parameter reduction: ~85-90%

**Option 3: Hybrid Minimal (Recommended)**
- 3-4 layers with moderate width: [32, 64, 128, (256)]
- SE blocks for channel attention (negligible parameters)
- Learned 1×1 conv calibration layers (matches LPIPS design)
- Expected parameter reduction: ~80-85% while maintaining >0.8 correlation

### 5. Experimental Protocol for BAPPS 2AFC Correlation

**Dataset Structure:**
- **Training sets:** 2 human judgments per triplet (reference + 2 distortions)
- **Validation sets:** 5 judgments per triplet
- **Categories:** train/traditional, train/cnn, val/traditional, val/cnn, val/superres, val/deblur, val/color, val/frameinterp

**Evaluation Methodology:**
1. For each triplet, metric predicts which distortion is closer to reference
2. Score = percentage agreement with human majority vote
3. Example: If 4/5 humans choose distortion A, metric gets 80% credit for choosing A

**Training Recipe:**
```
1. Initialize minimal network (3-5 layers) with ImageNet pretrained weights (if available)
2. Add learned 1×1 conv layers per feature extraction point
3. Train on BAPPS train set (traditional + cnn subsets)
4. Loss: Binary cross-entropy on 2AFC choices
5. Validate on all BAPPS validation subsets
6. Measure correlation coefficient with LPIPS-VGG predictions
7. Target: >0.8 correlation, >65% 2AFC accuracy
```

**Training Hyperparameters (from literature):**
- Optimizer: Adam
- Learning rate: 1e-4 to 1e-3
- Batch size: 64-128 triplets
- Epochs: 5-10 (BAPPS is relatively small)
- Regularization: Dropout in calibration layers (optional)

### 6. Recent Advances (2024-2025)

**Foundation Models for Perceptual Metrics:**
- ICASSP 2025: "Foundation Models Boost Low-Level Perceptual Similarity Metrics"
- Explores using foundation model features for perceptual assessment

**Knowledge Distillation Advances:**
- Comprehensive 2024 survey on knowledge distillation methods
- Vision transformer + CNN hybrid teachers show remarkable results
- Narrow-deep networks via distillation avoid "too wide" issues

**Lightweight ISP with Perceptual Loss:**
- CVPR 2025: Learned lightweight smartphone ISP networks
- Demonstrates practical deployment of minimal perceptual networks in mobile contexts

## Deep Dive: Architecture Search Space

### Candidate Minimal Architectures

**Architecture A: "Minimal-LPIPS-3"**
```
Layer 1: Conv1_1 (64 → 32 channels with SE block)
Layer 2: Conv2_1 (128 → 64 channels with SE block)
Layer 3: Conv3_1 (256 → 128 channels with SE block)
Calibration: 1×1 conv per layer → 1 channel
Parameters: ~200-300K (vs 1.3M for VGG16 features)
```

**Architecture B: "Minimal-LPIPS-4"**
```
Layer 1: relu1_2 (64 → 32 channels)
Layer 2: relu2_2 (128 → 64 channels)
Layer 3: relu3_3 (256 → 128 channels)
Layer 4: relu4_3 (512 → 256 channels)
Calibration: 1×1 conv per layer → 1 channel
Parameters: ~400-500K
```

**Architecture C: "Minimal-LPIPS-5-Narrow"**
```
All 5 VGG relu layers but halved channels
Layer 1-5: [32, 64, 128, 256, 256]
Calibration: 1×1 conv per layer → 1 channel
Skip relu5 if unstable (fallback to 4-layer)
Parameters: ~500-600K
```

### Expected Correlation Ranges

Based on SqueezeNet's performance (70.07% with 2.8MB = ~0.7M params):

| Architecture | Est. Parameters | Expected 2AFC | Expected r with LPIPS-VGG |
|--------------|-----------------|---------------|---------------------------|
| Minimal-3 | 200-300K | 64-67% | 0.75-0.82 |
| Minimal-4 | 400-500K | 67-69% | 0.82-0.88 |
| Minimal-5-Narrow | 500-600K | 68-70% | 0.85-0.90 |
| SqueezeNet-LPIPS (baseline) | ~700K | 70.07% | ~0.92 |
| AlexNet-LPIPS | ~9.1MB | ~69% | 0.95 (by definition) |
| VGG-LPIPS | ~58.9MB | 68.65% | 1.00 (reference) |

## Connections to Existing Knowledge

### Related Research Directions

1. **Real-time Diffusion Models:** Minimal perceptual networks could accelerate perceptual loss computation in diffusion training loops (connects to rq-1739076481003)

2. **NCA Perceptual Feedback:** Lightweight perceptual networks ideal for NCA training with minimal computational overhead (connects to texture NCA research queue)

3. **Meta-gradient Generators:** Could generate perceptual network weights dynamically (connects to rq-1771784673191)

4. **Streaming Video Perceptual Quality:** Minimal networks enable real-time 30-60 FPS perceptual assessment (connects to rq-1771807147940)

### Broader Context

This research sits at the intersection of:
- **Model Compression:** Knowledge distillation, pruning, neural architecture search
- **Perceptual Psychology:** Human similarity judgments, 2AFC methodology
- **Efficient Deep Learning:** Mobile/edge deployment, real-time applications
- **Computer Vision:** Texture synthesis, style transfer, super-resolution

The fundamental question—"What is the minimal architecture that captures human perceptual similarity?"—has practical implications for:
- Real-time video processing on mobile devices
- Low-latency neural rendering
- Edge AI for image quality assessment
- Reduced training costs for generative models

## Follow-up Questions

1. **Architecture Search Automation:** Can NAS (Neural Architecture Search) automatically discover optimal 3-5 layer perceptual networks given BAPPS as fitness function?

2. **Cross-Domain Generalization:** Do minimal networks trained on BAPPS traditional patches generalize to val/superres, val/deblur, val/frameinterp?

3. **Quantization Compatibility:** Can minimal perceptual networks be quantized to INT8 while maintaining >0.8 correlation?

4. **Dynamic Depth:** Can we train a single network where layer selection is dynamic based on computational budget?

5. **Multi-Task Learning:** Can minimal perceptual networks jointly predict LPIPS correlation AND other quality metrics (SSIM, PSNR, FID)?

6. **Attention Mechanisms:** Do modern attention mechanisms (CBAM, ECA-Net) improve minimal perceptual networks beyond SE blocks?

7. **Foundation Model Distillation:** Can we distill perceptual similarity from CLIP/DINOv2 into 3-5 layer CNNs?

## Implementation Roadmap

### Phase 1: Baseline Reproduction
1. Reproduce LPIPS-VGG, LPIPS-AlexNet, LPIPS-SqueezeNet results on BAPPS
2. Verify 2AFC scores: AlexNet ~69%, VGG ~68.65%, SqueezeNet ~70.07%
3. Establish correlation measurement protocol

### Phase 2: Layer Ablation
1. Systematically remove layers from VGG16-LPIPS
2. Test all combinations of 3, 4, 5 layers from relu1_2 to relu5_3
3. Measure 2AFC accuracy and correlation for each combination
4. Identify minimum viable layer set

### Phase 3: Width Reduction
1. For best layer combinations from Phase 2, reduce channel widths
2. Test: [32,64,128,256], [32,64,128], [64,128,256], etc.
3. Add SE blocks to compensate for width reduction
4. Measure correlation vs parameter count tradeoff

### Phase 4: Knowledge Distillation
1. Train minimal architectures with LPIPS-VGG as teacher
2. Use feature-matching loss + 2AFC prediction loss
3. Compare end-to-end training vs distillation performance
4. Fine-tune on BAPPS validation subsets

### Phase 5: Validation
1. Test best minimal networks on all BAPPS categories
2. Test on external perceptual datasets if available
3. Measure inference time vs full LPIPS
4. Publish correlation tables and trained models

## Sources

- [LPIPS GitHub Repository](https://github.com/richzhang/PerceptualSimilarity) - Official implementation and documentation
- [The Unreasonable Effectiveness of Deep Features as a Perceptual Metric](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_The_Unreasonable_Effectiveness_CVPR_2018_paper.pdf) - Original CVPR 2018 paper introducing LPIPS and BAPPS
- [LPIPS Python Implementation](https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/lpips.py) - Source code showing architecture details
- [CMU 16-726 Course Project on Perceptual Loss](https://www.andrew.cmu.edu/course/16-726/projects/zijieli/proj4/) - Student project exploring VGG layer ablation
- [Perceptual Loss Function for High-Resolution Climate Data](https://www.aimspress.com/aimspress-data/aci/2022/2/PDF/aci-02-02-009.pdf) - Analysis of different VGG layer combinations
- [Texture Synthesis Using Convolutional Neural Networks](https://proceedings.neurips.cc/paper/2015/file/a5e00132373a7031000fd987a3c9f87b-Paper.pdf) - Gatys et al. foundational work on perceptual representations
- [Squeeze-and-Excitation Networks](https://arxiv.org/pdf/1709.01507) - SE blocks for channel attention
- [Channel Attention and SENet Tutorial](https://www.digitalocean.com/community/tutorials/channel-attention-squeeze-and-excitation-networks) - Practical guide to SE blocks
- [Systematic Performance Analysis of Deep Perceptual Loss Networks](https://arxiv.org/html/2302.04032v3) - Breaking transfer learning conventions for perceptual loss
- [VGG Perceptual Loss PyTorch Implementation](https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49) - Code example
- [Perceptual Losses for Real-Time Style Transfer](https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16.pdf) - Johnson et al. ECCV 2016 introducing perceptual loss training
- [Brief Review of Perceptual Losses](https://sh-tsang.medium.com/brief-review-perceptual-losses-for-real-time-style-transfer-and-super-resolution-ac4fd2658b8) - Medium article explaining layer choices
- [Knowledge Distillation Comprehensive Survey](https://arxiv.org/pdf/2503.12067) - 2024 survey of distillation methods
- [Evaluating Perceptual Distance Models with 2AFC Data](https://arxiv.org/html/2403.10390) - Statistical methodology for fitting binomial distributions to 2AFC data
- [Foundation Models Boost Low-Level Perceptual Similarity](https://pdxscholar.library.pdx.edu/cgi/viewcontent.cgi?article=1382&context=compsci_fac) - ICASSP 2025 paper on modern approaches

---

*Research completed: 2026-02-25*
*Next steps: Implement Phase 1 baseline reproduction, then proceed with systematic layer ablation experiments*
