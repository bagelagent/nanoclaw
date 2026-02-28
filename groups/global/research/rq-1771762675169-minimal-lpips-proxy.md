# Minimal Perceptual Networks: Achieving >0.8 LPIPS Correlation with 3-5 Layer CNNs

**Research Topic:** Minimal perceptual network that maintains >0.8 correlation with LPIPS - systematic search for 3-5 layer CNN architectures

**Research Date:** 2026-02-26

**Tags:** perceptual-metrics, model-compression, lpips, distillation

---

## Executive Summary

The goal of creating minimal CNN architectures (3-5 layers) that maintain >0.8 correlation with LPIPS (Learned Perceptual Image Patch Similarity) is both achievable and actively researched. The key insight from recent literature is that **perceptual quality assessment is an emergent property of deep networks that doesn't require massive model capacity**. SqueezeNet (2.8 MB, ~1.2M parameters) achieves similar perceptual scores to VGG (58.9 MB, ~138M parameters) on the BAPPS benchmark, demonstrating that compact architectures can capture human perceptual judgments effectively.

The path forward involves three main strategies:
1. **Leveraging existing compact backbones** (SqueezeNet, MobileNetV3-Small)
2. **Knowledge distillation from LPIPS teacher models**
3. **Neural architecture search (NAS) with perceptual metrics as optimization targets**

---

## Background: LPIPS and Human Perception

### What is LPIPS?

LPIPS (Learned Perceptual Image Patch Similarity) was introduced by Zhang et al. (CVPR 2018) in their seminal paper "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric." The metric computes perceptual similarity by comparing deep network activations between two images, with a learned linear calibration on top of intermediate convolutional features.

**Key Finding:** Deep network activations work surprisingly well as perceptual similarity metrics **across different architectures and supervisory signals** (supervised, self-supervised, even unsupervised).

### BAPPS Benchmark

The Berkeley-Adobe Perceptual Patch Similarity (BAPPS) dataset is the standard benchmark for evaluating perceptual metrics:

- **36,344 samples** with reference + 2 distorted images
- **Two evaluation modes:**
  - 2AFC (Two Alternative Forced Choice): humans select which distorted image is closer to reference
  - JND (Just Noticeable Differences): humans judge if patches are same/different
- **Baseline human agreement:** 73.9% (cross-rater agreement on 2AFC task)
- **LPIPS performance:** Outperforms all previous metrics by large margins
- **Alternative metrics:** PIM (Perceptual Information Metric) achieves ~69.06% on BAPPS-2AFC

---

## Key Insight: Size Doesn't Matter (Much)

The most important finding for minimal architecture design:

> **Network architectures of vastly different sizes (SqueezeNet 2.8MB, AlexNet 9.1MB, VGG 58.9MB) achieve similar perceptual similarity scores on BAPPS.**

This suggests that:
1. Perceptual quality assessment is an **emergent property** of deep networks
2. **Parameter efficiency** is achievable without sacrificing perceptual accuracy
3. The **supervisory signal** (how the network was trained) matters less than expected

---

## Existing Compact Architectures

### 1. SqueezeNet-based LPIPS (Baseline: 2.8 MB)

- **Architecture:** Fire modules with squeeze + expand layers
- **Parameters:** ~1.2M
- **Performance:** Comparable to VGG on BAPPS despite 20x smaller size
- **Status:** Already implemented in official LPIPS library
- **Layers:** 18 convolutional layers (but many are 1x1 convolutions)

**Analysis:** SqueezeNet is already quite minimal, but it's not a 3-5 layer architecture. However, it proves that massive parameter counts are unnecessary.

### 2. MobileNetV3-Small (<5M parameters)

Recent research (2025) demonstrates MobileNetV3-Small's effectiveness for perceptual quality assessment:

- **Face Image Quality Assessment:** Achieves high accuracy with sub-million-parameter ensembles
- **Architecture features:**
  - Depthwise-separable convolutions
  - Squeeze-and-excitation (SE) attention modules
  - Hard swish activation functions
- **Training strategy:** Multi-stage progressive training with correlation-aware loss (MSE + Pearson correlation regularizers)

**Key advantage:** Specifically designed for resource-constrained devices while maintaining perceptual alignment with human judgments.

### 3. EfficientNet Variants

- **EfficientNetV2-S:** Achieves 96.53% accuracy and 0.9607 F1 score in lightweight model comparisons
- **Compound scaling:** Systematically coordinates depth, width, and resolution for Pareto-optimal performance
- **Status:** More complex than 3-5 layers, but provides architectural insights

---

## Architecture Design Strategies

### Strategy 1: Knowledge Distillation from LPIPS Teachers

**Approach:** Train a 3-5 layer "student" CNN to mimic the perceptual distance outputs of a pre-trained LPIPS "teacher" model.

**Recent advances:**
- **Teacher-Student Collaborative Knowledge Distillation (TSKD):** Student learns from both final outputs and intermediate layers of teacher
- **Feature-based distillation:** Transfer knowledge from intermediate activations, not just final predictions
- **Common teachers:** VGG19, AlexNet, ResNext architectures
- **Medical imaging success:** Lightweight students with fewer deep layers successfully distilled from VGG and ResNext teachers

**Training protocol:**
1. Use pre-trained LPIPS (VGG or AlexNet backbone) as teacher
2. Design compact student (3-5 conv layers + attention)
3. Loss function: MSE between student and teacher perceptual distances
4. Optionally: Add intermediate layer distillation for richer feature learning
5. Fine-tune on BAPPS 2AFC task with human judgments

**Expected correlation:** Studies show VGG19→VGG13 distillation maintains strong performance, suggesting VGG→Minimal-CNN could achieve >0.8 correlation.

### Strategy 2: Multi-Resolution Feature Extraction

**Inspired by MR-Perceptual (Multi-Resolution Perceptual) metric:**

Design a shallow network that extracts features at multiple resolutions:

```
Layer 1: Conv 7x7, stride 2 → downsample
Layer 2: Conv 5x5, stride 1 → main features
Layer 3: Conv 3x3, stride 1 → fine details
Layer 4: Conv 3x3, stride 2 → downsample
Layer 5: Conv 1x1 → channel-wise weighting
```

**Rationale:** Multi-resolution processing compensates for shallow depth by capturing both coarse structure and fine texture.

**Enhancement:** Add lightweight attention modules (SE blocks, ~2 additional layers) to focus on perceptually salient regions.

### Strategy 3: Structure + Texture Decomposition (DISTS-inspired)

**Based on DISTS (Deep Image Structure and Texture Similarity):**

DISTS achieves excellent perceptual quality assessment by explicitly separating structure and texture:

- **Architecture:** VGG16 with 6 stages (3, 64, 128, 256, 512, 512 channels)
- **Two-term measurement:**
  1. Spatial averages comparison (texture properties)
  2. Structural details comparison
- **Final score:** Weighted sum optimized to match human ratings

**Minimal adaptation (3-5 layers):**

```
Layer 1: Conv 7x7, 64 channels → initial feature extraction
Layer 2: Conv 5x5, 128 channels → structure pathway
Layer 3: Conv 3x3, 128 channels → texture pathway (depthwise separable)
Layer 4: Global average pool → texture statistics
Layer 5: Structure-texture fusion → final perceptual distance
```

**Key innovation:** Use depthwise-separable convolutions (MobileNet-style) to keep parameters minimal while maintaining separate structure/texture pathways.

### Strategy 4: Neural Architecture Search (NAS) with Perceptual Objectives

**Modern NAS for perceptual metrics:**

Recent research (2025) shows NAS is increasingly used to discover architectures optimized for perceptual metrics rather than pixel-level metrics:

- **Optimization targets:** LPIPS, NIQE (No-Reference Image Quality), PI (Perceptual Index)
- **Search space:** Encoder-decoder architectures with perceptual metric-guided evolution
- **Multi-objective:** Balance perceptual quality, model size, and inference latency
- **Success cases:** NAS for Deep Image Prior (DIP) with perceptual metrics consistently improves visual quality

**Practical approach for 3-5 layers:**

1. **Define search space:**
   - Layer types: standard conv, depthwise conv, SE attention
   - Kernel sizes: 3x3, 5x5, 7x7
   - Channel counts: 32, 64, 128, 256
   - Activation functions: ReLU, Swish, Hard Swish

2. **Fitness function:**
   - Primary: Spearman correlation with LPIPS-VGG on BAPPS dataset
   - Secondary: Parameter count penalty (target <1M params)
   - Tertiary: Inference time (target <5ms on GPU)

3. **Evolution strategy:**
   - Start with SqueezeNet Fire module as initial population
   - Progressively remove layers and optimize remaining architecture
   - Use evolutionary algorithms or reinforcement learning for search

4. **Pareto optimization:** Find architectures on the Pareto frontier of correlation vs. efficiency

---

## Proposed Minimal Architectures

### Architecture A: Attention-Enhanced Compact CNN (AECC)

**Total layers: 5 (3 conv + 2 attention)**

```
1. Conv 7x7, 64 channels, stride 2, ReLU
2. Conv 5x5, 128 channels, stride 1, ReLU
3. Conv 3x3, 128 channels, stride 1, ReLU
4. Squeeze-Excitation block (channel attention)
5. Conv 1x1, 256 channels → perceptual embedding
```

**Features:**
- Multi-scale receptive fields (7x7 → 5x5 → 3x3)
- SE attention focuses on perceptually important channels
- Final 1x1 conv projects to learned perceptual space
- Parameters: ~500K (estimate)

**Training:**
- Initialize with ImageNet pre-training (transfer from MobileNetV3-Small)
- Distill from LPIPS-AlexNet teacher
- Fine-tune on BAPPS 2AFC task

### Architecture B: Structure-Texture Minimal Network (STMN)

**Total layers: 5 (parallel structure/texture pathways)**

```
1. Conv 7x7, 64 channels, stride 2 → shared stem
   ├─ 2a. Conv 5x5, 128 channels → structure pathway
   └─ 2b. Depthwise Conv 5x5, 128 channels → texture pathway
3. Concatenate [structure, texture] → 256 channels
4. Conv 1x1, 128 channels → fusion
5. Global pooling + linear → perceptual distance
```

**Features:**
- Explicit structure/texture decomposition (DISTS-inspired)
- Depthwise convolutions for efficient texture modeling
- Lightweight fusion layer
- Parameters: ~400K (estimate)

**Training:**
- Two-stage: (1) structure/texture pathways separately, (2) joint fusion
- Loss: MSE with LPIPS + correlation-aware regularization
- Data augmentation: texture resampling, structural perturbations

### Architecture C: Ultra-Minimal Distilled Network (UMDN)

**Total layers: 3 (absolute minimum)**

```
1. Conv 9x9, 128 channels, stride 2, Hard Swish
2. Depthwise Conv 7x7, 128 channels, stride 1, Hard Swish
3. Conv 1x1, 256 channels → perceptual embedding
```

**Features:**
- Large kernels compensate for shallow depth (9x9, 7x7)
- Depthwise convolution for parameter efficiency
- Hard Swish activation (proven in MobileNetV3)
- Parameters: ~250K (estimate)

**Training:**
- Aggressive knowledge distillation from VGG-LPIPS
- Intermediate layer matching from teacher's early layers
- BAPPS 2AFC fine-tuning with hard negative mining

**Risk:** May struggle to reach >0.8 correlation with only 3 layers, but worth testing as lower bound.

---

## Training Protocols

### Phase 1: Pre-training

**Option A: ImageNet Transfer**
- Use MobileNetV3-Small or SqueezeNet as initialization
- Prune to target 3-5 layer depth
- Fine-tune remaining layers

**Option B: Self-Supervised Pre-training**
- SimCLR or MoCo on diverse image datasets
- Learn perceptually meaningful features without labels
- Research shows unsupervised features work well for LPIPS

### Phase 2: Knowledge Distillation

**Teacher:** LPIPS with AlexNet or VGG backbone

**Student:** Target minimal architecture (3-5 layers)

**Loss function:**
```
L_total = L_distill + λ_corr * L_correlation + λ_feat * L_feature

Where:
- L_distill = MSE(student_distance, teacher_distance)
- L_correlation = 1 - PearsonCorr(student, teacher)
- L_feature = MSE(student_features, teacher_features) [optional]
```

**Hyperparameters:**
- λ_corr = 0.1 (correlation regularizer)
- λ_feat = 0.05 (feature matching, optional)
- Learning rate: 1e-4 with cosine annealing
- Batch size: 64-128 image pairs
- Epochs: 50-100

**Data:**
- Training: Large-scale distortion datasets (ImageNet perturbations, generative model outputs)
- Validation: BAPPS validation split
- Testing: BAPPS test split + additional perceptual datasets (TID2013, KADID-10k)

### Phase 3: Fine-tuning on Human Judgments

**Dataset:** BAPPS 2AFC task (human preference labels)

**Loss:** Binary cross-entropy on 2AFC predictions
```
L_2AFC = -log(P(human_choice | student_distances))
```

**Technique:** Hard negative mining
- Focus on triplets where student disagrees with humans
- Emphasize edge cases where perceptual distance is subtle

**Augmentation:**
- Texture resampling (following DISTS tolerance for texture)
- Mild geometric transforms (rotation ±5°, scaling 0.95-1.05)
- Colorspace perturbations (HSV jitter)

---

## Evaluation Metrics

### Primary: Correlation with LPIPS-VGG

**Target:** >0.8 Spearman/Pearson correlation on held-out test set

**Datasets:**
- BAPPS test split (official benchmark)
- TID2013 (2,500 distorted images, 25 distortion types)
- KADID-10k (10,125 distorted images, 25 distortion types)
- PieAPP dataset (perceptual image quality)

### Secondary: Human Agreement

**Target:** >65% on BAPPS 2AFC (recall human cross-rater agreement is 73.9%)

### Efficiency Metrics

**Parameters:** <1M (ideally <500K)

**Inference time:** <5ms on NVIDIA RTX GPU, <50ms on CPU

**Model size:** <5 MB (for deployment in real-time applications like shader synthesis)

### Robustness Tests

- **Adversarial perturbations:** R-LPIPS shows LPIPS can be vulnerable; test student robustness
- **Texture resampling:** DISTS-style tolerance (should not penalize texture resampling heavily)
- **Cross-domain generalization:** Test on artistic images, medical images, synthetic renders

---

## Connections to Related Research

### 1. Real-Time Diffusion Models

This research directly supports the "path to real-time diffusion models" goal (research queue priority 5). A fast perceptual loss function is critical for:
- Interactive texture synthesis (shader-based)
- One-step diffusion model guidance
- Real-time style transfer

A 3-5 layer perceptual network with <5ms inference enables tight feedback loops for generative models.

### 2. Neural Cellular Automata (NCA)

Minimal perceptual metrics are valuable for:
- NCA training loss functions (texture synthesis objective)
- Fine-tuning evaluation (measuring perceptual similarity between NCA outputs and targets)
- Zero-shot transfer benchmarking (quantifying cross-domain perceptual similarity)

NCAs benefit from lightweight metrics since training involves thousands of forward passes.

### 3. Knowledge Distillation for Generative Models

The teacher-student distillation framework used here mirrors:
- GAN compression techniques
- Diffusion model distillation (progressive distillation, consistency models)
- Perceptual loss networks for super-resolution (Johnson et al.)

### 4. Neural Architecture Search Trends

This work aligns with modern NAS emphasis on:
- Pareto optimization (performance vs. efficiency)
- Perceptual metrics as optimization objectives (beyond PSNR/SSIM)
- Hardware-aware search (latency, energy constraints)

---

## Follow-Up Research Questions

### Question 1: How far can we push minimal architectures?

Can a 2-layer or even 1-layer CNN with sophisticated attention/pooling mechanisms achieve >0.7 correlation with LPIPS?

**Hypothesis:** There's a lower bound (likely 3-4 layers) below which perceptual understanding fundamentally breaks down. But the boundary is unclear.

**Experiment:** Systematically ablate layers from SqueezeNet-LPIPS and measure degradation curve.

### Question 2: Does domain-specific training improve minimal networks?

For shader texture synthesis specifically, can we train a minimal network on *texture* images only and achieve >0.8 correlation on texture-specific perceptual judgments?

**Hypothesis:** Domain-specific training (textures, not general ImageNet) could allow even smaller networks for specialized tasks.

**Experiment:** Train minimal architectures on DescribableTextures dataset (DTD) with texture-specific distortions.

### Question 3: Can linear/non-parametric methods compete?

LASI (Linear Prediction-based method) claims competitive performance with LPIPS *without* neural networks. Is this a viable minimal approach?

**Hypothesis:** LASI might achieve >0.75 correlation but plateau below 0.8 due to lack of learned non-linearities.

**Experiment:** Benchmark LASI on BAPPS and compare to minimal CNN approaches.

### Question 4: Multi-task learning for better efficiency?

Can we train a single minimal network that simultaneously:
- Predicts LPIPS perceptual distance
- Classifies distortion types
- Estimates image quality scores (NIQE, BRISQUE)

**Hypothesis:** Multi-task learning provides richer feature representations, potentially improving correlation with limited capacity.

**Experiment:** Add auxiliary heads to minimal architectures and measure correlation improvements.

### Question 5: What about interpretability?

Can we design minimal networks where each layer's contribution to perceptual distance is interpretable (structure vs. texture vs. color)?

**Hypothesis:** Explicit structure-texture decomposition (DISTS-style) in minimal networks could provide both interpretability and performance.

**Experiment:** Ablate structure/texture pathways in STMN architecture and analyze contribution to BAPPS predictions.

---

## Implementation Roadmap

### Phase 1: Baseline Reproduction (Week 1-2)
1. Reproduce LPIPS-AlexNet and LPIPS-SqueezeNet from official library
2. Benchmark on BAPPS dataset (validate correlation scores)
3. Establish ground truth teacher models for distillation

### Phase 2: Architecture Design (Week 3-4)
1. Implement three proposed architectures (AECC, STMN, UMDN)
2. Pre-training with ImageNet transfer or self-supervised learning
3. Initial forward passes to verify computational efficiency (<5ms target)

### Phase 3: Knowledge Distillation (Week 5-8)
1. Generate training data: image pairs with LPIPS-VGG perceptual distances
2. Train minimal students with distillation loss (MSE + correlation regularization)
3. Hyperparameter tuning: learning rate, λ_corr, augmentation strategies
4. Monitor validation correlation with LPIPS teacher

### Phase 4: BAPPS Fine-Tuning (Week 9-10)
1. Fine-tune on BAPPS 2AFC human judgment task
2. Hard negative mining for edge cases
3. Evaluate on BAPPS test split: 2AFC accuracy and correlation with LPIPS

### Phase 5: Evaluation & Ablation (Week 11-12)
1. Benchmark on additional datasets (TID2013, KADID-10k, PieAPP)
2. Robustness tests: adversarial attacks, texture resampling, cross-domain
3. Ablation studies: layer depth, attention mechanisms, kernel sizes
4. Efficiency profiling: parameters, FLOPs, latency (GPU/CPU)

### Phase 6: NAS Exploration (Optional, Week 13-16)
1. Define search space based on insights from manual architectures
2. Run evolutionary search or RL-based NAS with BAPPS correlation as fitness
3. Discover novel architectures on Pareto frontier
4. Compare NAS-discovered vs. hand-designed architectures

### Phase 7: Deployment & Integration (Week 17+)
1. Export minimal network to ONNX/TorchScript for real-time inference
2. Integrate into shader texture synthesis pipeline
3. Test in interactive application: measure perceptual quality improvements
4. Open-source release: model weights, training code, evaluation scripts

---

## Expected Outcomes

### Conservative Estimate (Likely Achievable)
- **5-layer architecture:** >0.80 Spearman correlation with LPIPS-VGG on BAPPS
- **Parameters:** ~500K
- **Inference time:** ~3ms on RTX GPU
- **BAPPS 2AFC accuracy:** ~65% (vs. 73.9% human agreement)

### Optimistic Estimate (Stretch Goal)
- **4-layer architecture:** >0.85 Spearman correlation with LPIPS-VGG
- **Parameters:** ~300K
- **Inference time:** ~2ms on RTX GPU
- **BAPPS 2AFC accuracy:** ~68% (approaching human agreement)

### Pessimistic Estimate (Lower Bound)
- **5-layer architecture:** ~0.75 Spearman correlation
- **3-layer architecture:** ~0.70 Spearman correlation (below target)
- **Trade-off accepted:** Slightly lower correlation for significant speed gains

**Key insight:** Even if 3 layers fall short of >0.8 correlation, the research will establish *where the performance cliff is*, informing future minimal architecture design.

---

## Conclusion

Creating minimal perceptual networks that maintain >0.8 correlation with LPIPS is a **feasible but non-trivial** challenge. The literature provides strong evidence that:

1. **SqueezeNet (2.8 MB) already achieves competitive performance** → Size reduction is possible
2. **Knowledge distillation from LPIPS teachers is a proven strategy** → Teacher-student framework is sound
3. **Multi-resolution + attention mechanisms compensate for shallow depth** → Architectural innovations can bridge gap
4. **NAS with perceptual objectives is an emerging frontier** → Automated search could discover novel designs

The proposed architectures (AECC, STMN, UMDN) leverage insights from recent research on lightweight models, structure-texture decomposition, and attention mechanisms. With systematic training (pre-training → distillation → fine-tuning on human judgments), achieving >0.8 correlation with 5 layers is realistic. Pushing to 3-4 layers is more speculative but worth exploring to find the fundamental limits.

This research directly enables real-time applications (shader synthesis, interactive diffusion models, NCA training) by providing fast, accurate perceptual feedback. The 18-week roadmap provides a clear path from reproduction to deployment, with concrete milestones and evaluation metrics.

**Next steps:** Implement baseline LPIPS models, design minimal architectures, and begin knowledge distillation experiments.

---

## Sources

### Foundational Papers & Resources

1. [GitHub - richzhang/PerceptualSimilarity: LPIPS metric](https://github.com/richzhang/PerceptualSimilarity) - Official LPIPS implementation
2. [LPIPS PyTorch-Metrics Documentation](https://lightning.ai/docs/torchmetrics/stable/image/learned_perceptual_image_patch_similarity.html) - PyTorch implementation
3. [The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (CVPR 2018)](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_The_Unreasonable_Effectiveness_CVPR_2018_paper.pdf) - Original LPIPS paper
4. [Berkeley-Adobe Perceptual Patch Similarity (BAPPS) Dataset](https://www.kaggle.com/datasets/chaitanyakohli678/berkeley-adobe-perceptual-patch-similarity-bapps)

### Lightweight Perceptual Networks

5. [THE UNREASONABLE EFFECTIVENESS OF LINEAR PREDICTION AS A PERCEPTUAL METRIC](https://openreview.net/pdf?id=e4FG5PJ9uC) - LASI method
6. [A study of deep perceptual metrics for image quality assessment (arXiv 2202.08692)](https://arxiv.org/abs/2202.08692) - Multi-resolution perceptual metrics
7. [Deep Ensembling for Perceptual Image Quality Assessment](https://arxiv.org/pdf/2305.09141) - Ensemble approaches
8. [Can perceptual similarity metrics be used to compare adversarial attacks?](https://transferlab.ai/blog/perceptual-metrics/) - SqueezeNet comparison

### Knowledge Distillation

9. [Efficient image classification through collaborative knowledge distillation: A novel AlexNet modification approach](https://pmc.ncbi.nlm.nih.gov/articles/PMC11305255/) - Teacher-student distillation
10. [Teacher–student knowledge distillation based on decomposed deep feature representation](https://www.sciencedirect.com/science/article/abs/pii/S0957417422008053) - Feature-based distillation
11. [Simplified Knowledge Distillation for Deep Neural Networks](https://www.mdpi.com/2079-9292/13/22/4530) - Novel architectures
12. [Adaptive multi-teacher multi-level knowledge distillation](https://www.sciencedirect.com/science/article/abs/pii/S0925231220311565) - Multi-teacher approaches

### Lightweight Architectures (MobileNet, EfficientNet)

13. [A Lightweight Ensemble-Based Face Image Quality Assessment Method](https://arxiv.org/html/2509.10114) - MobileNetV3-Small + ShuffleNetV2
14. [MSPT: A Lightweight Face Image Quality Assessment Method with Multi-stage Progressive Training](https://arxiv.org/html/2508.07590) - MobileNetV3-Small <5M params
15. [Comparative Analysis of Lightweight Deep Learning Models](https://arxiv.org/html/2505.03303v1) - EfficientNetV2 benchmarks
16. [Efficient Super-Resolution Using MobileNetV3](https://vinbhaskara.github.io/files/eccvw20-pdf.pdf) - Perceptual loss with MobileNet
17. [A lightweight approach for image quality assessment](https://link.springer.com/article/10.1007/s11760-024-03349-0) - GhostDPD module

### Neural Architecture Search

18. [A review of neural architecture search methods for super-resolution imaging](https://link.springer.com/article/10.1007/s10462-025-11488-0) - NAS with perceptual metrics
19. [Neural architecture search for deep image prior](https://www.sciencedirect.com/science/article/abs/pii/S0097849321001126) - Perceptual metric-guided evolution
20. [Neural Architecture Search: A Survey (JMLR)](https://jmlr.org/papers/volume20/18-598/18-598.pdf) - Comprehensive NAS overview

### Structure-Texture Similarity (DISTS)

21. [GitHub - dingkeyan93/DISTS: IQA: Deep Image Structure and Texture Similarity Metric](https://github.com/dingkeyan93/DISTS) - Official DISTS implementation
22. [Deep Image Structure And Texture Similarity (DISTS) PyTorch-Metrics](https://lightning.ai/docs/torchmetrics/stable/image/dists.html) - Documentation
23. [Image Quality Assessment: Unifying Structure and Texture Similarity (arXiv 2004.07728)](https://ar5iv.labs.arxiv.org/html/2004.07728) - DISTS paper
24. [Locally Adaptive Structure and Texture Similarity (arXiv 2110.08521)](https://arxiv.org/abs/2110.08521) - LAST-IQA variant

### Attention Mechanisms for Perceptual Quality

25. [An LCD Defect Image Generation Model Integrating Attention Mechanism and Perceptual Loss](https://www.mdpi.com/2073-8994/17/6/833) - VGG-19 with attention
26. [Edge-Aware Normalized Attention for Efficient Super-Resolution](https://arxiv.org/html/2509.14550v1) - Lightweight attention design
27. [Deep Learning Image Compression Method Based On Efficient Channel-Time Attention Module](https://www.nature.com/articles/s41598-025-00566-6) - ETAM method
28. [Image Inpainting Using Lightweight Transformer Neural Network Based on Channel Attention](https://dl.acm.org/doi/10.1145/3638837.3638876) - Attention-based inpainting

### BAPPS Benchmark & Human Perception

29. [Shift-tolerant Perceptual Similarity Metric (ECCV 2022)](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136780089.pdf) - BAPPS analysis
30. [Scene Perceived Image Perceptual Score (SPIPS)](https://arxiv.org/html/2504.17234) - Global + local perception
31. [An Unsupervised Information-Theoretic Perceptual Quality Metric (NeurIPS 2020)](https://proceedings.nips.cc/paper/2020/file/00482b9bed15a272730fcb590ffebddd-Paper.pdf) - PIM method

### Additional Relevant Research

32. [R-LPIPS: An Adversarially Robust Perceptual Similarity Metric](https://openreview.net/pdf/010bd9e426c2c4e83d9e213bf2fd87bba87c172f.pdf) - Robustness testing
33. [Training a Task-Specific Image Reconstruction Loss (arXiv 2103.14616)](https://ar5iv.labs.arxiv.org/html/2103.14616) - MDF loss
34. [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://www.researchgate.net/publication/308278061_Perceptual_Losses_for_Real-Time_Style_Transfer_and_Super-Resolution) - Johnson et al. foundational work
35. [A Systematic Performance Analysis of Deep Perceptual Loss Networks (arXiv 2302.04032)](https://arxiv.org/pdf/2302.04032) - Breaking transfer learning conventions

---

**Research completed by:** research-agent
**Total sources consulted:** 35 (10 web searches, comprehensive literature review)
**Next recommended follow-up:** Implement baseline LPIPS reproduction and begin architecture prototyping
