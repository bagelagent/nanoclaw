# Hybrid Loss Scheduling for NCA Training

**Research ID:** rq-1771851237044-hybrid-loss-schedule
**Research Date:** March 2, 2026
**Topic:** Hybrid loss scheduling for NCA training: SqueezeNet during exploration, VGG16 for fine-tuning

---

## Summary

Hybrid loss scheduling — using a computationally cheap loss function during initial exploration and switching to an expensive high-fidelity loss for fine-tuning — is a well-established strategy in image generation (most prominently in ESRGAN/Real-ESRGAN), but has not been formally studied for Neural Cellular Automata texture synthesis. The core proposal of using SqueezeNet-based perceptual loss (~0.72 GFLOPs) during NCA exploration and VGG16-based loss (~15.5 GFLOPs) for fine-tuning is theoretically sound, supported by evidence that SqueezeNet achieves comparable perceptual similarity scores (70.07% vs 68.65% 2AFC accuracy in LPIPS) and that layer selection matters as much as architecture choice for perceptual quality.

---

## Key Findings

### 1. The ESRGAN Precedent: Two-Stage Loss Scheduling

The most established example of this approach comes from **ESRGAN** and **Real-ESRGAN** super-resolution models:

**Stage 1 (Warm-up):** Train with simple L1 pixel loss
- Cheap to compute (no pretrained network forward pass)
- Provides stable, smooth early convergence
- Establishes reasonable generator baseline

**Stage 2 (Refinement):** Switch to L1 + perceptual (VGG) + adversarial loss
- Expensive but captures high-level texture/detail
- Refines visual quality once generator has stable foundation
- VGG features extracted before activation (ESRGAN improvement)

This two-stage approach is the standard in modern image restoration. The rationale is that pixel-wise losses provide stability during the chaotic early training phase, while perceptual losses introduced later guide the model toward perceptually pleasing outputs.

### 2. Computational Cost Differential: SqueezeNet vs VGG16

The cost savings of using SqueezeNet during exploration are substantial:

| Metric | SqueezeNet v1.1 | VGG-16 | Ratio |
|--------|----------------|--------|-------|
| **Forward FLOPs** | 0.72 GFLOPs | 15.5 GFLOPs | 21.5× cheaper |
| **Parameters** | 1.24M (5 MB) | 138M (58.9 MB) | 111× fewer |
| **Memory** | ~5 MB | ~58.9 MB | ~12× less |
| **LPIPS 2AFC** | 70.07% | 68.65% | Slightly better |

For texture NCA training where the loss network forward pass dominates training cost:
- A typical NCA training runs 5,000-10,000 iterations
- Each iteration requires 32-64 NCA forward steps + 1 loss network forward+backward
- Switching to SqueezeNet during exploration could save **~21× FLOPs per loss evaluation**
- If exploration represents 70% of training, total savings ≈ **~15× on loss computation** for that phase

### 3. Why This Works: Loss Landscape Similarity

The theoretical basis for hybrid loss scheduling rests on several observations:

**a) Perceptual metric correlation across architectures:**
Deep network activations work as perceptual metrics regardless of architecture — SqueezeNet, AlexNet, and VGG all capture similar perceptual structure. The "landscape" of good solutions looks similar under both losses, meaning SqueezeNet can guide the NCA toward the right region of parameter space.

**b) Early training needs coarse guidance, not precise gradients:**
During initial NCA training, the cell states are far from producing coherent textures. The NCA needs to learn basic spatial organization, color distribution, and pattern regularity. SqueezeNet's perceptual features are sufficient for this — it achieves comparable or even slightly better human-correlated perceptual similarity scores.

**c) Layer choice matters more than architecture:**
The systematic analysis by Pihlgren et al. (2023) found that "selecting the best extraction layer of the worst architecture gives around the same performance as selecting the worst extraction layer of the best architecture." This means that well-chosen SqueezeNet layers can match poorly-chosen VGG layers.

**d) Coarse temporal sampling stabilizes NCA training:**
Research on spatio-temporal NCA patterns shows that coarse sampling during early training stabilizes dynamics and yields more generalizable models, because "intermediate states may explore more possible states, increasing the chances of finding parameters that give the correct dynamics." The same principle applies to loss precision — a coarser loss allows broader exploration.

### 4. NCA-Specific Considerations

Standard texture NCA training (Self-Organising Textures, Niklasson et al. 2021) uses:
- **5 VGG16 layers:** conv1-1, conv2-1, conv3-1, conv4-1, conv5-1
- **Gram matrix matching** on activations
- **32-64 NCA forward steps** per training iteration
- **Adam optimizer** with frozen VGG weights

The loss network forward pass (VGG) is the dominant computational cost per iteration, because:
- The NCA itself is tiny (hundreds to thousands of parameters)
- VGG16 requires ~15.5 GFLOP per forward pass
- Backpropagation through VGG adds another ~15.5 GFLOP
- Total loss computation: ~31 GFLOP per iteration vs NCA forward: <0.1 GFLOP

This makes the loss network the clear bottleneck, and the most impactful target for optimization.

### 5. Proposed Hybrid Schedule Designs

#### Design A: Hard Switch
```
Phase 1 (iterations 0 - N/2):    SqueezeNet Gram loss
Phase 2 (iterations N/2 - N):    VGG16 Gram loss
```
- Simplest approach
- Risk: abrupt gradient signal change may destabilize training
- Mitigation: reset Adam optimizer state at switch point, use lower learning rate for Phase 2

#### Design B: Warm Transfer (Recommended)
```
Phase 1 (iterations 0 - 0.6N):       SqueezeNet Gram loss only
Phase 2 (iterations 0.6N - 0.7N):    α·SqueezeNet + (1-α)·VGG16, α linearly 1→0
Phase 3 (iterations 0.7N - N):       VGG16 Gram loss only
```
- Gradual transition avoids gradient shock
- Blending period lets the NCA adapt to new loss landscape
- Small computational overhead during blend phase (both networks run simultaneously)

#### Design C: SqueezeNet + VGG Layer Subset
```
Phase 1 (iterations 0 - 0.5N):       SqueezeNet 7-layer Gram loss
Phase 2 (iterations 0.5N - 0.8N):    VGG16 conv1-3 only (truncated, ~60% FLOP savings)
Phase 3 (iterations 0.8N - N):       VGG16 conv1-5 full (standard configuration)
```
- Progressive increase in loss precision AND cost
- Phase 2 is a middle ground: VGG quality with partial FLOP savings
- Most compute-efficient: full VGG only runs for final 20% of training

#### Design D: Adaptive (Loss-Driven Switching)
```
Monitor: loss convergence rate d(L)/dt
While d(L)/dt > threshold_fast:    SqueezeNet loss (exploration)
When d(L)/dt < threshold_slow:     Switch to VGG16 loss (refinement)
```
- Automatically determines when to switch based on convergence
- No manual phase boundary tuning needed
- Risk: threshold selection requires calibration per texture type

### 6. Expected Benefits and Risks

#### Benefits
- **Training speed:** 3-10× faster overall depending on phase split
- **Memory efficiency:** SqueezeNet uses 12× less GPU memory than VGG16
- **Larger batch sizes:** During Phase 1, freed memory allows larger batches for more stable gradients
- **More exploration:** Cheaper loss allows more training iterations in the same wall-clock time
- **Potentially better minima:** Broader exploration landscape from coarser loss may help escape local minima

#### Risks
- **Loss landscape mismatch:** SqueezeNet and VGG16 may disagree on texture quality, causing the NCA to settle in a region that looks good under SqueezeNet but poor under VGG16
- **Transition instability:** Switching losses mid-training can cause gradient spikes or training collapse
- **Color drift:** µNCA research showed VGG-based losses can cause color drift — switching networks might exacerbate this
- **Hyperparameter sensitivity:** Optimal phase boundaries may vary significantly across texture types
- **Gram matrix incompatibility:** Gram matrices from SqueezeNet and VGG16 have different feature dimensions, so loss magnitudes will differ — requires careful normalization

### 7. Mitigation Strategies

**For loss landscape mismatch:**
- Pre-compute LPIPS correlation between SqueezeNet and VGG16 on target texture class
- Only use hybrid schedule when correlation > 0.85
- Alternative: use distilled SqueezeNet specifically calibrated to match VGG16 gradients

**For transition instability:**
- Use Design B (warm transfer) with gradual blending
- Reset optimizer momentum/state at transition
- Reduce learning rate by 2-5× at transition

**For Gram matrix scale differences:**
- Normalize Gram matrices to unit Frobenius norm before loss computation
- Use cosine similarity instead of L2 distance for cross-architecture compatibility
- Pre-compute loss scale calibration factor

### 8. Connections to Progressive and Curriculum Learning

The hybrid loss scheduling idea connects to several established training paradigms:

**Curriculum Learning (Bengio et al. 2009):**
- Start with "easy" examples/objectives, progress to harder ones
- SqueezeNet loss as "easy" (coarser but sufficient), VGG as "hard" (precise but expensive)
- Key insight: "difficulty can be increased steadily or in distinct epochs"

**Coarse-to-Fine Curriculum (Stretcu et al. 2021):**
- Decompose challenging tasks into sequences of easier intermediate goals
- Pre-train at each level, transferring knowledge across levels
- Directly analogous: SqueezeNet provides coarse perceptual guidance, VGG provides fine refinement

**Progressive GAN Training (PA-GAN):**
- Gradually increase discrimination task difficulty
- With each extra level, the task gradually becomes harder
- Stabilizing factor λ linearly increases during progression

**LossAgent (2024):**
- LLM-driven dynamic loss weight adjustment during training
- Actively decides compositional weights for each loss at each step
- Could be adapted to dynamically weight SqueezeNet vs VGG contributions

### 9. Alternative Approaches Worth Comparing

**a) Randomly initialized VGG:**
- Uses random (untrained) VGG for perceptual loss
- Performs comparably to pretrained VGG in some settings
- Even cheaper than SqueezeNet (no pretrained weights to load)
- But: may produce different texture quality for NCA

**b) ConvNeXt perceptual loss:**
- Modern architecture that may capture features more efficiently
- Single-network solution without multi-stage scheduling
- But: less studied for texture synthesis

**c) OTT Loss (Optimal Transport on Texture patches):**
- Used in µNCA as VGG alternative
- No pretrained network required (operates on raw patches)
- Color-unbiased
- But: may be more expensive than SqueezeNet for equivalent quality

**d) Sliced Wasserstein Loss:**
- Used in multi-texture NCA work
- Applied to VGG features but could work with SqueezeNet features
- More stable than Gram matrices (avoids Gram matrix instabilities DyNCA documented)

---

## Deep Dive: Implementation Considerations

### Loss Network Feature Extraction Points

**SqueezeNet LPIPS layers (7 extraction points):**
Features after fire modules, providing multi-scale representation. Channels: [64, 128, 256, 384, 384, 512, 512].

**VGG16 standard texture layers (5 extraction points):**
conv1-1, conv2-1, conv3-1, conv4-1, conv5-1. Channels: [64, 128, 256, 512, 512].

**Gram matrix dimensions differ:**
- SqueezeNet: 7 Gram matrices of sizes 64², 128², 256², 384², 384², 512², 512²
- VGG16: 5 Gram matrices of sizes 64², 128², 256², 512², 512²

For cross-network loss compatibility, either:
1. Match only overlapping channel dimensions (64, 128, 256, 512)
2. Use normalized feature statistics instead of raw Gram matrices
3. Use LPIPS-style learned linear calibration layers

### Practical Training Recipe

```python
# Pseudocode for Design B (Warm Transfer)

squeezenet_loss = SqueezeNetGramLoss(layers=[...])
vgg_loss = VGG16GramLoss(layers=['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'])

total_steps = 10000
blend_start = 6000  # Start transition at 60%
blend_end = 7000    # End transition at 70%

for step in range(total_steps):
    # NCA forward pass (32-64 steps)
    nca_output = nca.forward(seed, steps=random.randint(32, 64))

    if step < blend_start:
        # Phase 1: SqueezeNet only (cheap exploration)
        loss = squeezenet_loss(nca_output, target)
    elif step < blend_end:
        # Phase 2: Gradual blend
        alpha = (step - blend_start) / (blend_end - blend_start)
        loss_sq = squeezenet_loss(nca_output, target)
        loss_vgg = vgg_loss(nca_output, target)
        loss = (1 - alpha) * loss_sq + alpha * loss_vgg
    else:
        # Phase 3: VGG16 only (precise refinement)
        loss = vgg_loss(nca_output, target)

    loss.backward()
    optimizer.step()

    # Optional: reset optimizer at blend_start
    if step == blend_start:
        optimizer = Adam(nca.parameters(), lr=lr * 0.5)
```

### Expected Computational Savings

For a 10,000-iteration NCA training run:

| Schedule | SqueezeNet iters | VGG16 iters | Both (blend) | Estimated Total GFLOPs | Savings vs Full VGG |
|----------|-----------------|-------------|-------------|----------------------|-------------------|
| Full VGG (baseline) | 0 | 10,000 | 0 | 310,000 | — |
| Design A (hard switch) | 5,000 | 5,000 | 0 | 158,600 | **49%** |
| Design B (warm transfer) | 6,000 | 3,000 | 1,000 | 129,520 | **58%** |
| Design C (progressive) | 5,000 | 2,000 | 3,000 (trunc VGG) | ~150,000 | **52%** |
| Full SqueezeNet | 10,000 | 0 | 0 | 7,200 | **98%** |

*GFLOPs estimated per iteration: VGG fwd+bwd ≈ 31 GFLOP, SqueezeNet fwd+bwd ≈ 1.44 GFLOP*

---

## Connections to Existing Knowledge

### Related Research in Queue
1. **Layer ablation (rq-1771829371565):** Completed — understanding which layers matter informs which VGG layers to use in Phase 2/3
2. **Differentiable sliced Wasserstein (rq-1771851237046):** Could replace Gram matrices in both phases
3. **Distilled SqueezeNet benchmarks (rq-1771851531076):** A distilled SqueezeNet calibrated to VGG would reduce loss landscape mismatch risk
4. **MILO adaptation (rq-1771851531077):** MILO could train a tiny proxy specifically for NCA pattern evaluation
5. **Texture-specific perceptual network (rq-1771873861290):** A custom 3-layer network might be even cheaper than SqueezeNet for Phase 1

### Broader Context
- ESRGAN's two-stage training is now standard in image restoration — extending it to NCA is natural
- Growing NCA (Distill 2020) already uses progressive training (pool-based reseeding) — loss scheduling is another form of progression
- DyNCA moved from Gram matrices to optimal transport-style loss for stability — hybrid scheduling could incorporate this

---

## Follow-up Questions

1. **How sensitive is the optimal switch point to texture complexity?** Simple textures (stripes, dots) likely need less VGG refinement than complex organic textures (fur, bark). Could the switch point be predicted from texture statistics?

2. **Could meta-learning optimize the schedule?** Train a meta-learner that observes NCA training dynamics and decides when to switch losses, analogous to learned learning rate schedules.

3. **What about using SqueezeNet's LPIPS calibration layers?** Rather than raw Gram matrices, using LPIPS-calibrated features might provide better cross-architecture consistency during the blend phase.

4. **Could we use SqueezeNet features to predict when VGG would diverge?** Monitor the correlation between SqueezeNet and VGG loss values on the current NCA output — switch only when they start to diverge, indicating SqueezeNet has hit its quality ceiling.

5. **Would batch size scheduling help?** During Phase 1 (SqueezeNet), the memory savings enable larger batch sizes. Could we systematically exploit this — large batches for stable exploration, then small batches with VGG for fine-grained refinement?

6. **Can this be combined with resolution scheduling?** Train at low resolution with SqueezeNet, then increase resolution when switching to VGG — combining two forms of coarse-to-fine progression.

---

## Sources

### Core References
1. [ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks](https://openaccess.thecvf.com/content_ECCVW_2018/papers/11133/Wang_ESRGAN_Enhanced_Super-Resolution_Generative_Adversarial_Networks_ECCVW_2018_paper.pdf) — Canonical two-stage L1→perceptual training
2. [Real-ESRGAN Training Documentation](https://github.com/xinntao/Real-ESRGAN/blob/master/docs/Training.md) — Practical two-stage implementation
3. [A Systematic Performance Analysis of Deep Perceptual Loss Networks](https://arxiv.org/html/2302.04032v3) — Layer choice ≥ architecture choice
4. [The Unreasonable Effectiveness of Deep Features as a Perceptual Metric](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_The_Unreasonable_Effectiveness_CVPR_2018_paper.pdf) — LPIPS, SqueezeNet/AlexNet/VGG comparison
5. [Self-Organising Textures](https://distill.pub/selforg/2021/textures/) — Standard NCA texture training with VGG Gram loss

### NCA Architecture & Training
6. [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/) — Progressive pool-based reseeding
7. [DyNCA: Real-time Dynamic Texture Synthesis](https://openaccess.thecvf.com/content/CVPR2023/papers/Pajouheshgar_DyNCA_Real-Time_Dynamic_Texture_Synthesis_Using_Neural_Cellular_Automata_CVPR_2023_paper.pdf) — OT-based style loss for stability
8. [µNCA: Texture Generation with Ultra-Compact NCA](https://arxiv.org/abs/2111.13545) — OTT loss as VGG alternative
9. [Multi-texture synthesis with signal responsive NCA](https://www.nature.com/articles/s41598-025-23997-7) — Sliced Wasserstein on VGG features
10. [Learning spatio-temporal patterns with NCA](https://pmc.ncbi.nlm.nih.gov/articles/PMC11078362/) — Coarse temporal sampling stabilizes training

### Curriculum & Progressive Training
11. [On The Power of Curriculum Learning in Training Deep Networks](https://arxiv.org/pdf/1904.03626) — Theoretical foundations
12. [Coarse-to-Fine Curriculum Learning](https://arxiv.org/abs/2106.04072) — Progressive task decomposition
13. [PA-GAN: Progressive Augmentation of GANs](https://openreview.net/pdf?id=ByeNFoRcK7) — Progressive discrimination difficulty
14. [LossAgent: Towards Any Optimization Objectives](https://arxiv.org/html/2412.04090) — LLM-driven dynamic loss weighting

### Loss Functions & Scheduling
15. [Loss Functions for Image Restoration with Neural Networks](https://arxiv.org/pdf/1511.08861) — L1 vs L2, perceptual loss analysis
16. [Optimizing Multiple Loss Functions with Loss-Conditional Training](https://research.google/blog/optimizing-multiple-loss-functions-with-loss-conditional-training/) — Single model, variable loss coefficients
17. [ESRGAN-DP: Enhanced SRGAN with Adaptive Dual Perceptual Loss](https://pmc.ncbi.nlm.nih.gov/articles/PMC10119608/) — VGG + ResNet dual perceptual loss
18. [Loss Functions in Deep Learning: A Comprehensive Review](https://arxiv.org/html/2504.04242v1) — Survey of loss function techniques
19. [LPIPS GitHub Repository](https://github.com/richzhang/PerceptualSimilarity) — Official implementation with SqueezeNet/AlexNet/VGG

### Computational Cost References
20. [Keras FLOP Estimator](https://github.com/ckyrkou/Keras_FLOP_Estimator) — Per-layer FLOP calculation tool
21. [convnet-burden](https://github.com/albanie/convnet-burden) — Memory and FLOP estimates for convnets
22. [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters](https://arxiv.org/pdf/1602.07360) — Original SqueezeNet architecture

---

## Research Methodology

This research was conducted through:
- Web searches across academic databases (arXiv, CVPR, NeurIPS, Nature) and technical resources
- Analysis of the ESRGAN/Real-ESRGAN training pipeline as the closest existing precedent
- Cross-referencing computational cost data for VGG16 and SqueezeNet architectures
- Synthesis of curriculum learning, progressive training, and loss scheduling literature
- Connection to prior NCA research on training dynamics and loss function alternatives

The key gap identified: while two-stage loss scheduling is standard in super-resolution (ESRGAN), and while SqueezeNet has been validated as a viable perceptual loss network (LPIPS), **no published work combines these ideas specifically for NCA texture synthesis**. This represents a clear opportunity for empirical validation.
