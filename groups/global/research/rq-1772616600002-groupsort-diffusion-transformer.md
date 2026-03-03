# GroupSort and MaxMin Activations in Diffusion Transformers

*Research completed: 2026-03-02*
*Topic: GroupSort and MaxMin activations in diffusion transformers — expressivity-robustness tradeoff for 1-Lipschitz video generation*

---

## Summary

GroupSort and MaxMin activations are gradient-norm-preserving nonlinearities that enable 1-Lipschitz neural networks to be universal function approximators — a property that standard activations like ReLU lose under Lipschitz constraints. While these activations have proven transformative for certified adversarial robustness (scaling to billion-parameter classifiers as of January 2026), their integration into diffusion transformers for video generation remains an open frontier. The key tension: strict 1-Lipschitz constraints provide provable stability guarantees (critical for temporal consistency in video), but current Lipschitz transformers suffer significant accuracy degradation at scale. Recent architectural innovations — Skip-DiT's spectral constraints, LipsFormer's CenterNorm attention, and spectral soft cap/hammer weight methods — suggest practical pathways for incorporating Lipschitz-aware design into video DiT architectures without requiring full 1-Lipschitz enforcement.

---

## Key Findings

### 1. GroupSort/MaxMin: The Activation That Saved Lipschitz Networks

The foundational insight from Anil et al. (ICML 2019) is deceptively simple but profound: for a 1-Lipschitz neural network to be expressive, each layer must *preserve gradient norms* during backpropagation. Standard activations (ReLU, sigmoid, tanh) fail this requirement — they are monotonic element-wise nonlinearities that cannot be both 1-Lipschitz and gradient-norm-preserving (GNP). The consequence: ReLU networks under Lipschitz constraints are forced into near-linearity, unable to represent even the absolute value function.

**GroupSort** solves this by operating on *groups* of activations rather than individual neurons:
- Split the pre-activation vector into groups of size *k*
- Sort each group in ascending order
- The operation is both 1-Lipschitz and gradient-norm-preserving

**Key variants:**
- **MaxMin** (k=2): Sorts pairs into (max, min). The most commonly used variant.
- **FullSort** (k=n): Sorts the entire input vector.
- Both are *equally expressive* — they can simulate each other without violating norm constraints.

**The universal approximation theorem:** Norm-constrained GroupSort networks can approximate *any* 1-Lipschitz function arbitrarily well on compact sets. This is a fundamental capability that ReLU networks under Lipschitz constraints provably lack.

### 2. The Expressivity-Robustness Tradeoff at Scale

The promise of 1-Lipschitz networks is clear: if every layer has Lipschitz constant ≤1, the entire network has Lipschitz constant ≤1, guaranteeing that small input perturbations produce at most equally small output changes. This provides certified adversarial robustness without any adversarial training.

**The scalability problem has been severe:**
- Early GroupSort networks (2019-2023) were limited to small CNNs on CIFAR-scale data
- Orthogonal weight constraints (Cayley transforms, BCOP, skew orthogonal convolutions) added computational overhead
- Training instability with low-precision arithmetic limited practical scaling

**LipNeXt (January 2026) broke the scaling barrier:**
- First constraint-free, convolution-free 1-Lipschitz architecture
- Scales to **1-2 billion parameters** on ImageNet
- Key innovations: manifold optimization directly on the orthogonal group, Spatial Shift Module (proving norm-preserving depthwise convolutions reduce to spatial shifts), β-Abs nonlinearity, L2 spatial pooling
- State-of-the-art certified robust accuracy: +8% CRA at ε=1 on ImageNet vs. prior Lipschitz models
- Trains stably in **bfloat16** (prior methods required float32)

**However, for generative models, the accuracy gap remains significant.** Newhouse (MIT, 2025) trained Lipschitz transformers:
- 4-Lipschitz transformer on Shakespeare: 60% validation accuracy
- 145M-parameter 10-Lipschitz transformer on internet text: 21% accuracy
- To match baseline (39.4% accuracy), Lipschitz bound rises to 10,274 — effectively unconstrained

This gap is the central challenge for applying Lipschitz constraints to diffusion transformers.

### 3. Lipschitz-Aware Diffusion Transformer Architectures

Several recent works demonstrate practical approaches to incorporating Lipschitz control into DiT architectures, short of full 1-Lipschitz enforcement:

**Skip-DiT (ICCV 2025):**
- Diagnoses *Dynamic Feature Instability* in vanilla DiT: spectral norms of layer Jacobians compound exponentially across depth
- Introduces long-skip-connections (LSCs) with spectral constraints, providing provably tighter Lipschitz bounds: σ_max(J_skip) ≤ (1−α)γ + αγ^{2l−L} < γ
- Achieves **4.4× training acceleration** and **1.5-2× inference acceleration** with negligible quality loss
- Works across both image and video generation tasks

**LipsFormer (IDEA Research):**
- Replaces every non-Lipschitz component in a Vision Transformer with Lipschitz-continuous alternatives:
  - **CenterNorm** (1-Lipschitz) instead of LayerNorm (not Lipschitz)
  - **Scaled cosine similarity attention** instead of dot-product attention (not globally Lipschitz on unbounded domains)
  - Spectral initialization instead of Xavier
  - Weighted residual shortcuts
- Trains without learning rate warmup, achieving faster convergence
- 82.70% accuracy on ImageNet with Swin-Tiny backbone

**Spectral Soft Cap / Spectral Hammer (Newhouse, 2025):**
- Novel weight constraint methods designed for the Muon optimizer
- Spectral soft cap: approximates σ → min(σ_max, σ) on all singular values via odd polynomial iteration
- Eliminates need for layer norm, QK norm, and logit softcapping
- Key insight: optimizer choice matters — Muon's fixed spectral norm updates synergize with spectral constraints better than AdamW

### 4. Why Video Generation Needs Lipschitz Control

Video diffusion models face unique stability challenges that Lipschitz-aware design directly addresses:

**Lipschitz singularities near t=0 (ICLR 2024 Oral):**
Yang et al. proved that diffusion models exhibit *infinite Lipschitz constants* as the timestep approaches zero. This is precisely where fine details — including temporal coherence — are resolved. Their E-TSDM method (timestep sharing near zero) reduces FID by 33%+ for fast sampling.

**Temporal error amplification:**
For video, Lipschitz singularities compound across frames. CAT-LVDM (2025) proved that structured corruption training enforces Lipschitz continuity along the temporal manifold, with practical gains: 396.3 MSR-VTT (vs. 422 baseline) using only 2M training videos.

**Frame-to-frame consistency:**
A network with bounded Lipschitz constant guarantees that similar latent inputs (adjacent frames) produce similar outputs. This is exactly what temporal consistency requires — and it's difficult to enforce without architectural Lipschitz constraints.

### 5. Toward GroupSort in Video DiT: A Synthesis

No existing work directly combines GroupSort/MaxMin activations with diffusion transformers for video generation. However, the components are converging:

**What would a GroupSort-DiT for video look like?**

1. **Attention mechanism:** Replace dot-product attention with scaled cosine similarity (LipsFormer) or L2-attention. Dot-product attention is not globally Lipschitz, making it the bottleneck for Lipschitz certification.

2. **Activation functions:** Replace GELU/SiLU in FFN blocks with MaxMin. This provides gradient-norm preservation without the expressivity loss of using ReLU under Lipschitz constraints.

3. **Normalization:** Replace LayerNorm with CenterNorm (1-Lipschitz). This eliminates a major source of Lipschitz-unbounded behavior.

4. **Weight constraints:** Use spectral soft cap or manifold optimization (LipNeXt) rather than hard orthogonal constraints. This trades strict 1-Lipschitz guarantees for practical trainability.

5. **Skip connections:** Adopt Skip-DiT's LSC architecture for Lipschitz-controlled residual paths.

6. **Temporal-specific design:** Apply tighter Lipschitz bounds along temporal dimensions than spatial ones (anisotropic constraints), as explored in our prior research on this topic.

**The critical tradeoff:**
- Strict 1-Lipschitz with GroupSort + orthogonal weights: provable guarantees but likely significant quality degradation for generation tasks (as seen in Newhouse's language modeling experiments)
- "Soft" Lipschitz control (spectral constraints, CenterNorm, skip connections): practical stability improvements without crippling expressivity
- **Hybrid approach** (most promising): Use GroupSort/MaxMin in specific bottleneck layers (e.g., temporal attention FFN) while allowing relaxed Lipschitz bounds elsewhere. This provides targeted stability where it matters most (temporal coherence) without global expressivity loss.

---

## Deep Dive: The Mathematics of GroupSort

### Why Sorting Preserves Gradient Norms

For a function f: ℝ^n → ℝ^n to be gradient-norm-preserving, we need:
‖∇f(x)‖₂ = 1 for almost all x

Sorting a group of k numbers is a piecewise-linear function. Within each region where the ordering is fixed, the Jacobian is a permutation matrix — which has operator norm exactly 1 in the ℓ₂ sense. On the boundaries between regions (where two elements are equal), GroupSort is non-differentiable but still Lipschitz.

This is fundamentally different from ReLU, whose Jacobian is a diagonal matrix of 0s and 1s — it has operator norm 1 when all activations are positive, but can reduce to 0, destroying gradient information.

### The Universal Approximation Proof (Sketch)

The key insight: any 1-Lipschitz function on a compact set can be uniformly approximated by a piecewise-linear 1-Lipschitz function. GroupSort networks with orthogonal weight matrices can represent any such piecewise-linear 1-Lipschitz function because:
1. Orthogonal matrices preserve norms (1-Lipschitz linear maps)
2. GroupSort preserves norms (1-Lipschitz nonlinear maps)
3. The composition is 1-Lipschitz
4. The FullSort operation provides enough "routing" capability to implement arbitrary piecewise-linear partitions

### Computational Cost

GroupSort/MaxMin adds negligible computational overhead:
- MaxMin: One comparison per pair — O(n/2) = O(n)
- FullSort: O(n log n) per group, but groups are typically small (k=2 to k=16)
- Compare to attention: O(n²) — GroupSort is never the bottleneck

---

## Connections to Prior Research

- **Lipschitz-Constrained Video Diffusion** (rq-1772228820007): Established the theoretical framework for Lipschitz bounds in video diffusion, including E-TSDM's singularity fix and Skip-DiT's spectral control. This research extends that by examining the specific role of activation functions.

- **Anisotropic Lipschitz Constraints** (rq-1772616600001): Proposed different Lipschitz bounds along spatial vs temporal axes. GroupSort/MaxMin activations could enable this by being inserted selectively in temporal processing pathways.

- **Adaptive Lipschitz Scheduling** (rq-1772616600003, queued): The idea of varying Lipschitz constraints across diffusion timesteps naturally connects to GroupSort's role — tighter Lipschitz control (more aggressive GroupSort, stricter orthogonal weights) during fine detail timesteps near t=0 where singularities are worst.

---

## Follow-up Questions

1. **Empirical benchmark: MaxMin vs GELU in DiT FFN blocks** — What happens to FID/FVD when you simply swap the activation function in a pretrained DiT architecture? Does fine-tuning recover quality?

2. **Anisotropic GroupSort scheduling** — Can GroupSort group sizes be varied across the temporal vs spatial attention pathways to provide tighter Lipschitz control along time while preserving spatial expressivity?

3. **Lipschitz-aware distillation for video DiT** — Can a large unconstrained video DiT teacher be distilled into a Lipschitz-constrained student using GroupSort activations, preserving quality while gaining stability guarantees?

---

## Sources

1. Anil, Lucas, Grosse. "Sorting Out Lipschitz Function Approximation." ICML 2019. https://arxiv.org/abs/1811.05381
2. Newhouse. "Training Transformers with Enforced Lipschitz Constants." arXiv 2507.13338, 2025. https://arxiv.org/abs/2507.13338
3. Newhouse. "Duality, Weight Decay, and Metrized Deep Learning." MIT MEng Thesis, 2025. https://dspace.mit.edu/bitstream/handle/1721.1/162956/newhouse-lakern-meng-eecs-2025-thesis.pdf
4. Chen et al. "Towards Stabilized and Efficient Diffusion Transformers through Long-Skip-Connections with Spectral Constraints." ICCV 2025. https://arxiv.org/abs/2411.17616
5. Qi et al. "LipsFormer: Introducing Lipschitz Continuity to Vision Transformers." arXiv 2304.09856. https://arxiv.org/abs/2304.09856
6. Yang et al. "Lipschitz Singularities in Diffusion Models." ICLR 2024 (Oral). https://arxiv.org/abs/2306.11251
7. Hu et al. "LipNeXt: Scaling up Lipschitz-based Certified Robustness to Billion-parameter Models." arXiv 2601.18513, 2026. https://arxiv.org/abs/2601.18513
8. Tanielian, Biau. "Approximating Lipschitz continuous functions with GroupSort neural networks." https://perso.lpsm.paris/~biau/BIAU/tsb.pdf
9. Prach et al. "1-Lipschitz Layers Compared: Memory, Speed, and Certifiable Robustness." CVPR 2024. https://openaccess.thecvf.com/content/CVPR2024/papers/Prach_1-Lipschitz_Layers_Compared_Memory_Speed_and_Certifiable_Robustness_CVPR_2024_paper.pdf
10. "Approximation Theory for Lipschitz Continuous Transformers." arXiv 2602.15503, 2026. https://arxiv.org/abs/2602.15503
