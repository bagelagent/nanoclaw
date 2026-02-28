# Spatial Mixture Models for Texture Synthesis: Per-Cell Routing for Region-Specific Multi-Texture Generation

**Research Topic ID:** rq-1739950000002-spatial-mixture-textures
**Completed:** 2026-02-19
**Tags:** neural-networks, mixture-of-experts, spatial-routing, multi-texture, nca

## Summary

**Can MNCA-style per-cell routing enable region-specific multi-texture generation?**

**Yes, definitively.** Two parallel paradigms have independently demonstrated this capability in 2025: **MNCA (Mixture of Neural Cellular Automata)** achieves spatial segmentation through probabilistic per-cell rule selection, while **genomic signal NCAs** enable region-specific textures through spatially-varying internal signals. These approaches are complementary and potentially combinable, offering distinct trade-offs between emergence vs. explicit control.

---

## Deep Dive

### 1. MNCA: Probabilistic Per-Cell Rule Selection

The **Mixture of Neural Cellular Automata** framework integrates stochastic mixture-of-experts concepts into NCAs, enabling heterogeneous local behaviors through probabilistic rule assignment.

#### Architecture

MNCA employs a **categorical mixture approach** where each cell's update is determined by:

- A **rule selector network π(s_i^t, η)** that takes only the current cell state and produces K probabilities
- Sampling z ~ Cat(π) to select which of K expert automata rules applies
- Gumbel-Softmax trick enables backpropagation through discrete selection
- **K distinct update networks (φ_k)**: Each expert has identical two-layer 1×1 convolutional architecture processing spatial features including Sobel gradients

#### Spatial Segmentation Capability

MNCA demonstrates **emergent spatial segmentation** without explicit spatial conditioning:

**Image morphogenesis experiments** show the model "automatically segment[s] heterogeneous cells" by assigning different rules to image regions—body, borders, and empty space receive distinct rules based purely on local cell state.

**Microscopy segmentation** reveals unsupervised clustering where "the model naturally grouped cells according to their morphological and proteomic profiles," with rule assignments correlating with phenotypic features.

This enables **region-specific texture generation** through learned spatial organization: different regions of a grid naturally develop different textures as cells self-organize into clusters governed by specialized expert rules.

#### Performance Characteristics

**Robustness improvements:**
- KL divergence reduction of **>100×** versus deterministic NCA (2.057 → 0.018) in tissue generation
- MSE consistently **2-10× lower** than baseline NCA across perturbations
- Enhanced recovery from damage: models return to original patterns after corruption

**Computational cost:**
- Multiple expert networks increase parameter count vs. single NCA
- Models train within feasible timelines on single GPUs (specific benchmarks not provided)
- Per-cell categorical sampling adds overhead but remains parallelizable

#### Limitations

- **Scalability questions:** Framework requires further development for larger-scale systems
- **Interpretability gaps:** While rules segment spatially, extracting interpretable information remains challenging
- **Data requirements:** Demands multiple time-series realizations for biological modeling
- **Stochasticity vs. mixture:** Internal noise helps rare events, but mixture components matter more than stochasticity alone for robustness

---

### 2. Genomic Signals: Explicit Spatial Control

**Multi-texture synthesis through signal-responsive NCAs** achieves region-specific generation through spatially-varying internal signals rather than probabilistic routing.

#### Architecture

- **Genomic channels:** A few hidden channels (ng = 2-3) use binary encoding for texture identity
- Single NCA learns to interpret these signals, generating 2^ng distinct textures
- **Grafting technique:** Different spatial regions receive different genomic encodings at initialization
- The automaton maintains "a consistent boundary transition area of cells, whether visualized as a barrier or a smooth transition"

#### Spatial Mixture Capability

**Yes, genomic signals enable explicit region-specific textures:**

Researchers create spatial compositions by "initializing...some cells with the seed values of a different genome," such as "a NCA formed of cells of genome 2 with a disc of genome 1 situated in the center."

**Interpolation:** Intermediate genome values enable smooth blending at texture boundaries, creating natural transition zones between regions.

**Multiple patterns coexist:** "Cells situated...of genome 1 and 2" develop their corresponding textures simultaneously within a single automaton instance.

#### Parameter Efficiency

- **G8L (Large):** ~10,000 parameters, regenerates textures in ~180 steps
- **G8M (Medium):** ~4,270 parameters, regenerates in ~420 steps
- **G8SNR (Small):** ~1,500 parameters, stable up to 500 steps

Training typically ~100 minutes on T4 GPUs. Most NCAs maintain stability to 6,000+ iterations before genome corruption.

#### Advantages

- **Explicit control:** Users directly specify which texture appears where
- **Extreme compactness:** 1,500-10,000 params for 8 textures
- **Real-time performance:** 25Hz generation rates on modest hardware
- **Interpretable:** Genomic encoding provides clear texture-to-signal mapping

#### Limitations

- **Genome corruption:** Long-term stability degrades beyond 6,000 steps
- **Binary encoding constraint:** Number of textures limited to powers of 2 (2^ng)
- **Manual spatial assignment:** Requires explicit initialization of genomic regions
- **Boundary artifacts:** Transition zones may exhibit inconsistent behaviors

---

### 3. Hybrid & Alternative Approaches

#### PD-NCA: Competition-Based Spatial Routing

**Petri Dish NCA** implements spatial heterogeneity through competitive dynamics rather than explicit assignment:

- Multiple NCA agents with distinct neural parameters compete for grid cells
- Differentiable attack/defense channels determine which agent's update "wins" locally
- Softmax normalization creates soft routing with weighted aggregation of proposals
- **Aliveness threshold of 0.4** allows up to 2 NCAs per cell, enabling coexistence

This creates spatial regions with genuinely heterogeneous computational rules through emergent territorial competition.

#### Universal NCA: Hardware Conditioning

Universal NCAs handle heterogeneous substrates by providing task-specific hardware vectors that condition the update rule, separating "what to compute" from "how to compute it." This enables per-cell task routing while sharing learned local computation primitives.

#### FG-MoE: Spatially-Aware Gating (Vision Context)

Recent computer vision work (FG-MoE, 2026) demonstrates spatially-aware MoE gating for image classification, using five specialized experts (global structures, regional semantics, local details, textures, part-level interactions) with dynamic per-region expert selection. This validates the broader pattern of spatial mixture models in visual domains.

---

### 4. Comparative Analysis

| Approach | Routing Mechanism | Spatial Control | Parameter Efficiency | Emergence vs. Control |
|----------|-------------------|-----------------|---------------------|----------------------|
| **MNCA** | Probabilistic per-cell rule selection | Emergent segmentation | Multiple experts increase count | High emergence |
| **Genomic Signals** | Spatially-varying internal signals | Explicit initialization | 1,500-10,000 params for 8 textures | High control |
| **PD-NCA** | Competitive agent dynamics | Territorial emergence | Multiple agent networks | Highest emergence |
| **Universal NCA** | Hardware vector conditioning | Task-based per-cell | Shared primitives + task vectors | Balanced |

---

### 5. Practical Implications

**For region-specific multi-texture generation:**

**MNCA excels when:**
- Spatial organization should emerge from data rather than be specified
- Biological/physical fidelity requires stochastic dynamics
- Interpretable unsupervised clustering is valuable
- Robustness to perturbations is critical

**Genomic signals excel when:**
- Explicit spatial control over texture placement is needed
- Extreme parameter efficiency is paramount (1,500-10,000 params)
- Real-time performance is required (25Hz+)
- Interpolation between textures is valuable
- Grafting/composition workflows are desired

**Hybrid approach (not yet implemented):**
Combining MNCA's emergent segmentation with genomic signal conditioning could enable:
- Coarse spatial assignment via genomic initialization
- Fine-grained local variation via probabilistic expert selection
- Best of both worlds: explicit control + adaptive emergence

---

## Connections to Existing Knowledge

**Mixture-of-Experts Paradigm:** MNCA directly applies sparse MoE concepts (used in transformers like Switch Transformer, DeepSeek-V3) to spatial domains. The key difference: spatial correlation means neighboring cells often select the same expert, creating natural segmentation rather than load balancing challenges.

**Cascade Routing:** Spatial mixture models could enable cascade routing (explored in research on NCA model zoos): simple regions use cheap models, complex regions escalate to expensive specialists. MNCA's probabilistic routing provides a natural framework for this.

**Hierarchical NCAs:** Spatial mixture models could be applied at multiple scales—coarse-scale experts handle large-region layout, fine-scale experts handle local detail. Connects to hierarchical NCA research (HNCA, hGCA, ViTCA).

**Genomic Scaling:** Current genomic signal work uses 2-3 bit encodings (8 textures). Scaling to ng=5-6 bits (32-64 textures) is identified in the research queue as an open question—spatial mixture models may provide alternative scaling path through expert specialization rather than genomic capacity.

---

## Follow-Up Questions

1. **Hybrid MNCA + Genomic Signals:** Can genomic channels serve as additional input to MNCA's rule selector, combining explicit control with emergent refinement?

2. **Computational Scaling:** At what K (number of experts) does MNCA routing overhead dominate single-NCA cost? Is there a sweet spot?

3. **Spatial Priors:** Can MNCA rule selectors be conditioned on spatial coordinates or distance fields to bias segmentation patterns?

4. **Temporal Routing:** Can expert selection vary over time (different rules govern early vs. late texture evolution)?

5. **Cross-Domain Transfer:** Do MNCA routing patterns learned for texture synthesis transfer to morphogenesis or simulation tasks?

6. **Quality-Aware Routing:** Can learned quality predictors automatically assign complex texture regions to more capable (expensive) expert NCAs?

7. **Genome-Guided Competition:** In PD-NCA style systems, could genomic signals guide competitive dynamics rather than pre-assign regions?

---

## Sources

### Primary Research

- [Mixtures of Neural Cellular Automata: A Stochastic Framework for Growth Modelling and Self-Organization](https://arxiv.org/html/2506.20486) - Salvatore Milite et al., arXiv 2506.20486, June 2025
- [Multi-Texture Synthesis through Signal Responsive Neural Cellular Automata](https://arxiv.org/html/2407.05991) - Scientific Reports, November 2025
- [Multi-texture synthesis through signal responsive neural cellular automata | Scientific Reports](https://www.nature.com/articles/s41598-025-23997-7)
- [Petri Dish Neural Cellular Automata](https://pub.sakana.ai/pdnca/) - Sakana AI, 2024
- [A Path to Universal Neural Cellular Automata](https://arxiv.org/html/2505.13058v1) - Gabriel Béna et al., May 2025

### Related MoE & Spatial Routing

- [FG-MoE: Heterogeneous mixture of experts model for fine-grained visual classification](https://www.sciencedirect.com/science/article/abs/pii/S0031320326000130) - Pattern Recognition, 2026
- [Learning Heterogeneous Tissues with Mixture of Experts for Gigapixel Whole Slide Images](https://openaccess.thecvf.com/content/CVPR2025/papers/Wu_Learning_Heterogeneous_Tissues_with_Mixture_of_Experts_for_Gigapixel_Whole_CVPR_2025_paper.pdf) - CVPR 2025
- [Learning Heterogeneous Mixture of Scene Experts for Large-scale Neural Radiance Fields](https://arxiv.org/abs/2505.02005) - Switch-NeRF++, arXiv 2505.02005, May 2025
- [Mixture of Experts in Large Language Models](https://arxiv.org/html/2507.11181v2) - Survey paper, July 2025

### Texture Synthesis & Spatial Generation

- [Texture Synthesis with Spatial Generative Adversarial Networks](https://arxiv.org/abs/1611.08207) - Jetchev et al., November 2016
- [Conditional Morphogenesis: Emergent Generation of](https://www.arxiv.org/pdf/2512.08360) - December 2025
- [DyNCA: Real-Time Dynamic Texture Synthesis Using Neural Cellular Automata](https://dynca.github.io/) - CVPR 2023
- [Self-Organising Textures](https://distill.pub/selforg/2021/textures/) - Distill, 2021
- [Texture Generation with Neural Cellular Automata](https://arxiv.org/abs/2105.07299) - Mordvintsev et al., May 2021

### NCA Performance & Efficiency

- [Parameter-efficient diffusion with neural cellular automata | npj Unconventional Computing](https://www.nature.com/articles/s44335-025-00026-4) - Diff-NCA paper, January 2025
- [Neural Cellular Automata: From Cells to Pixels](https://arxiv.org/html/2506.22899v2) - Comprehensive NCA survey
- [Neural Cellular Automata for ARC-AGI](https://arxiv.org/html/2506.15746v1) - Computational efficiency focus
- [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/) - Foundational Distill article, 2020

### Heterogeneous CA & Routing

- [Emergent Dynamics in Heterogeneous Life-Like Cellular Automata](https://arxiv.org/html/2406.13383v1) - June 2024
- [Learning spatio-temporal patterns with Neural Cellular Automata](https://pmc.ncbi.nlm.nih.gov/articles/PMC11078362/) - PMC article on spatial patterns
- [Scaling Vision with Sparse Mixture of Experts](https://research.google/blog/scaling-vision-with-sparse-mixture-of-experts/) - Google Research on vision MoE

---

## Conclusion

**MNCA-style per-cell routing definitively enables region-specific multi-texture generation** through two validated paradigms:

1. **Emergent segmentation** (MNCA): Probabilistic rule selection creates natural spatial organization
2. **Explicit control** (Genomic signals): Spatially-varying internal signals directly specify regions

Both approaches achieve **extreme parameter efficiency** (1,500-10,000 parameters for 8 textures) and **real-time performance**, making them viable for practical applications. The complementary nature of these methods suggests hybrid approaches could offer the best of both worlds: user-specified coarse layout with adaptive fine-grained emergence.

**Key insight:** Spatial mixture models for NCAs avoid the load balancing challenges of token-based MoE because spatial correlation naturally creates contiguous regions governed by specialized experts. This makes per-cell routing more tractable in spatial domains than in language models.

The field is actively developing (2024-2025), with open questions around scaling, hybrid architectures, and cross-domain transfer representing high-value research opportunities.
