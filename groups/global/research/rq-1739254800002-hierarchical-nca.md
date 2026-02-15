# Hierarchical Multi-Scale NCAs with Cross-Scale Communication

**Research Date:** 2026-02-14
**Research ID:** rq-1739254800002-hierarchical-nca
**Tags:** neural-networks, hierarchical-models, multi-scale, architecture, nca

---

## Summary

Hierarchical multi-scale Neural Cellular Automata (NCAs) represent an emerging architectural paradigm that addresses fundamental limitations of standard NCAs—particularly their struggle with long-range dependencies and scaling to high resolutions. By organizing NCAs into multi-level hierarchies with explicit cross-scale communication mechanisms, researchers are achieving foundation-model-like capabilities including transfer learning, compositional generalization, and multi-task performance. Key innovations include hierarchical parent-child communication layers, attention-based cross-scale interactions (ViTCA), coarse-to-fine generation (hGCA), and hybrid NCA-transformer architectures (AdaNCA). While true "foundation NCAs" don't yet exist, hierarchical designs bridge the gap between NCAs' strengths (compactness, robustness, local computation) and transformers' capabilities (global context, zero-shot learning, emergent abilities).

---

## Key Findings

### 1. **Standard NCA Limitations Drive Hierarchical Innovation**

Neural Cellular Automata face three critical scaling limitations:

- **Quadratic growth in training time/memory** with grid size
- **Strictly local information propagation** (one cell per update step)
- **Heavy compute demands** for real-time high-resolution inference

These constraints confine NCAs to low-resolution grids (typically 128×128 or smaller). Increasing grid size requires many more update steps for distant cells to communicate, which hinders convergence and increases VRAM requirements. The fundamental tension: maintaining beneficial local, decentralized properties while enabling sufficient global information flow for complex tasks.

**Sources:** [Neural Cellular Automata: From Cells to Pixels](https://arxiv.org/html/2506.22899v1), [Parameter-efficient diffusion with neural cellular automata](https://www.nature.com/articles/s44335-025-00026-4), [Neural cellular automata: Applications to biology and beyond classical AI](https://www.sciencedirect.com/science/article/pii/S1571064525001757)

### 2. **Hierarchical Architectures Enable Cross-Scale Communication**

Hierarchical NCAs (HNCAs) comprise multiple integrated layers operating at progressively coarser resolutions. Key architectural innovations:

**Parent-Child Communication:**
- Cells integrate state information from lateral neighbors AND neighbors in adjacent layers
- Higher-level NCAs progressively condition collective behavior of lower levels
- Both parent and child NCAs consume signals via actuator/sensor interfaces at each timestep
- Communication layers enable bidirectional information flow between scales

**Performance Benefits:**
- **GECCO 2024 findings:** Organizing NCAs hierarchically improves ability to evolve them for morphogenesis and homeostasis compared to flat architectures
- Enables multi-scale competency architecture perspective on evolution, development, regeneration, and morphogenesis
- Addresses how living systems organize as hierarchical arrangements of semi-independent components at different scales

**Sources:** [Hierarchical Neural Cellular Automata (MIT Press)](https://direct.mit.edu/isal/proceedings-pdf/isal2023/35/20/2354957/isal_a_00601.pdf), [Evolving Hierarchical Neural Cellular Automata (GECCO 2024)](https://dl.acm.org/doi/10.1145/3638529.3654150), [Neural cellular automata: applications to biology and beyond classical AI](https://arxiv.org/abs/2509.11131)

### 3. **ViTCA: Attention-Based Cross-Scale Communication**

Vision Transformer Cellular Automata (ViTCA) represents a breakthrough in attention-based NCAs using spatially localized yet globally organized self-attention:

**Architecture:**
- Extends NCA with Transformer mechanisms: self-attention and positional encoding
- Circumvents quadratic complexity of self-attention through spatial localization
- Over CA iterations, effective receptive field grows until implicitly incorporating information across all cells
- Enables global propagation from spatially localized attention

**Performance:**
- **Superior performance:** When comparing architectures at similar parameter complexity, ViTCA yields superior performance across all benchmarks and nearly every metric
- **Parameter efficiency:** Allows lower model complexity by limiting ViT depth while retaining expressivity through CA iterations with same encoder weights
- **State-of-the-art:** Stands as SOTA within NCA family, benchmark for subsequent research

**Sources:** [Attention-based Neural Cellular Automata](https://arxiv.org/abs/2211.01233), [ViTCA NeurIPS 2022](https://papers.neurips.cc/paper_files/paper/2022/file/361e5112d2eca09513bbd266e4b2d2be-Paper-Conference.pdf), [ViTCA (OpenReview)](https://openreview.net/forum?id=9t24EBSlZOa)

### 4. **AdaNCA: NCAs as Plug-and-Play Adaptors for Vision Transformers**

AdaNCA inserts NCAs as adaptors between ViT layers, creating hybrid architectures that combine benefits of both paradigms:

**Architecture:**
- **Plug-and-play:** NCAs inserted into middle layers of ViT to improve robustness
- **Dynamic Interaction:** Point-wise weighted sum on interaction results from depth-wise convolutions; weights obtained from token states so each token dynamically adjusts interaction strategy
- **Optimal placement:** Maximum improvement when inserted between two similar layer sets

**Performance:**
- **Parameter efficiency:** <3% parameter increase yields >10% absolute accuracy improvement under adversarial attacks (ImageNet1K)
- **Robustness gains:** Improvements don't originate from parameter increase but from AdaNCA architecture itself
- **Consistent benefits:** Improves robustness across 8 robustness benchmarks and 4 ViT architectures
- **Multiple benefits:** Enhances adversarial robustness, out-of-distribution performance, AND clean accuracy

**Key Insight:** NCAs enable modeling of global visual-token representations through local interactions, with training strategies and architecture conferring strong generalization ability and robustness against noisy input.

**Sources:** [AdaNCA: Neural Cellular Automata As Adaptors For More Robust Vision Transformer](https://arxiv.org/abs/2406.08298), [AdaNCA NeurIPS 2024](https://neurips.cc/virtual/2024/poster/96193), [AdaNCA (OpenReview)](https://openreview.net/forum?id=BQh1SGvROG)

### 5. **hGCA: Hierarchical Generative Cellular Automata for Coarse-to-Fine 3D Scene Generation**

Hierarchical Generative Cellular Automata (hGCA) demonstrates how coarse-to-fine hierarchies enable scalable generation at unprecedented scales:

**Architecture:**
- **Spatially scalable:** Grows geometry recursively with local kernels in coarse-to-fine manner
- **Two-stage hierarchy:** Coarse stage generates initial geometry at lower resolution; fine stage recursively refines with progressively higher detail
- **Lightweight planner:** Induces global consistency across scales
- **Built on GCA framework:** Extends 3D generative model that recursively applies local kernels to incrementally grow geometry

**Performance:**
- **Large-scale generation:** Achieves 100+ meter scene completion on single 24GB GPU
- **Higher fidelity:** Generates plausible scene geometry with higher fidelity and completeness vs SOTA baselines
- **Strong generalization:** Sim-to-real transfer qualitatively outperforms baselines on Waymo-open dataset despite synthetic training
- **Handles complexity:** Processes sparse LiDAR scans, completes occluded geometry, generates novel objects

**Application:** Fine-grained 3D geometry from large-scale sparse LiDAR scans captured by autonomous vehicles, extrapolating beyond spatial limits of scans toward realistic, high-resolution simulation-ready 3D street environments.

**Sources:** [Outdoor Scene Extrapolation with Hierarchical Generative Cellular Automata (NVIDIA)](https://research.nvidia.com/labs/toronto-ai/hGCA/), [hGCA CVPR 2024](https://arxiv.org/abs/2406.08292), [hGCA Paper (CVPR Open Access)](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Outdoor_Scene_Extrapolation_with_Hierarchical_Generative_Cellular_Automata_CVPR_2024_paper.html)

### 6. **Foundation Model Capabilities: Can Hierarchical NCAs Achieve Them?**

Foundation models exhibit four key capabilities through pre-training at scale:

**1. Zero-Shot Learning**
- Perform tasks without direct training, purely from context/instructions
- Parameter-free adaptation via natural language prompts
- Relies on emergent capabilities from large-scale pre-training

**2. Few-Shot Learning**
- Learn from handful of examples in prompt (in-context learning)
- Rapid generalization to new tasks/concepts without direct training
- Enabled by generalized representations from pre-training

**3. Transfer Learning**
- Rapid adaptation via fine-tuning, few-shot learning, or prompt engineering
- Specialized for new tasks without retraining from scratch
- Generalized representations enable competitive accuracy with far less data/compute

**4. Emergent Abilities**
- Capabilities appearing suddenly at scale, not explicitly programmed
- Examples: chain-of-thought reasoning, complex problem decomposition
- Improve as model scale increases

**Sources:** [Zero-Shot and Few-Shot Learning with Foundation Models](https://medium.com/@chaituapatil3/zero-shot-and-few-shot-learning-with-foundation-models-redefining-the-capabilities-of-ai-33b1bf3f203d), [Foundation Models (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10550749/), [Emergent Abilities in Large Language Models](https://arxiv.org/html/2503.05788v2)

### 7. **NCAs vs Transformers: Architectural Trade-offs for Foundation Model Capabilities**

**Computation Model:**
- **NCAs:** Decentralized, local computation with identical small networks across grid; massive parallelism and robustness
- **Transformers:** Centralized parameters with global self-attention; look at everything at once

**Information Propagation:**
- **NCAs:** Strictly local propagation impedes long-range communication; information travels one cell per update
- **Transformers:** Excel at long-range dependencies through global attention

**Expressiveness:**
- **NCAs:** Internalize goal-directed dynamics; convergence emerges from local updates; self-repair, adaptation, compositional generalization
- **Transformers:** Expressivity capped by rank but additional layers yield exponential complexity gains

**Hierarchical Processing:**
- **NCAs:** Stack blocks to build hierarchical representations; hierarchical temporal structure (fast neural dynamics + slower hardware dynamics)
- **Transformers:** Comprehend global context effectively; hierarchical architectures (SWIN) operate at multiple scales

**Parameter Efficiency:**
- **NCAs:** Ultra-compact (68-8000 params for texture synthesis)
- **Transformers:** Require billions of parameters for foundation capabilities
- **Hybrid (ViTCA):** Lower complexity while retaining expressivity through CA iterations

**Training Stability:**
- **NCAs:** Difficult to train, sensitive to architecture choices; no straightforward way to measure "solution complexity"; gradient descent settles on arbitrary local optima
- **Transformers:** Mature training paradigms but face vanishing gradients for long-range dependencies

**Sources:** [NCAs vs transformers search results](https://www.emergentmind.com/topics/neural-cellular-automata-nca), [Neural Cellular Automata: From Cells to Pixels](https://arxiv.org/abs/2506.22899), [A Path to Universal Neural Cellular Automata](https://arxiv.org/html/2505.13058v1)

### 8. **Multi-Scale Models in Computer Vision Provide Blueprint**

Established multi-scale architectures demonstrate successful hierarchical feature extraction:

**U-Net:**
- Encoder extracts contour information at different sizes
- Decoder combines features from coarse to fine layers
- Effective for medical image segmentation, remote sensing

**Feature Pyramid Networks (FPN):**
- Top-down architecture with lateral connections
- Blends high-level semantic information with low-level spatial information
- Significantly improves multi-scale feature extraction

**Coarse-to-Fine Principles:**
- Progressively refine focus across hierarchical levels
- Capture both coarse and fine-grained contextual information
- Gradual shift from global understanding to fine detail refinement

**Design Principles Applicable to NCAs:**
- Multi-scale features critical for capturing both object structure and subtle local details
- Shallow layers capture low-level features (edges, textures)
- Deeper layers extract complex, high-level semantic information
- Effective fusion of features across scales is key challenge

**Sources:** [Multi-scale feature progressive fusion network](https://www.nature.com/articles/s41598-022-16329-6), [Medical image segmentation with UNet-based multi-scale context fusion](https://www.nature.com/articles/s41598-024-66585-x), [Feature Pyramid Network for Multi-Scale Detection](https://rumn.medium.com/feature-pyramid-network-for-multi-scale-detection-f573a889c7b1)

---

## Deep Dive: Path to Foundation-Model-Like NCAs

### Current State: Hierarchical NCAs Bridge the Gap

True "foundation NCAs" don't exist yet—no pre-trained NCA exhibits the zero-shot, few-shot, and emergent capabilities of large language models or vision transformers. However, hierarchical multi-scale architectures represent the most promising path forward by addressing NCAs' fundamental limitations while preserving their unique strengths.

### Three Successful Hierarchical Paradigms

**1. Explicit Hierarchy (HNCA, hGCA)**
- Multiple NCA layers at different resolutions
- Parent-child communication via actuator/sensor interfaces
- Coarse-to-fine generation for scalability
- Demonstrated benefits: improved morphogenesis, 100m+ scene generation on single GPU

**2. Attention-Based Hierarchy (ViTCA)**
- Spatially localized self-attention that grows over iterations
- Combines NCA local computation with Transformer global awareness
- Superior performance at similar parameter count vs standard NCAs
- Most parameter-efficient within NCA family

**3. Hybrid Architecture (AdaNCA)**
- NCAs as plug-and-play adaptors between Transformer layers
- Best of both worlds: Transformer global context + NCA robustness
- Minimal parameter overhead (<3%) for major gains (>10% accuracy)
- Demonstrates synergy between paradigms

### Barriers to Foundation-Model Capabilities

**1. Architectural Mismatch**
- Foundation models rely on global context; NCAs are fundamentally local
- Information propagates slowly across large grids (one cell per step)
- Hierarchical designs partially address this but don't eliminate constraint

**2. Training Paradigm Gap**
- Foundation models use self-supervised pre-training on massive heterogeneous corpora
- NCAs typically trained supervised on specific tasks
- No established "pre-training strategy" for general-purpose NCAs (but see related research on self-supervised NCA pre-training: rq-1739254800000-nca-self-supervised)

**3. Task Representation Challenge**
- Foundation models learn to condition on natural language prompts
- NCAs condition on genomic signals, CLIP embeddings, or input patterns
- No unified "prompt" interface for general task specification (but see related research: rq-1770914400001-prompt-to-genome)

**4. Gradient Flow and Stability**
- NCAs difficult to train, sensitive to architecture choices
- No straightforward way to measure "solution complexity"
- Gradient descent settles on arbitrary local optima based on initialization
- Recurrent nature creates vanishing gradient challenges for long-range dependencies

### Promising Directions

**1. Self-Supervised Pre-Training + Hierarchical Architecture**
- Combine hierarchical multi-scale design with self-supervised objectives
- Contrastive learning on pattern dynamics across scales
- Masked pattern prediction at multiple resolutions
- Could yield transferable "foundation" NCAs

**2. CLIP-Conditioned Hierarchical NCAs**
- Use CLIP embeddings to condition NCA behavior at multiple scales
- Enables zero-shot generalization via natural language
- CLIP at coarse scales for semantic guidance, VGG at fine scales for texture
- Bridges task specification gap (see related research: rq-1739254800001-clip-conditioned-nca, rq-1770914400002-multiscale-clip-vgg)

**3. NCA Model Zoos with Learned Routers**
- Collections of specialized hierarchical NCAs
- Learned routing mechanisms select appropriate NCA(s) for task
- Approximates foundation model behavior through mixture-of-experts
- Maintains compactness while achieving versatility (see related research: rq-1739254800004-nca-model-zoos)

**4. Scaling Laws Research**
- Systematic study of relationships between parameter count, hierarchy depth, training compute, and generalization
- Identify optimal scaling strategies for hierarchical NCAs
- Could reveal emergent capabilities at sufficient scale (see related research: rq-1739254800003-nca-scaling-laws)

### Critical Open Questions

**Can hierarchical NCAs achieve foundation model capabilities?**

**Optimistic Case (Yes):**
- Hierarchical designs already demonstrate improved generalization and transfer learning
- AdaNCA shows NCAs enhance robustness across diverse tasks with minimal overhead
- ViTCA proves attention mechanisms compatible with local computation paradigm
- hGCA achieves unprecedented scalability (100m+ scenes on single GPU)
- Self-supervised pre-training on massive pattern dynamics datasets could unlock emergent abilities
- CLIP conditioning provides pathway to zero-shot generalization
- Parameter efficiency (8000 params vs billions) means "scale" could come from ensemble size, not individual model size

**Pessimistic Case (No):**
- NCAs' local computation paradigm fundamentally incompatible with foundation models' global reasoning
- Hierarchical communication insufficient to match Transformers' global attention
- Training instability and local optima problems worsen at scale
- No clear pathway to emergent abilities like chain-of-thought reasoning
- Task diversity required for foundation models may exceed what local update rules can learn

**Middle Ground (Partial Capabilities):**
- Hierarchical NCAs achieve **transfer learning** and **few-shot learning** through genomic signal modulation and fine-tuning protocols
- **Zero-shot learning** possible with CLIP conditioning but limited compared to language models
- **Emergent abilities** may require architectural innovations not yet discovered
- Most likely outcome: hierarchical NCAs carve out specialized niche where compactness + robustness + local computation matter more than general-purpose reasoning

### What Success Looks Like

A "foundation hierarchical NCA" would exhibit:

1. **Pre-training:** Self-supervised learning on diverse pattern dynamics datasets
2. **Zero-shot transfer:** Generate novel patterns from CLIP text embeddings without fine-tuning
3. **Few-shot adaptation:** Fine-tune on 5-10 examples to learn new texture categories
4. **Multi-scale reasoning:** Coordinate information across hierarchical levels to achieve complex goals
5. **Compositional generalization:** Combine learned patterns in novel ways
6. **Robustness:** Self-repair, adaptation to perturbations, graceful degradation
7. **Extreme efficiency:** 10k-1M parameters, real-time on edge devices
8. **Emergent behaviors:** Capabilities not explicitly trained, arising from scale/architecture

Current hierarchical NCAs achieve #4, #6, #7. Research directions aim for #1, #2, #3. Achieving #5, #8 remains open question.

---

## Connections to Existing Knowledge

### Direct Connections

1. **Self-Supervised Pre-Training (rq-1739254800000-nca-self-supervised):** Hierarchical architectures are ideal candidates for self-supervised pre-training strategies like masked prediction at multiple scales, contrastive learning on cross-scale dynamics, and predictive coding of coarse-to-fine evolution.

2. **CLIP-Conditioned NCAs (rq-1739254800001-clip-conditioned-nca):** Hierarchical NCAs could use CLIP embeddings at coarse scales for semantic guidance while using VGG/texture loss at fine scales—best of both worlds for controllability.

3. **NCA Model Zoos (rq-1739254800004-nca-model-zoos):** Collections of hierarchical NCAs with learned routers could approximate foundation model behavior through mixture-of-experts, where each specialist NCA handles specific pattern domains.

4. **Scaling Laws (rq-1739254800003-nca-scaling-laws):** Understanding how hierarchical depth, cross-scale communication bandwidth, and parameter distribution across levels affect performance is critical for identifying optimal architectures.

5. **Fine-Tuning Protocols (rq-1739254800005-nca-fine-tuning):** Hierarchical NCAs raise questions about which levels to freeze during fine-tuning—freeze coarse layers for global structure, fine-tune fine layers for detail?

### Indirect Connections

6. **Hybrid Procedural Techniques (rq-1770847193001-hybrid):** Hierarchical NCAs fit naturally into hybrid pipelines—use coarse levels with procedural noise initialization, fine levels for learned refinement.

7. **Real-Time Performance (rq-1770925716000-hybrid-performance):** Hierarchical designs enable progressive computation—run coarse levels every frame, update fine levels less frequently for real-time balance.

8. **Multi-Texture Synthesis (rq-1739076481002-controllable-nca):** Hierarchical architectures enable multi-scale conditioning—different signals at different levels for complex multi-texture control.

### Biological Inspiration

9. **Multi-Scale Competency Architecture:** Hierarchical NCAs align with biology's multi-scale organization—molecular, cellular, tissue, organ-level coordination through local interactions and hierarchical signaling.

10. **Morphogenesis and Development:** Evolution uses hierarchical control—coarse-scale body plan genes, fine-scale tissue differentiation genes. Hierarchical NCAs mirror this organization.

---

## Follow-Up Questions

### Architecture & Design
1. **Optimal hierarchy depth:** How many levels? What resolution ratios between levels?
2. **Communication bandwidth:** How much information should flow between levels? Dense connections vs sparse?
3. **Asymmetric hierarchies:** Should all paths be coarse-to-fine, or can fine-level patterns inform coarse-level decisions?
4. **Hybrid attention:** Can we combine local NCA updates with sparse global attention only at coarsest level?

### Training & Optimization
5. **Curriculum learning across scales:** Train coarse levels first, progressively add fine levels?
6. **Level-specific loss functions:** Different objectives at different scales (semantic at coarse, texture at fine)?
7. **Gradient flow through hierarchy:** How to prevent vanishing gradients across many levels?
8. **Stability at scale:** Do hierarchical NCAs exhibit better or worse training stability than flat NCAs?

### Capabilities & Applications
9. **Compositional generation:** Can hierarchical NCAs compose learned patterns in zero-shot manner?
10. **Continual learning:** Can we add new levels or refine existing levels without catastrophic forgetting?
11. **Cross-domain transfer:** Do hierarchies learned for textures transfer to 3D volumetric generation? Video dynamics?
12. **Interactive applications:** Can users control generation by intervening at specific hierarchical levels?

### Fundamental Limits
13. **Expressiveness vs efficiency trade-off:** At what point do hierarchical NCAs match Transformer expressiveness while preserving efficiency?
14. **Emergent capabilities threshold:** Is there a "phase transition" in capabilities at sufficient hierarchical depth/scale?
15. **Local-to-global ceiling:** What types of reasoning/generation will ALWAYS require global attention and thus be impossible for hierarchical NCAs?

---

## Sources

### Hierarchical NCA Architectures
- [Hierarchical Neural Cellular Automata (MIT Press)](https://direct.mit.edu/isal/proceedings-pdf/isal2023/35/20/2354957/isal_a_00601.pdf)
- [Evolving Hierarchical Neural Cellular Automata (GECCO 2024)](https://dl.acm.org/doi/10.1145/3638529.3654150)
- [Neural cellular automata: applications to biology and beyond classical AI](https://arxiv.org/abs/2509.11131)

### Attention-Based NCAs
- [Attention-based Neural Cellular Automata](https://arxiv.org/abs/2211.01233)
- [ViTCA NeurIPS 2022](https://papers.neurips.cc/paper_files/paper/2022/file/361e5112d2eca09513bbd266e4b2d2be-Paper-Conference.pdf)
- [ViTCA (OpenReview)](https://openreview.net/forum?id=9t24EBSlZOa)
- [ViTCA (ServiceNow Research)](https://www.servicenow.com/research/publication/mattie-tesfaldet-atte-neurips2022.html)

### Hybrid NCA-Transformer Architectures
- [AdaNCA: Neural Cellular Automata As Adaptors For More Robust Vision Transformer](https://arxiv.org/abs/2406.08298)
- [AdaNCA NeurIPS 2024](https://neurips.cc/virtual/2024/poster/96193)
- [AdaNCA (OpenReview)](https://openreview.net/forum?id=BQh1SGvROG)
- [AdaNCA HTML](https://arxiv.org/html/2406.08298v5)

### Coarse-to-Fine Generation
- [Outdoor Scene Extrapolation with Hierarchical Generative Cellular Automata (NVIDIA)](https://research.nvidia.com/labs/toronto-ai/hGCA/)
- [hGCA ArXiv](https://arxiv.org/abs/2406.08292)
- [hGCA CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Outdoor_Scene_Extrapolation_with_Hierarchical_Generative_Cellular_Automata_CVPR_2024_paper.html)

### NCA Limitations & Challenges
- [Neural Cellular Automata: From Cells to Pixels](https://arxiv.org/html/2506.22899v1)
- [Parameter-efficient diffusion with neural cellular automata](https://www.nature.com/articles/s44335-025-00026-4)
- [Neural cellular automata: Applications to biology and beyond classical AI](https://www.sciencedirect.com/science/article/pii/S1571064525001757)
- [NCAs (Emergent Mind)](https://www.emergentmind.com/topics/neural-cellular-automata-ncas)
- [Frequency-Time Diffusion with Neural Cellular Automata](https://arxiv.org/html/2401.06291v1)
- [NCA for Decentralized Sensing](https://arxiv.org/html/2502.01242)

### Foundation Model Capabilities
- [Zero-Shot and Few-Shot Learning with Foundation Models](https://medium.com/@chaituapatil3/zero-shot-and-few-shot-learning-with-foundation-models-redefining-the-capabilities-of-ai-33b1bf3f203d)
- [Foundation Models (Robot Post)](https://www.therobotpost.com/2025/12/foundation-models.html)
- [Foundation Model (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10550749/)
- [Zero-Shot Learning (IBM)](https://www.ibm.com/think/topics/zero-shot-learning)
- [Zero-Shot Learning Explained (Encord)](https://encord.com/blog/zero-shot-learning-explained/)
- [Emergent Abilities in Large Language Models](https://arxiv.org/html/2503.05788v2)
- [Foundation model (Grokipedia)](https://grokipedia.com/page/Foundation_model)

### Multi-Scale Vision Models
- [Multi-scale feature progressive fusion network](https://www.nature.com/articles/s41598-022-16329-6)
- [Multi-Scale Attention-Driven Hierarchical Learning](https://www.mdpi.com/2079-9292/14/14/2869)
- [Multi-scale Feature Enhancement in Multi-task Learning](https://arxiv.org/html/2412.00351v1)
- [Medical image segmentation with UNet-based multi-scale context fusion](https://www.nature.com/articles/s41598-024-66585-x)
- [Feature Pyramid Network for Multi-Scale Detection](https://rumn.medium.com/feature-pyramid-network-for-multi-scale-detection-f573a889c7b1)
- [Multi-scale Unified Network for Image Classification](https://arxiv.org/html/2403.18294v1)

### Multi-Task & Transfer Learning
- [Transfer Learning to Learn with Multitask Neural Model Search](https://arxiv.org/abs/1710.10776)
- [Task Adaptation of RL-Based NAS Agents](https://arxiv.org/html/2412.01420v1)
- [Meta, Multi-Task, and Transfer Learning: A meta review](https://arxiv.org/html/2111.12146v7)
- [Multi-task learning (Wikipedia)](https://en.wikipedia.org/wiki/Multi-task_learning)
- [Multi-Task Learning Guide (V7 Labs)](https://www.v7labs.com/blog/multi-task-learning-guide)
- [An Overview of Multi-Task Learning](https://www.ruder.io/multi-task/)

### Additional NCA Research
- [Growing Neural Cellular Automata (Distill)](https://distill.pub/2020/growing-ca/)
- [A Path to Universal Neural Cellular Automata](https://arxiv.org/pdf/2505.13058)
- [Multi-texture synthesis through signal responsive neural cellular automata](https://www.nature.com/articles/s41598-025-23997-7)
- [Latent Neural Cellular Automata](https://arxiv.org/html/2403.15525)
- [Neural Cellular Automata for ARC-AGI](https://arxiv.org/html/2506.15746v1)

---

*Research conducted: 2026-02-14*
*Researcher: Bagel (NanoClaw research agent)*
