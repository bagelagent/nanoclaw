# Large-scale Pretraining for Neural Cellular Automata

**Research Topic:** Can foundation NCAs be pretrained and fine-tuned like diffusion models?

**Research Date:** 2026-02-13

## Summary

The concept of "foundation NCAs" trained at scale and fine-tuned for downstream tasks remains largely unexplored as of early 2026. However, recent research reveals several promising directions: (1) Multi-texture synthesis using genomic signal conditioning in a single NCA model, (2) AdaNCA as plug-and-play adaptors for Vision Transformers, (3) CLIP-guided discovery of novel CA behaviors, and (4) Attention-based NCAs that incorporate transformer mechanisms. While true foundation model paradigms (pretraining → fine-tuning) haven't been established for NCAs, the field is rapidly converging toward more generalizable, parameter-efficient architectures that share conceptual parallels with modern foundation models.

## Key Findings

### 1. Current State: Per-Task Training vs. Universal Models

**The Fragmentation Problem:**
Traditional NCA research trains separate models for each texture or task. The original [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/) (Mordvintsev et al., 2020) uses a frozen VGG-19 discriminator to guide training, but each target pattern requires a new training run. The NCA learns to match VGG feature statistics (gram matrices) of a specific texture, making each model highly specialized.

**Emergence of Multi-Task Approaches:**
Recent work addresses this limitation. [Multi-texture synthesis through signal responsive NCAs](https://www.nature.com/articles/s41598-025-23997-7) (Catrina et al., 2025) introduces **genomic signals** — binary codes embedded in hidden cell state channels that specify texture identity. With 3 genome channels, a single NCA can learn 2³ = 8 different textures. This represents a shift toward more versatile models, though still far from the scale of foundation models.

### 2. Parameter Efficiency: NCAs vs. Traditional Networks

NCAs achieve extraordinary parameter efficiency compared to standard deep learning architectures:

- **μNCA:** [68-8000 parameters](https://arxiv.org/abs/2111.13545) for texture synthesis
- **Diff-NCA:** [336k parameters](https://www.nature.com/articles/s44335-025-00026-4) for 512×512 pathology image generation
- **FourierDiff-NCA:** 1.1M parameters achieving 2× lower FID than 4× larger UNet

For comparison, diffusion models like Stable Diffusion require billions of parameters. This suggests that **if foundation NCAs emerge, they could be dramatically more compact** than current foundation models — possibly in the tens to hundreds of millions of parameters rather than billions.

### 3. Architectural Innovations Toward Generalizability

#### A. Attention-Based NCAs

[Attention-based Neural Cellular Automata](https://arxiv.org/abs/2211.01233) (Tesfaldet & Fleet, 2022) introduce **Vision Transformer Cellular Automata (ViTCA)**, combining self-attention with NCA local update rules. This architecture:
- Achieves superior performance on denoising autoencoding benchmarks
- Circumvents quadratic complexity of full self-attention through spatial localization
- Represents the first integration of transformer mechanisms into NCAs

#### B. AdaNCA: Adaptors for Vision Transformers

[AdaNCA](https://arxiv.org/html/2406.08298) (2024) uses NCAs as **plug-and-play modules** inserted between pretrained ViT layers:
- <3% parameter increase → >10% absolute accuracy improvement under adversarial attacks
- Enhances robustness across 8 benchmarks and 4 ViT architectures
- Critical limitation: **Cannot generalize to unseen recurrent steps** — trained with [3,5] steps, fails at step 6+

This approach is philosophically similar to fine-tuning, but rather than adjusting existing weights, it introduces new architectural components. It demonstrates that **NCAs can enhance pretrained models**, though it doesn't enable zero-shot task adaptation.

#### C. Universal NCAs

[A Path to Universal Neural Cellular Automata](https://arxiv.org/html/2505.13058v1) (2025) explores NCAs that encode fundamental computational primitives:
- Matrix multiplication and inversion
- Emulation of neural networks (solving MNIST)
- Separation of update model (maximal generality) from task-specific configurations

This work suggests NCAs can become general-purpose computational substrates, but practical pretraining approaches remain undemonstrated.

### 4. Foundation Models Guiding NCA Discovery

While pretraining NCAs themselves remains unexplored, **using pretrained foundation models to guide NCAs** has shown remarkable success:

#### CLIP-Guided Cellular Automata Discovery

[Automating the Search for Artificial Life with Foundation Models](https://dspace.mit.edu/bitstream/handle/1721.1/163679/kumar-akumar01-sm-eecs-2025-thesis.pdf) (Kumar, MIT 2025) uses **CLIP** to discover novel Lenia and CA behaviors:
- CLIP evaluates population fitness via semantic similarity between text prompts and simulation outputs
- Discovers previously unseen Lenia lifeforms resembling microscopy images
- Works across diverse ALife substrates: Boids, Particle Life, Game of Life, Lenia, NCAs

This demonstrates **foundation models can guide parameter search**, but not yet learn reusable NCA representations.

### 5. Scaling Laws and Data Requirements

Current research reveals fundamental differences from traditional deep learning scaling:

**Minimal Data Requirements:**
- NCAs can train on [**single images**](https://link.springer.com/article/10.1007/s40192-023-00335-1) with statistical descriptors
- Training data size is **independent of output resolution** — models trained at 64×64 synthesize 512×512+

**Training Challenges:**
- [Extensive hyperparameter tuning required](https://arxiv.org/html/2506.15746) for complex patterns
- Quadratic growth in memory/time with grid size limits resolution (typically ≤128×128)
- No established scaling laws like those for transformers (performance vs. compute/data)

**Parameter Count vs. Performance:**
- Medical image segmentation: NCAs with [**~1000× fewer parameters than U-Net**](https://arxiv.org/html/2408.15557) achieve better out-of-domain performance
- Suggests NCAs may benefit from **different optimization strategies** than standard scaling

### 6. Transfer Learning: Current Approaches

The closest existing analogs to fine-tuning:

**VGG-Based Feature Matching:**
- [Self-Organising Textures](https://distill.pub/selforg/2021/textures/) uses frozen VGG-19 for perceptual loss
- NCA learns to match target texture's VGG feature statistics
- This is **one-way transfer** — VGG guides NCA, but NCA doesn't build on pretrained representations

**Signal-Responsive Conditioning:**
- [Multi-texture NCAs](https://arxiv.org/html/2407.05991v2) use genome channels as learned internal representations
- Enables interpolation between textures and grafting techniques
- More analogous to **conditional generation** than fine-tuning

**Hybrid NCA-Diffusion Models:**
- [Diff-NCA and FourierDiff-NCA](https://www.nature.com/articles/s44335-025-00026-4) integrate NCA local dynamics into diffusion timesteps
- Achieves diffusion-quality output with NCA-level efficiency
- Represents **architectural fusion** rather than pretraining/fine-tuning

## Deep Dive: Why Foundation NCAs Don't (Yet) Exist

### Conceptual Barriers

**1. Architectural Mismatch**
Foundation models (LLMs, diffusion models, ViTs) share a common pattern:
- Large, centralized computation (transformer layers, UNet bottlenecks)
- Global information aggregation
- Direct input → output mapping

NCAs operate fundamentally differently:
- Purely local interactions (3×3 or 5×5 neighborhoods)
- Emergent global behavior through iterated local rules
- Recurrent, self-organizing dynamics

**2. Task Representation**
Foundation models are pretrained on diverse datasets (billions of images, trillions of tokens), learning representations that transfer across tasks. NCAs currently require:
- Explicit task encoding (genomic signals, target statistics)
- Problem-specific stopping criteria
- Domain-specific architectures (2D grids for images, 3D for volumes)

**3. Optimization Challenges**
- **Gradient flow:** Backpropagation through many CA steps (100-1000+) creates vanishing/exploding gradients
- **Stability:** NCAs must remain stable across arbitrary step counts, unlike feedforward networks with fixed depth
- **Expressivity vs. compactness:** Foundation models benefit from overparameterization; NCAs excel at extreme compactness

### Technical Requirements for Foundation NCAs

If foundation NCAs are to emerge, they likely need:

#### 1. **Pretraining Objective**
Current NCAs use task-specific losses (texture matching, morphogenesis). A foundation NCA would need:
- Self-supervised pretraining (masked pattern prediction? contrastive learning on patterns?)
- Multi-task objectives capturing diverse spatial dynamics
- Reward models for assessing "useful" emergent behaviors

#### 2. **Conditioning Mechanism**
Genomic signals show promise, but foundation models typically use:
- **Text conditioning** (CLIP embeddings, attention over text tokens)
- **Class embeddings** (learned vectors for categorical tasks)
- **Continuous control** (diffusion timesteps, classifier-free guidance)

A foundation NCA might embed task descriptions in cell state or use attention to query external memory.

#### 3. **Architecture Scaling**
Current NCAs: 68-8000 parameters (μNCA), 336k (Diff-NCA), 1.1M (FourierDiff-NCA)
Foundation models: 100M-1B+ parameters

Potential scaling strategies:
- **Deeper networks:** Current NCAs use 1-2 layer MLPs; could scale to 10+ layers
- **Richer cell states:** Increase hidden channels from 16-32 to 128-256
- **Hierarchical NCAs:** Multi-scale grids with communication between scales
- **Mixture of experts:** Different update rules activated by cell state

#### 4. **Training Infrastructure**
- **Hardware acceleration:** [CAX (Cellular Automata Accelerated in JAX)](https://arxiv.org/html/2410.02651v1) enables massive parallelization across GPUs/TPUs
- **Automatic differentiation:** JAX provides backpropagation through time for NCA training
- **Distributed training:** Scale to billions of CA steps across many devices

#### 5. **Evaluation Framework**
Foundation models are evaluated on benchmark suites (GLUE for LLMs, ImageNet for vision). NCAs would need:
- Standard tasks: texture synthesis, pattern completion, morphogenesis, dynamic evolution
- Zero-shot evaluation: Can pretrained NCA adapt to unseen pattern types?
- Transfer metrics: How much fine-tuning data needed for new domains?

## Connections to Existing Knowledge

### 1. Diffusion Models Parallel
Diffusion models iteratively denoise images through learned timestep-conditioned networks. NCAs iteratively refine patterns through learned neighborhood-conditioned updates. **Diff-NCA** makes this parallel explicit by replacing UNet with NCA dynamics, achieving 336k parameter diffusion models.

Could we pretrain a "foundational denoising NCA" on diverse image datasets, then fine-tune for specific styles? This remains unexplored.

### 2. Biological Development Metaphor
Biological cells share the same genome but differentiate into specialized types (neurons, muscle, skin). Multi-texture NCAs with genomic signals approximate this: all cells run the same update rule but interpret genome channels differently.

A foundation NCA would be the ultimate generalization: a single "universal genome" that, given appropriate conditioning, grows arbitrary patterns.

### 3. Meta-Learning Connection
[Meta-learning for NCAs](https://link.springer.com/article/10.1007/s10994-023-06334-9) (previous research) suggests NCAs can learn to discover parameter spaces. Could meta-learning enable **NCA-generating NCAs** — a pretrained system that outputs task-specific automata?

### 4. Cellular Models in Biology
Recent work on [Large Cellular Models (LCMs)](https://academic.oup.com/nsr/article/11/11/nwae340/7775526) for single-cell transcriptomics shows pretraining on massive biological data enables superior cell type annotation and perturbation prediction. This demonstrates **cellular-level foundation models are viable** — but in discrete symbolic space (gene expression) rather than continuous spatial dynamics.

## Follow-Up Questions & New Research Directions

Based on this research, several promising directions emerge:

### 1. **Self-Supervised Pretraining for NCAs** (Priority: 8)
Can NCAs be pretrained with masked pattern prediction (mask regions of evolved patterns, train NCA to reconstruct)? Would contrastive learning on pattern dynamics enable transferable representations?

### 2. **CLIP-Conditioned NCAs** (Priority: 7)
CLIP guides CA parameter search, but could CLIP embeddings **directly condition NCA cell states**? This would enable text → texture synthesis with ~1M parameter models instead of billion-parameter diffusion models.

### 3. **Hierarchical Foundation NCAs** (Priority: 7)
Could multi-scale NCAs with cross-scale communication achieve foundation model capabilities? Lower resolution = global planning, higher resolution = local detail refinement.

### 4. **Scaling Laws for NCAs** (Priority: 6)
What are the relationships between:
- Parameter count vs. pattern complexity
- Training compute vs. generalization
- Cell state dimension vs. task diversity

This is critical for guiding foundation NCA development.

### 5. **NCA Model Zoos** (Priority: 6)
If individual NCAs remain specialized, could a **collection of pretrained NCAs** with a learned router approximate foundation model behavior? (Similar to mixture-of-experts or LLM routing systems.)

### 6. **Differentiable Genome Evolution** (Priority: 5)
Can gradient-based optimization discover optimal genomic encodings for multi-task NCAs? Current approaches use binary codes; learned continuous embeddings might enable more tasks per model.

### 7. **NCA Fine-Tuning Protocols** (Priority: 5)
Even without pretraining, systematic study of fine-tuning strategies (freezing vs. adapting layers, learning rate schedules, few-shot vs. full data) would establish best practices.

### 8. **Benchmarking Zero-Shot NCA Transfer** (Priority: 5)
Create evaluation suite: train NCA on texture class A, test on texture class B without fine-tuning. Measure: reconstruction quality, semantic coherence, failure modes.

## Sources

### Neural Cellular Automata: Core Research

- [Neural cellular automata: applications to biology and beyond classical AI](https://arxiv.org/abs/2509.11131) — 2025 survey paper
- [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/) — Seminal work by Mordvintsev et al.
- [Self-Organising Textures](https://distill.pub/selforg/2021/textures/) — VGG-guided texture synthesis
- [μNCA: Texture Generation with Ultra-Compact Neural Cellular Automata](https://arxiv.org/abs/2111.13545) — 68-8000 parameter models
- [Neural Cellular Automata: From Cells to Pixels](https://arxiv.org/html/2506.22899v2) — Resolution scaling challenges

### Multi-Task and Conditional NCAs

- [Multi-texture synthesis through signal responsive neural cellular automata](https://www.nature.com/articles/s41598-025-23997-7) — Genomic signal conditioning (2025)
- [Multi-Texture Synthesis through Signal Responsive Neural Cellular Automata (arXiv)](https://arxiv.org/html/2407.05991v2) — Technical details
- [DyNCA: Real-Time Dynamic Texture Synthesis](https://dynca.github.io/) — Dynamic video textures

### Attention and Transformer-Based NCAs

- [Attention-based Neural Cellular Automata](https://arxiv.org/abs/2211.01233) — ViTCA architecture (2022)
- [Attention-based Neural Cellular Automata (NeurIPS proceedings)](https://proceedings.neurips.cc/paper_files/paper/2022/file/361e5112d2eca09513bbd266e4b2d2be-Paper-Conference.pdf)
- [AdaNCA: Neural Cellular Automata as Adaptors for Vision Transformers](https://arxiv.org/html/2406.08298) — Plug-and-play robustness (2024)

### Hybrid NCA-Diffusion Models

- [Parameter-efficient diffusion with neural cellular automata](https://www.nature.com/articles/s44335-025-00026-4) — Diff-NCA and FourierDiff-NCA (2025)

### Universal and General-Purpose NCAs

- [A Path to Universal Neural Cellular Automata](https://arxiv.org/html/2505.13058v1) — Computational primitives (2025)
- [Neural Cellular Automata for ARC-AGI](https://arxiv.org/html/2506.15746) — Abstract reasoning tasks (2025)

### Foundation Models Guiding NCAs

- [Automating the Search for Artificial Life with Foundation Models (MIT thesis)](https://dspace.mit.edu/bitstream/handle/1721.1/163679/kumar-akumar01-sm-eecs-2025-thesis.pdf) — CLIP-guided CA discovery
- [Guiding Evolution of Artificial Life Using Vision-Language Models](https://arxiv.org/pdf/2509.22447)

### Scaling, Performance, and Efficiency

- [Generalization Capabilities of Neural Cellular Automata for Medical Image Segmentation](https://arxiv.org/html/2408.15557) — 1000× fewer parameters than U-Net
- [CAX: Cellular Automata Accelerated in JAX](https://arxiv.org/html/2410.02651v1) — Hardware acceleration
- [Learning spatio-temporal patterns with Neural Cellular Automata](https://pmc.ncbi.nlm.nih.gov/articles/PMC11078362/)

### Training Data and Requirements

- [Reconstructing Microstructures From Statistical Descriptors Using Neural Cellular Automata](https://link.springer.com/article/10.1007/s40192-023-00335-1) — Single-image training

### Related: Large Cellular Models in Biology

- [General-purpose pre-trained large cellular models for single-cell transcriptomics](https://academic.oup.com/nsr/article/11/11/nwae340/7775526) — Foundation models for gene expression
- [Current opinions on large cellular models](https://onlinelibrary.wiley.com/doi/full/10.1002/qub2.65)

### Self-Supervised Learning Context

- [Understanding Self-Supervised Pretraining with Part-Aware Representation Learning](https://arxiv.org/abs/2301.11915)
- [Rethinking Pre-training and Self-training](https://proceedings.neurips.cc/paper/2020/file/27e9661e033a73a6ad8cefcde965c54d-Paper.pdf)
- [Self-supervised pre-training with contrastive and masked autoencoder methods](https://www.nature.com/articles/s41598-023-46433-0)

### Emergent Behavior and Zero-Shot Learning

- [Emergent Complexity and Zero-shot Transfer via Unsupervised Environment Design](http://aima.eecs.berkeley.edu/~russell/papers/neurips20-paired.pdf)
- [Emergent Abilities in Large Language Models: A Survey](https://arxiv.org/html/2503.05788v2)

### Awesome Lists and Resources

- [awesome-neural-cellular-automata (GitHub)](https://github.com/dwoiwode/awesome-neural-cellular-automata)

---

**Research completed:** 2026-02-13
**Next recommended topic:** Self-supervised pretraining strategies for NCAs (priority 8)
