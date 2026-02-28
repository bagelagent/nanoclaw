# Training NCAs to Accept Perceptual Feedback Signals at Inference Time

**Research ID:** rq-1771762675170-nca-perceptual-feedback
**Completed:** 2026-02-22
**Tags:** nca, perceptual-guidance, real-time, dynamic-adaptation

## Summary

Neural Cellular Automata can be trained to accept various control signals (genomic, goal-based, environmental), but true perceptual feedback at inference time—where gradients flow through perceptual loss functions to dynamically guide generation—remains largely unexplored in NCA literature. However, recent advances in test-time adaptation, meta-gradient generation, and differentiable rendering provide promising pathways for enabling NCAs to respond to runtime perceptual feedback with minimal overhead.

## Key Findings

### 1. Current State: Signal-Responsive NCAs (Feedforward Only)

**Multi-texture Signal-Responsive NCAs** (Catrina et al., 2025) demonstrated that NCAs can be trained to respond to different types of signals:

- **Genomic signals**: Binary-encoded texture IDs placed in dedicated "genome channels" (e.g., 3 channels encode 8 textures as 000, 001, 010, etc.)
- **Signal propagation**: All cells receive identical genomic information at initialization; after timestep 1, the NCA must maintain this information internally
- **Training strategy**: Modified pool sampling that cycles through genome types to ensure balanced learning across all signal types
- **Interpolation capability**: By setting genome channels to intermediate values (e.g., 0.5), NCAs can blend between learned textures at inference

**Critical limitation**: These are purely feedforward at inference time. The genomic signal conditioning is learned during training via backpropagation through time, but at inference, the NCA simply applies its learned update rule without any gradient computation. As the paper explicitly states: "At inference time, there is **no gradient computation**."

**Goal-Guided NCAs** (2022) take this further by allowing goal encodings to "control cell behavior dynamically at every step of cellular growth," enabling the NCA to "continually change behavior" and generalize to unseen scenarios. Goals can change during growth, and the system is robust even when only a portion of cells receive goal information.

**Environmental Signal Response** (2023) showed NCAs can respond to both internal (genomic) and external (environmental) signals, with signals presented "to a single pixel for a single timestep"—triggering responses like color changes throughout the organism.

### 2. The Missing Piece: Runtime Gradient-Based Perceptual Feedback

None of these approaches use **gradient-based perceptual feedback at inference time**. They all rely on learned conditioning mechanisms that operate in feedforward mode. True perceptual feedback would involve:

1. **Forward pass**: NCA evolves for N steps
2. **Perceptual evaluation**: Compare output to target using VGG/CLIP/LPIPS features
3. **Backward pass**: Compute gradients through perceptual loss
4. **State update**: Adjust NCA state (or parameters) to reduce perceptual error
5. **Iterate**: Repeat for dynamic guidance

This is fundamentally different from current signal-responsive approaches, which pre-train the NCA to respond to discrete signals rather than continuous perceptual gradients.

### 3. Historical Context: Iterative Optimization for Neural Texture Synthesis

**Gatys et al. (2015-2016)** pioneered neural texture synthesis via iterative optimization:

- Started with **white noise images**
- Performed **gradient descent on pixel values** to match Gram matrix statistics from VGG features
- Required "hundreds of gradient descent iterations" to produce high-quality results
- Each iteration involved full forward+backward passes through VGG
- **Major limitation**: Extremely slow (several seconds for small images on high-end GPUs)

This approach proved that gradient-based perceptual feedback *works* but is computationally prohibitive for real-time applications. The solution at the time was to train **feedforward generators** (Johnson et al., 2016 "Perceptual Losses for Real-Time Style Transfer") that learned to approximate the result of iterative optimization, achieving 500× speedups by amortizing the optimization into trained weights.

### 4. Deep Image Prior: Inference-Time Optimization Without Training Data

**Deep Image Prior (DIP)** (Ulyanov et al., 2017) demonstrated that the **structure of a generator network captures image statistics** even without training:

- Optimize a **randomly-initialized network** on a single target image
- Network architecture acts as implicit prior, capturing natural image statistics
- Achieves excellent results in denoising, super-resolution, inpainting
- **No separate training data required**—optimizes at inference time

**Key insights for NCAs**:
- The network structure itself encodes useful priors (similar to NCA's local update rule)
- Inference-time optimization is viable but slow (minutes per image)
- **Early stopping is critical**: Models first learn desired content, then overfit to noise (ELTO behavior)

**Acceleration techniques**:
- **Deep Random Projector**: Optimize the input seed while freezing random weights (significant speedup)
- **Reduced network depth**: Fewer parameters to optimize
- **Self-guided methods**: Iteratively optimize both network and input

### 5. Test-Time Adaptation: Efficient Gradient Optimization at Inference

Recent breakthroughs in **Test-Time Adaptation (TTA)** (2024-2025) show how to efficiently optimize models at inference:

**Meta Gradient Generator (MGTTA)** (2024):
- **Problem**: Unsupervised losses (like entropy minimization) produce noisy, unreliable gradients
- **Solution**: Learn a meta-optimizer that refines gradients using historical patterns
- **Results**:
  - 4.2× faster adaptation speed than previous SOTA
  - Only 10 updates needed (vs hundreds for baselines)
  - Minimal overhead: +28MB GPU memory, faster than standard methods
- **Mechanism**:
  1. **Gradient Memory Layer**: Encode historical gradients into network parameters
  2. **Meta Gradient Generator**: Transform unreliable gradients into reliable descent directions

**Key insight for perceptual feedback**: Rather than directly optimizing with noisy perceptual losses, train a meta-optimizer that learns to generate good gradients from perceptual feedback signals. This could stabilize and accelerate runtime perceptual guidance for NCAs.

**Other TTA advances**:
- **Layer-wise dynamic allocation**: Allocate gradient updates non-uniformly across layers per sample
- **Query-only test-time training**: Use small gradient budgets for targeted context adaptation
- **Strategic warm-starting**: Partially relax parameters while preserving inductive bias

### 6. Reinforcement Learning + Perceptual Feedback (2024-2025 Surge)

The integration of RL with visual generative models has exploded: **13 papers (2019-2020) → 91 papers (2024-2025)**, with 77 papers in the first half of 2025 alone.

**Why RL for perceptual feedback?**
- Generative models typically train on surrogate objectives (likelihood, reconstruction loss) that "misalign with perceptual quality, semantic accuracy, or physical realism"
- RL provides "a principled framework for optimizing non-differentiable, preference-driven, and temporally structured objectives"
- Enables use of **non-differentiable reward sources**: VLM APIs, human feedback, discrete metrics

**Key challenge**: Methods requiring differentiable rewards limit applicable feedback sources. True perceptual feedback needs to handle both differentiable (VGG, LPIPS) and non-differentiable (CLIP scores, human preferences) signals.

### 7. Hybrid Approaches: NCAs at Scale with Implicit Decoders

**"Neural Cellular Automata: From Cells to Pixels"** (2026) addresses NCA scalability:

**Problem**: Traditional NCAs have:
- Training time/memory that scales quadratically with grid size
- Strictly local information propagation (impedes long-range communication)
- Heavy compute demands for real-time high-resolution inference

**Solution**: Pair a coarse-grid NCA with a lightweight implicit decoder:
- NCA evolves on low-resolution grid
- Decoder maps cell states + local coordinates → appearance attributes at arbitrary resolution
- Both components are local → highly parallelizable inference

**Result**: High-resolution outputs in real-time while preserving self-organizing behavior.

**Relevance to perceptual feedback**: This architecture separates spatial resolution from cellular dynamics. Perceptual feedback could target the coarse NCA grid (fewer cells = faster gradient computation) while the decoder handles high-resolution rendering. This could enable efficient runtime perceptual guidance even at high output resolutions.

### 8. Perceptual Loss in Modern Generative Models

**Diffusion + Perceptual Loss** (2024-2025):
- Perceptual loss (VGG, LPIPS, CLIP) integrated at training, inference, or latent stages
- "Significantly enhances FID and CLIP scores while reducing blurriness"
- **Inference-time guidance**: "Additional gradient computations and a regression head are introduced, at the cost of increased inference time"

**Key tradeoff**: Perceptual feedback improves quality but increases compute. Modern approaches focus on:
- Latent-space gradients (cheaper than pixel-space)
- Selective application (only when needed)
- Learned approximations (amortize gradient computation)

## Deep Dive: Pathways to Perceptual Feedback in NCAs

Based on the research, here are **four viable approaches** for training NCAs to accept perceptual feedback at inference:

### Approach 1: Direct Perceptual Optimization (Gatys-style)

**Method**: At inference, perform gradient descent on NCA states/inputs to minimize perceptual loss.

**Advantages**:
- Conceptually simple
- Maximum flexibility—any differentiable perceptual metric works
- No special training required

**Disadvantages**:
- Extremely slow (hundreds of iterations)
- Memory intensive (backprop through multiple NCA steps)
- Can overfit to noise (requires early stopping)

**Optimization strategies**:
- Use Deep Random Projector technique: freeze NCA weights, optimize only input seed
- Reduce NCA depth during perceptual refinement
- Apply perceptual feedback every K steps rather than every step
- Use perceptual loss only at final output (not intermediate states)

### Approach 2: Meta-Learned Perceptual Gradient Generator

**Method**: Train a meta-network that takes perceptual loss gradients and outputs refined gradients tailored for NCA dynamics.

**Training procedure**:
1. Pre-train NCA on diverse textures/patterns
2. During meta-training:
   - Evolve NCA to generate output
   - Compute perceptual loss gradient
   - MGG refines gradient
   - Apply refined gradient to NCA state
   - Measure whether perceptual error decreased
3. Train MGG to minimize perceptual error after K refinement steps

**Advantages**:
- 4-10× faster than direct optimization (based on MGTTA results)
- Learns to handle noisy perceptual signals
- Small overhead (~28MB)
- Only needs 10-20 iterations instead of hundreds

**Disadvantages**:
- Requires meta-training phase
- May not generalize to perceptual metrics not seen during meta-training

### Approach 3: Perceptual Signal Channels

**Method**: Extend signal-responsive NCA architecture to include perceptual error signals as additional input channels.

**Architecture**:
- Standard NCA state channels (RGBA, hidden)
- Genomic/goal channels (as in existing work)
- **New: Perceptual feedback channels** (e.g., 4-8 channels)

**Training procedure**:
1. Train NCA to generate diverse outputs
2. During training, periodically inject perceptual error signals:
   - Compute VGG/CLIP features for current state
   - Compute feature difference from target
   - Encode difference as spatial signal (e.g., L2 error per region)
   - Write signal to perceptual feedback channels
3. NCA learns to adjust dynamics based on feedback channels

**Inference**:
- Evolve NCA for N steps
- Compute perceptual error, encode to feedback channels
- Continue evolution with feedback active
- Repeat until convergence or max steps

**Advantages**:
- Purely feedforward (no backprop at inference)
- Aligns with existing signal-responsive NCA paradigm
- Fast inference (same speed as standard NCA)

**Disadvantages**:
- Limited expressiveness (fixed-size feedback channels)
- Requires curriculum training (simple→complex feedback)
- May not generalize to unseen perceptual metrics

### Approach 4: RL-Based Perceptual Feedback

**Method**: Train NCA update rule via reinforcement learning with perceptual quality as reward.

**Formulation**:
- **State**: Current NCA grid + target image features
- **Action**: NCA update parameters (or continuous action modulating update rule)
- **Reward**: Improvement in perceptual similarity (LPIPS, CLIP score, etc.)
- **Policy**: Neural network that maps state → update modulation

**Training**:
- Use policy gradient methods (PPO, SAC)
- Reward = ΔPerceptual_loss between consecutive steps
- Train on diverse texture/image generation tasks

**Advantages**:
- Can use non-differentiable perceptual metrics (human feedback, VLM APIs)
- Learns temporally structured behavior (plan ahead for later perceptual improvements)
- Naturally handles credit assignment across many NCA steps

**Disadvantages**:
- Sample inefficient (requires many training episodes)
- More complex training infrastructure
- Potential instability (RL training notoriously finicky)

## Connections to Existing Knowledge

### Relation to Texture Synthesis
- Traditional Gatys optimization: Perceptual feedback via pure gradient descent
- Feedforward generators: Amortize optimization into trained weights (fast but inflexible)
- **Proposed NCA approach**: Hybrid—feedforward NCA with lightweight runtime perceptual refinement

### Relation to Differentiable Rendering
- Differentiable renderers enable gradient flow from image space → scene parameters
- **Similar for NCAs**: Enable gradient flow from perceptual metrics → cellular states
- Key difference: NCAs have many update steps (temporal dimension), renderers typically single pass

### Relation to Neural Architecture Search
- NAS uses meta-learning to discover architectures
- **Perceptual feedback NCAs**: Meta-learn how to respond to perceptual gradients
- Both involve differentiating through iterative processes

### Relation to Self-Organizing Systems
- Biological cells respond to chemical gradients in environment
- **Perceptual feedback as artificial morphogen**: Perceptual error signals act as guiding gradient field
- Cells (NCA) sense local perceptual error and adjust behavior accordingly

## Follow-Up Questions

1. **Computational budget analysis**: What's the Pareto frontier of perceptual quality vs. inference compute for different approaches? Can we get 80% of Gatys quality with 10× less compute?

2. **Perceptual metric selection**: Which perceptual metrics (VGG, LPIPS, CLIP, learned) provide the strongest training signal for NCA perceptual feedback? Do hierarchical approaches (CLIP for coarse, VGG for fine) work better?

3. **Temporal credit assignment**: When perceptual feedback is applied every K steps, how should credit be assigned across intermediate NCA states? Does RL-style temporal difference learning help?

4. **Generalization**: Can an NCA trained with VGG perceptual feedback generalize to CLIP feedback at inference? Or does it overfit to the specific perceptual metric?

5. **Robustness to feedback noise**: How sensitive are perceptual-feedback NCAs to noisy or adversarial perceptual signals? Can meta-gradient approaches improve robustness?

6. **Multi-scale perceptual feedback**: Can hierarchical NCAs (coarse + fine grids) use different perceptual metrics at each scale? E.g., CLIP guidance at coarse scale, texture metrics at fine scale?

7. **Sample efficiency**: How many training examples are needed to train perceptual-feedback NCAs via different approaches (direct supervision, meta-learning, RL)?

8. **Real-time performance benchmarks**: What's achievable on consumer hardware (single GPU)? Can perceptual feedback run at 30fps for 512×512 outputs?

9. **Human-in-the-loop**: Can perceptual feedback channels accept human preference signals (click here to improve this region)? How to encode sparse human feedback spatially?

10. **Biological plausibility**: Do any of these approaches align with biological morphogen gradients and chemotaxis? Could insights from developmental biology inspire better perceptual feedback mechanisms?

## Sources

### Core NCA Research
- [Neural Cellular Automata: From Cells to Pixels (2026)](https://arxiv.org/abs/2506.22899)
- [Multi-texture synthesis through signal responsive neural cellular automata (2025)](https://www.nature.com/articles/s41598-025-23997-7)
- [Multi-Texture Synthesis through Signal Responsive Neural Cellular Automata (arXiv)](https://arxiv.org/html/2407.05991v2)
- [Neural Cellular Automata Can Respond to Signals (2023)](https://arxiv.org/abs/2305.12971)
- [Goal-Guided Neural Cellular Automata (2022)](https://arxiv.org/abs/2205.06806)
- [Growing Neural Cellular Automata (Distill, 2020)](https://distill.pub/2020/growing-ca/)

### Test-Time Adaptation & Meta-Learning
- [Learning to Generate Gradients for Test-Time Adaptation via Test-Time Training Layers (2024)](https://arxiv.org/abs/2412.16901)
- [Test-Time Adaptation for Unsupervised Combinatorial Optimization (2026)](https://arxiv.org/abs/2601.21048)
- [Learning on the Fly: Rapid Policy Adaptation via Differentiable Simulation](https://arxiv.org/html/2508.21065)

### Perceptual Loss & Neural Style Transfer
- [Texture Synthesis Using Convolutional Neural Networks (Gatys, NeurIPS 2015)](https://proceedings.neurips.cc/paper/2015/file/a5e00132373a7031000fd987a3c9f87b-Paper.pdf)
- [A Neural Algorithm of Artistic Style (Gatys, 2015)](https://arxiv.org/abs/1508.06576)
- [Perceptual Losses for Real-Time Style Transfer and Super-Resolution (Johnson et al., 2016)](https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16.pdf)

### Deep Image Prior
- [Deep Image Prior (2017)](https://arxiv.org/abs/1711.10925)
- [Deep Image Prior Website](https://dmitryulyanov.github.io/deep_image_prior)
- [Early Stopping for Deep Image Prior (2021)](https://arxiv.org/abs/2112.06074)

### Reinforcement Learning + Generative Models
- [Integrating Reinforcement Learning with Visual Generative Models (2025)](https://arxiv.org/html/2508.10316v1)

### Diffusion + Perceptual Loss
- [Perceptual Loss Function (Emergent Mind Topic)](https://www.emergentmind.com/topics/perceptual-loss-function)
- [Diffusion Model with Perceptual Loss (Emergent Mind Topic)](https://www.emergentmind.com/topics/diffusion-model-with-perceptual-loss)

### Neural Texture Synthesis & Real-Time Methods
- [Texture Networks: Feed-forward Synthesis of Textures and Stylized Images (2016)](http://proceedings.mlr.press/v48/ulyanov16.pdf)
- [Precomputed Real-Time Texture Synthesis with Markovian GANs (2016)](https://link.springer.com/chapter/10.1007/978-3-319-46487-9_43)

---

**Conclusion**: Training NCAs to accept perceptual feedback at inference is technically feasible via multiple pathways (direct optimization, meta-learning, signal channels, RL), each with distinct trade-offs. The most promising near-term approach appears to be **meta-learned gradient generators** (Approach 2), which combine the flexibility of gradient-based feedback with 4-10× speedups over naive optimization. For real-time applications, **perceptual signal channels** (Approach 3) offer feedforward inference with learned sensitivity to runtime perceptual guidance, though at the cost of reduced flexibility. The optimal choice depends on the application's requirements for speed, quality, and adaptability to novel perceptual metrics.
