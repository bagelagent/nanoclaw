# Load Balancing in Spatial MoE Systems: Do NCAs Need Auxiliary Losses?

**Research ID:** rq-1770915400000-nca-load-balancing
**Date:** 2026-02-15
**Tags:** neural-networks, mixture-of-experts, load-balancing, spatial-routing, nca

---

## Summary

Load balancing in Mixture-of-Experts (MoE) systems prevents expert collapse—when only a few experts receive most inputs while others remain underutilized. In LLM MoE architectures, auxiliary losses explicitly penalize routing imbalances by encouraging uniform expert utilization. But do spatial MoE systems like Neural Cellular Automata need similar mechanisms?

**Key Finding:** NCAs operate in a fundamentally different paradigm than token-based MoE systems. The architectural mismatch creates distinct collapse mechanisms and mitigation strategies. Standard NCAs with homogeneous update rules don't route to different "experts"—they achieve heterogeneous behavior through emergent dynamics from uniform rules. Spatial MoE-NCAs like Mixtures of NCAs (MNCA) and Petri Dish NCA do implement per-cell routing but address load balancing through **competitive dynamics, environmental pressure, and stochastic mechanisms** rather than auxiliary losses.

---

## Deep Dive

### 1. The Expert Collapse Problem in Token-Based MoE

In transformer-based MoE systems, unbalanced expert load leads to **routing collapse**—a degenerative state where the gating network converges to activate only a small subset of experts, which self-reinforces as favored experts receive more training.

**Traditional Mitigation: Auxiliary Losses**

The most common approach comes from the Switch Transformer (Fedus et al., 2022), which adds a load balancing loss:

```
L_balance = α · dot(f, P)
```

Where:
- `f` = fraction of tokens routed to each expert
- `P` = fraction of router probability allocated to each expert
- `α` = scaling coefficient

This loss is differentiable and creates gradients proportional to how overloaded each expert is, providing downward pressure on router probabilities for overloaded experts.

**Recent Innovations Beyond Auxiliary Loss (2024-2026)**

The field has evolved toward **auxiliary-loss-free approaches**:

1. **Loss-Free Balancing (DeepSeek-V3)**: Applies expert-wise bias to routing scores before top-K routing, dynamically updating bias based on recent load to maintain balance without interference gradients.

2. **Similarity-Preserving Routing (Omi et al., 2025)**: Enforces consistent expert assignment for related inputs, yielding provable reductions in expert collapse and variance over long training horizons.

3. **Expert Choice Routing**: Inverts perspective—each expert picks its top-K tokens rather than tokens selecting experts, guaranteeing perfect load balance.

4. **Adaptive Auxiliary Losses**: Layer-wise dynamic coefficients that update load-balance regularization in response to token-drop rates, preventing expert starvation.

**The Core Challenge**

Auxiliary losses introduce tension: strong balancing can overshadow the primary objective (language modeling), degrading expert specialization and routing accuracy. The goal is preventing degenerate collapse *without* compromising what makes experts valuable—their specialization.

### 2. Spatial Routing: A Different Paradigm

**Token-Based vs. Spatial Routing**

Token-based MoE makes **independent routing decisions per token**, treating each token as an isolated unit regardless of context or relationships.

Spatial MoE systems operate on **structured 2D/3D grids** where spatial relationships matter. Each location (pixel, grid cell, lattice point) requires an expert selection, but neighboring locations exhibit correlated structure.

**Spatial Mixture-of-Experts (SMOE)**

The Spatial MoE architecture (Dryden et al., NeurIPS 2022) uses a learned gating function to route from a shared set of experts to each spatial location (e.g., pixel) in an input sample. Key characteristics:

- **Fine-grained routing**: Expert selection happens at each spatial location
- **Structural learning**: Gate learns underlying spatial structure, routing specific experts to different "regions"
- **Regional specialization**: Experts specialize to unique characteristics of spatial regions

**Training Techniques for SMOE**

The authors developed specialized training methods including:
- **Self-supervised routing loss**: Encourages routing consistency within spatial regions
- **Damping expert errors**: Prevents error propagation across spatial neighborhoods

SMOE achieved state-of-the-art performance on weather prediction tasks, demonstrating that spatial routing can leverage domain structure (latitude/longitude grids, atmospheric dynamics) that global token-based routing ignores.

**Load Balancing in Spatial Systems**

Weather forecasting architectures like **ARROW** (Adaptive-Rollout Multi-scale temporal Routing) use Shared-Private Mixture-of-Experts to capture both shared patterns and scale-specific characteristics. **ClimateLLM** introduces Frequency Mixture-of-Experts (FMoE) using 2D FFT to analyze spatial patterns in frequency domain.

These systems implement load balancing through:
- Token-level KL divergence penalties for skewed traffic
- Utilization-variance terms
- Adaptive rebalancing responding to input distributions
- Expert-wise regularization for parameter diversity

### 3. Neural Cellular Automata: Local Computation, Emergent Heterogeneity

**Core Architecture**

Standard NCAs implement a fundamentally different paradigm:

- **Homogeneous update rules**: Every cell runs the *same* learned neural network
- **Local computation only**: Each cell processes only its neighborhood (typically 3×3 Moore or von Neumann)
- **Iterative dynamics**: Repeated application of the update rule produces emergent complexity

This is **not** a mixture-of-experts architecture in the traditional sense—there are no different "experts" to route between. All cells share identical parameters.

**How Heterogeneous Behavior Emerges from Homogeneous Rules**

Despite uniform update rules, NCAs achieve **heterogeneous cell behavior and specialization** through:

1. **Emergent dynamics**: Simple local rules produce complex global patterns
2. **State-dependent computation**: The same rule applied to different local states produces different outcomes
3. **Self-organization**: Cells differentiate through iterative interactions with spatially-varying neighborhoods

This mirrors biological development: identical genetic instructions (homogeneous rules) produce specialized cell types (heterogeneous behavior) through position, timing, and local signaling.

**Training Dynamics and "Collapse" in Standard NCAs**

NCAs face different collapse mechanisms:

- **Gradient flow challenges**: When training on many timesteps with loss only on final state, gradients must backpropagate through long iterative trajectories
- **Error compounding**: Small errors accumulate over time
- **Dead cell problem**: Cells can converge to inactive states from which they cannot recover
- **Homogenization risk**: All cells could converge to identical behavior if the learned rule lacks sufficient state-dependent branching

**Mitigation strategies** differ from MoE auxiliary losses:

- **Incremental additive updates**: Network outputs Δstate rather than full new state, improving gradient flow
- **Stochastic masking**: Randomly mask cell updates during training with varying probabilities, improving robustness
- **Constant supervision**: Measure loss at multiple timesteps, not just final state
- **Overflow prevention**: Clip or normalize states to prevent numerical instability

### 4. Mixture-of-NCAs: When NCAs Meet MoE

Recent work introduces true MoE architectures for NCAs:

#### **Mixture of Neural Cellular Automata (MNCA)**

Architecture (arxiv:2506.20486):

- **Multiple update rules**: System maintains K different NCA update rules ("experts")
- **Per-cell probabilistic routing**: At each location, a Rule Selector MLP outputs categorical distribution π(s_i^t, η) over rules
- **Gumbel-Softmax sampling**: Differentiable sampling via Gumbel-Softmax trick: z ~ Cat(π(s_i^t))
- **Stochastic dynamics**: Combines probabilistic rule assignment with intrinsic noise

**Critical Finding: No Auxiliary Losses for Load Balancing**

The MNCA paper contains **no mention** of:
- Auxiliary losses encouraging uniform rule utilization
- Techniques preventing collapse to single rule
- Load balancing strategies from MoE literature

The only regularization discussed: gradient normalization during training.

**Why might MNCAs not need explicit load balancing?**

1. **Biological motivation**: Real biological systems exhibit stochastic, heterogeneous dynamics without global coordination—modeling this requires diverse rule usage by construction
2. **Spatial locality constraint**: Each cell's rule selection depends only on local state, not global statistics, inherently limiting routing collapse patterns
3. **Stochastic sampling**: Categorical sampling with intrinsic noise provides exploration
4. **Different objective**: NCAs optimize for pattern emergence and robustness, not computational efficiency or expert utilization

Results demonstrate MNCAs achieve:
- Superior robustness to perturbations
- Better recapitulation of biological growth patterns
- Interpretable rule segmentation (can visualize which rules govern which regions)

#### **Petri Dish Neural Cellular Automata (PD-NCA)**

Architecture (Sakana AI):

- **Competitive dynamics**: Multiple independent NCA agents compete for spatial territory on shared 2D grid
- **Top-2 routing**: Threshold value of 0.4 allows up to 2 NCAs per cell, "inspired by Mixture-of-Experts Top-2 routing"
- **Strength-based arbitration**: Agents propose updates, competition through attack/defense channels determines contribution weights via softmax

**Collapse Prevention Mechanisms**

Unlike standard MoE, PD-NCA uses:

1. **Environmental competition**: Static background environment tensor continuously competes against all agents, preventing territorial stagnation
2. **Continual backpropagation**: Agents undergo "continual backpropagation throughout entire simulation," enabling real-time adaptation to competitive pressures
3. **Defensive requirements**: Agents must maintain active attack/defense even in controlled territory to resist environmental pressure

**Inverted Objective from MoE Load Balancing**

Standard MoE load balancing aims for **uniform expert utilization**—maximizing efficiency by ensuring all experts process similar numbers of tokens.

PD-NCA explicitly encourages **unequal, dynamic spatial distribution**—agents compete for territory. Load balancing would *defeat the purpose* of competitive dynamics.

This represents a **fundamentally different design philosophy**: competition as organizing principle, not computational efficiency.

### 5. The Architectural Mismatch

**Why Standard MoE Load Balancing Doesn't Apply to Spatial NCAs**

| Dimension | Token-Based MoE | Standard NCAs | Spatial MoE-NCAs |
|-----------|----------------|---------------|------------------|
| **Routing granularity** | Per token (independent) | No routing (homogeneous rules) | Per cell (spatially correlated) |
| **Expert count** | Fixed set (e.g., 8-64 experts) | Single shared rule | Small set (2-8 rules) |
| **Collapse mechanism** | Gating network converges to favor few experts | Cells converge to similar states; gradient flow issues | Potential: all cells select same rule |
| **Primary objective** | Efficient computation while maintaining quality | Pattern formation, robustness, self-organization | Modeling stochastic/heterogeneous dynamics |
| **Load balancing goal** | Uniform expert utilization for efficiency | Not applicable | Depends on application: competitive (uneven), biological (naturally diverse) |
| **Mitigation strategies** | Auxiliary losses, expert-choice routing, bias terms | Stochastic masking, additive updates, multi-step supervision | Environmental pressure, competitive dynamics, intrinsic noise |

**Critical Insight: Spatial Correlation Changes the Problem**

In token-based MoE, routing collapse happens because **independent routing decisions** amplify small biases through self-reinforcing gradients.

In spatial NCAs, **neighboring cells have correlated inputs** (similar local states). This introduces two competing forces:

1. **Collapse pressure**: If nearby cells benefit from same expert, spatial correlation could accelerate collapse across entire regions
2. **Anti-collapse from diversity**: If pattern formation *requires* different rules in different regions (e.g., edge detection rule vs. interior growth rule), the primary objective naturally opposes uniform routing

Whether auxiliary losses are needed depends on which force dominates for a given task.

### 6. When Do Spatial NCAs Need Load Balancing?

**Case 1: Auxiliary Losses NOT Needed**

**Scenario**: Task inherently requires heterogeneous spatial behavior

Examples:
- Biological growth modeling (MNCA): Different regions naturally use different rules (proliferation at boundary, differentiation in interior)
- Competitive multi-agent systems (PD-NCA): Unequal distribution is the *goal*
- Pattern formation with regional specialization: Task structure enforces diverse routing through primary loss

**Mechanism**: Primary objective naturally penalizes collapse. If all cells use the same rule and system fails to produce target pattern, gradients push toward diversity.

**Case 2: Auxiliary Losses HELPFUL**

**Scenario**: Task could succeed with homogeneous routing, but diversity improves robustness or generalization

Examples:
- Texture synthesis: Single rule could produce uniform texture, but diverse rules improve variety and robustness to perturbations
- Multi-texture genomic NCAs: System could collapse to using one "average" texture rule rather than learning distinct specialists

**Mechanism**: Primary objective satisfied by suboptimal solution. Auxiliary loss encourages exploration of diverse routing strategies.

**Case 3: Anti-Collapse Mechanisms Beyond Auxiliary Loss**

Spatial NCAs might use alternatives inspired by recent MoE innovations:

1. **Similarity-preserving routing**: Enforce consistent rule assignment for spatially-adjacent cells, stabilizing regional boundaries
2. **Environmental/competitive pressure**: External forcing that rewards or requires diverse behaviors (as in PD-NCA)
3. **Stochastic exploration**: Intrinsic noise in rule selection (as in MNCA) provides continuous exploration
4. **Self-supervised routing objectives**: Loss terms that encourage routing to respect spatial structure (as in SMOE)
5. **Rule dropout**: Randomly disable rules during training to prevent over-reliance on favorites

### 7. Open Research Questions

1. **Empirical characterization**: Under what conditions do MNCAs exhibit routing collapse without intervention? Need systematic studies across task types.

2. **Optimal routing granularity**: Should routing happen per-cell, per-region, or hierarchically across scales?

3. **Spatial correlation and load balancing**: How does spatial correlation in inputs affect collapse dynamics compared to independent token routing?

4. **Transfer from MoE literature**: Can recent innovations (loss-free balancing, adaptive auxiliary losses, similarity-preserving routing) transfer to spatial domains?

5. **Biological plausibility**: Real morphogenesis doesn't have global load balancing—cells differentiate through local signaling. Should NCAs modeling biology explicitly avoid global balancing objectives?

6. **Computational efficiency**: In deployed NCAs (e.g., real-time graphics), does load balancing matter for GPU utilization when different rules have different compute costs?

7. **Hierarchical spatial MoE**: Can hierarchical NCAs (coarse-to-fine architectures) benefit from scale-specific expert routing with cross-scale load balancing?

---

## Connections to Existing Knowledge

**From NCA Model Zoos Research**:
- Model zoo approach with learned routers can achieve 60-63% compute reduction
- Critical question from that research was whether load balancing would be needed—this investigation reveals it depends on architectural choice (homogeneous rules vs. true mixture) and task objectives

**From CLIP-Conditioned NCAs**:
- Gradient flow issues in NCAs parallel the challenge of long-horizon optimization in MoE training
- Both require careful training techniques to maintain stable learning

**From Hierarchical NCAs**:
- Multi-scale architectures might benefit from scale-specific expert routing
- Coarse scales handle global structure, fine scales handle local details—natural expert specialization

**From Diffusion-NCA Hybrids**:
- Hybrid models combining NCA efficiency with diffusion quality represent a different form of "mixture"—architectural blending rather than learned routing

**Broader Pattern**:
The load balancing question reveals a deeper insight: **NCAs succeed precisely because they avoid the architectural assumptions of transformers and MoE systems**. Local computation, homogeneous rules, and emergent complexity represent a different paradigm—one where "expert collapse" isn't the right conceptual frame.

When NCAs do adopt MoE-style routing (MNCA, PD-NCA), the spatial structure and task objectives create collapse dynamics distinct from token-based systems, requiring specialized solutions beyond auxiliary losses.

---

## Follow-Up Questions

1. **Practical experiment**: Train MNCAs with and without auxiliary load balancing losses across diverse tasks (texture synthesis, morphogenesis, maze solving). Measure: rule utilization variance, final performance, robustness to perturbations, training stability. Hypothesis: biological tasks don't need balancing; computational tasks benefit from it.

2. **Spatial correlation study**: How does neighborhood similarity affect routing collapse? Compare: independent random states vs. spatially-correlated states. Measure gradient magnitudes and routing entropy over training.

3. **Loss-free spatial balancing**: Can DeepSeek-V3's bias-based approach adapt to spatial domains? Implement expert-wise bias updated per local region rather than globally.

4. **Multi-scale routing**: Do hierarchical NCAs need different load balancing strategies at different scales? Explore whether coarse levels naturally balance but fine levels collapse.

5. **GPU utilization profiling**: In real-time NCA applications (games, interactive art), does heterogeneous rule usage create GPU bottlenecks? Compare wall-clock time for balanced vs. imbalanced routing.

---

## Sources

### Mixture-of-Experts Load Balancing
- [Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts (OpenReview)](https://openreview.net/forum?id=y1iU5czYpE)
- [Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts (arXiv)](https://arxiv.org/abs/2408.15664)
- [How MoE Models Actually Learn: A Guide to Auxiliary Losses and Expert Balancing (Medium)](https://medium.com/@chris.p.hughes10/how-moe-models-actually-learn-a-guide-to-auxiliary-losses-and-expert-balancing-293084e3f600)
- [Mixture of Experts in Large Language Models (arXiv)](https://arxiv.org/html/2507.11181v2)
- [Mixture-of-Experts (MoE) LLMs (Cameron R. Wolfe, Ph.D.)](https://cameronrwolfe.substack.com/p/moe-llms)
- [A Visual Guide to Mixture of Experts (MoE)](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts)
- [Mixture of Experts (MoE) Layer Insights (EmergentMind)](https://www.emergentmind.com/topics/mixture-of-experts-moe-layer)
- [Load Balancing Mixture of Experts with Similarity Preserving Routers (arXiv)](https://arxiv.org/html/2506.14038v1)
- [Auxiliary Balancing Loss: Preventing Expert Collapse in MoE (Michael Brenndoerfer)](https://mbrenndoerfer.com/writing/auxiliary-balancing-loss-mixture-of-experts-moe)
- [A Review on the Evolvement of Load Balancing Strategy in MoE LLMs (HuggingFace)](https://huggingface.co/blog/NormalUhr/moe-balance)
- [Auxiliary Loss Functions for MoE Load Balancing (APXML)](https://apxml.com/courses/mixture-of-experts/chapter-3-moe-training-dynamics-optimization/auxiliary-loss-load-balancing)
- [Mixture of Experts Explained (HuggingFace)](https://huggingface.co/blog/moe)
- [Advanced Modern LLM Part 5: MoE and Switch Transformer (Medium)](https://medium.com/@ikim1994914/advanced-modern-llm-part-5-mixture-of-experts-moe-and-switch-transformer-b3d1ce40ced2)
- [Understanding Mixture-of-Experts: Switch Transformer's Load Balancing vs. Mixtral's Natural Balance (Medium)](https://medium.com/@pilliudayaditya1207/understanding-mixture-of-experts-switch-transformers-load-balancing-vs-mixtral-s-natural-balance-25ed528cadfe)
- [Optimizing MoE Routers (arXiv)](https://arxiv.org/html/2506.16419v1)

### Spatial Mixture-of-Experts
- [Petri Dish Neural Cellular Automata (Sakana AI)](https://pub.sakana.ai/pdnca/)
- [Dynamic Load Balancing Based on Hypergraph Partitioning (MDPI)](https://www.mdpi.com/2220-9964/14/3/109)
- [Mixtures of Neural Cellular Automata (arXiv)](https://arxiv.org/html/2506.20486)
- [Mixture-of-Experts with Expert Choice Routing (Google Research)](https://research.google/blog/mixture-of-experts-with-expert-choice-routing/)
- [Spatial Mixture-of-Experts (arXiv)](https://arxiv.org/abs/2211.13491)
- [ARROW: Adaptive Rollout and Routing Method for Global Weather Forecasting (arXiv)](https://arxiv.org/html/2510.09734)
- [ClimateLLM: Efficient Weather Forecasting (arXiv)](https://arxiv.org/pdf/2502.11059)
- [Load-Balancing Strategies for Forecasting with MoE Architecture (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S1877050925035380)

### Neural Cellular Automata - Core Architecture
- [Growing Neural Cellular Automata (Distill.pub)](https://distill.pub/2020/growing-ca/)
- [Neural Cellular Automata (EmergentMind)](https://www.emergentmind.com/topics/neural-cellular-automata-ncas)
- [Neural cellular automata: applications to biology and beyond classical AI (arXiv)](https://arxiv.org/abs/2509.11131)
- [Neural Cellular Automata: From Cells to Pixels (arXiv)](https://arxiv.org/abs/2506.22899)
- [Learning spatio-temporal patterns with Neural Cellular Automata (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11078362/)
- [A Path to Universal Neural Cellular Automata (arXiv)](https://arxiv.org/html/2505.13058v1)
- [Neural Cellular Automata for ARC-AGI (arXiv)](https://arxiv.org/html/2506.15746v1)
- [Learning Graph Cellular Automata (NeurIPS)](https://proceedings.neurips.cc/paper/2021/file/af87f7cdcda223c41c3f3ef05a3aaeea-Paper.pdf)
- [A Sensitivity Analysis of Cellular Automata and Heterogeneous Topology Networks (arXiv)](https://arxiv.org/html/2407.18017)
- [Emergent Dynamics in Heterogeneous Life-Like Cellular Automata (arXiv)](https://arxiv.org/html/2406.13383v1)

### Related Topics
- [MoE Expert Specialization Collapse Prevention (APXML)](https://apxml.com/courses/mixture-of-experts/chapter-3-moe-training-dynamics-optimization/expert-specialization-collapse)
- [Understanding Mixture of Experts (MoE) Neural Networks (IntuitionLabs)](https://intuitionlabs.ai/articles/mixture-of-experts-moe-models)
- [Modern MoE Language Models (EmergentMind)](https://www.emergentmind.com/topics/modern-mixture-of-experts-moe-language-models)
