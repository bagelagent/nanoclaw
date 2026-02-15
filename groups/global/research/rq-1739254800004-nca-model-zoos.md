# NCA Model Zoos with Learned Routers

**Research Question**: Can collections of specialized NCAs with learned routing mechanisms approximate foundation model behavior?

**Date**: 2026-02-14
**Status**: Completed
**Priority**: 6

---

## Summary

Collections of specialized Neural Cellular Automata with learned routing mechanisms represent a promising middle ground between single foundation models and fixed specialized models. While true "foundation NCAs" don't yet exist, research demonstrates three viable paths toward multi-task NCA systems: (1) **genomic signal encoding** enabling single NCAs to generate 2-8+ specialized outputs, (2) **mixture-of-NCAs architectures** with probabilistic rule routing, and (3) **attention-based gating** for universal computation. These approaches leverage MoE-style routing principles—learned gating networks that specialize experts during training—to achieve extreme parameter efficiency (68-8000 params per expert) while maintaining versatility. However, fundamental questions remain about whether local computation paradigms can achieve the global reasoning capabilities of true foundation models.

---

## Key Findings

### 1. Three Architectural Paradigms for NCA Model Zoos

Current research converges on three distinct approaches for building collections of specialized NCAs:

#### **Genomic Signal Encoding (Single Model, Multiple Outputs)**
- Single compact NCA generates multiple textures based on internal "genomic" channels
- **Capacity**: 2^ng textures where ng = number of genomic channels (1 bit = 2 textures, 2 bits = 4 textures, 3 bits = 8 textures)
- **Architecture**: Cell state splits into communication channels (nc) and genomic channels (ng)
- **Mechanism**: Binary-encoded texture identity initialized at t=0, preserved throughout evolution
- **Advantages**: Enables texture interpolation and grafting; single training process
- **Trade-offs**: Limited to texture synthesis domain; capacity grows exponentially but requires retraining for new textures

#### **Mixture of Neural Cellular Automata (MNCA)**
- Multiple distinct NCA rules (K rules, each with parameters θₖ) combined via probabilistic routing
- **Router Architecture**: Multi-Layer Perceptron "Rule Selector" processes cell state and outputs categorical distribution over K rules
- **Mechanism**: Each cell samples rule assignment z ~ Categorical(p), applies selected rule φₖ, optionally adds Gaussian noise
- **Advantages**: Captures diverse local behaviors and biological stochasticity; interpretable rule assignments enable post-hoc analysis
- **Trade-offs**: Requires careful balancing of rule diversity vs. coordination; training complexity increases with K

#### **Attention-Based Universal NCAs**
- Hardware vectors + attention gating enable task specialization within shared update rules
- **Architecture**: Cells receive perception vector (local spatial patterns) + hardware vector (task-specific information)
- **Routing Mechanism**: α = softmax((I·W_embed)/T) where hardware vector weights N parallel MLP pathways
- **Training Strategy**: Shared NCA parameters updated with gradients aggregated across all tasks; task-specific hardware parameters updated only with corresponding task gradients
- **Advantages**: True multi-task learning; general computational primitives emerge; demonstrated success on matrix operations and MNIST within CA state
- **Trade-offs**: Architectural complexity; requires careful task curriculum design

---

### 2. Mixture-of-Experts Routing Principles Applied to NCAs

The theoretical foundation for NCA model zoos draws heavily from MoE architectures in LLMs and transformers:

#### **Core MoE Mechanisms Relevant to NCAs**

**Learned Gating Networks:**
- Neural network routers (typically MLPs with softmax) analyze inputs and assign weights to experts
- In NCAs: Rule Selector networks or attention-based hardware conditioning play analogous roles
- Mathematical form: G_σ(x) = Softmax(x · W_g) where W_g are learned gating parameters

**Expert Specialization Through Training:**
- Positive feedback effect: if expert A is slightly better than expert B at a task, weighting favors A, A receives stronger gradients and improves further while B specializes elsewhere
- In NCAs: Different rules/pathways naturally diverge toward distinct pattern formation behaviors through gradient descent
- Critical for MNCA: Rule assignments become interpretable as rules cluster around specific cell behaviors

**Sparse Activation for Efficiency:**
- Only top-k experts activate per token, dramatically reducing compute
- In NCAs: Only selected rules execute per cell (MNCA) or only relevant attention pathways contribute (universal NCAs)
- Enables scaling to large expert counts (MoE models use 8-128+ experts) without proportional compute increase

**Load Balancing Challenges:**
- Risk of "expert collapse" where few experts dominate, others atrophy
- Solutions: auxiliary losses penalizing uneven distribution, noisy top-k gating, expert capacity limits
- **Open question for NCAs**: Do similar load balancing mechanisms apply? How do spatial patterns affect expert utilization?

#### **Routing Strategies and NCA Applications**

**Token-Choice Routing (Standard MoE):**
- Tokens select top-k experts based on router network output
- **NCA Analog**: Cells select which rule to apply (MNCA) or which computational pathway to activate (attention-based)
- Trade-off: Simple, effective, but can lead to load imbalance

**Expert-Choice Routing:**
- Experts select which tokens to process, achieving optimal load balancing
- **NCA Application**: Could reverse MNCA paradigm—rules "claim" cells they're best suited for
- **Unexplored in current research**

**Mixture vs. Switch Routing:**
- Mixture: Top-k experts with weighted average (k=2 common)
- Switch: Top-1 routing for maximum simplicity and speed
- **NCA Reality**: MNCA uses categorical sampling (effectively top-1), attention-based uses weighted mixture
- **Design choice**: Depends on whether hard specialization (switch) or soft blending (mixture) is desired

---

### 3. Model Zoo Paradigm vs. Foundation Model Paradigm

Fundamental tension between two approaches to multi-task AI:

#### **Foundation Models**
- **Definition**: Homogeneous, task-agnostic, single neural architecture trained on massive datasets with broad generalization
- **Strengths**: Transfer learning across domains; zero-shot capabilities on out-of-distribution tasks; emergent reasoning abilities
- **Weaknesses**: Computationally expensive (4-12GB VRAM for diffusion models); poor parameter efficiency; tightly coupled to pretraining distribution
- **NCA Limitation**: Current NCAs lack foundation-scale pretraining infrastructure; architectural mismatch (local computation vs. global reasoning)

#### **Model Zoos**
- **Definition**: Heterogeneous collections of specialized models optimized for narrow domains, requiring task-dependent selection
- **Strengths**: Domain-specific models can outperform foundation models with orders of magnitude fewer parameters; efficiency under data scarcity and distribution shift
- **Weaknesses**: Requires separate training for each task; no emergent cross-domain reasoning; routing/selection overhead
- **NCA Advantage**: Extreme compactness (68-8000 params) makes training large collections feasible; genomic signals enable single-model multi-task

#### **Which Paradigm for NCAs?**

Research suggests **hybrid path is most promising**:

1. **Pretrain universal NCA substrate** (attention-based architecture) on diverse pattern formation tasks
2. **Build specialized genomic/rule collections** for texture synthesis, morphogenesis, simulation domains
3. **Deploy learned routers** (RouteLLM-style preference-based or NAS-style performance predictors) to select appropriate NCA for queries
4. **Enable zero-shot generalization** via CLIP conditioning on pretrained NCAs

**Critical insight from foundation model research**: "Carefully designed, domain-adapted models can deliver competitive or even superior performance compared to large foundation models, while offering substantial advantages in efficiency, accessibility, and environmental sustainability."

For NCAs, this suggests model zoos with learned routing may be MORE appropriate than pursuing single foundation NCAs, given:
- NCAs' fundamental architecture (local computation) inherently differs from global-reasoning foundation models
- Extreme parameter efficiency (8000 params vs. millions) makes training specialized collections trivial
- Specialized NCAs already outperform larger models (Diff-NCA at 336k params outperforms 4× larger UNet)

---

### 4. Learned Routing Systems: Lessons from RouteLLM

RouteLLM framework offers concrete blueprint for NCA model zoos:

#### **Architecture Components**

**Four Router Types** (in order of complexity):
1. **Similarity-Weighted Ranking**: Weighted Elo scoring where votes weighted by similarity to query prompt
2. **Matrix Factorization**: Bilinear scoring function of model embeddings and query embeddings
3. **BERT Classifier**: Transformer-based classifier trained on preference data
4. **Causal LLM Classifier**: Large language model fine-tuned for routing decisions

#### **Training Methodology**
- **Primary signal**: Human preference data (query → pairwise model comparison → win/loss/tie label)
- **Data augmentation**: Synthetic generation to enhance coverage of edge cases
- **Key finding**: Routers learn generalizable principles—transfer learning works even when strong/weak model pairs change at test time

#### **Performance Metrics**
- 2× cost reduction without quality compromise on widely-recognized benchmarks
- Significant transfer capabilities across model pairs

#### **Application to NCA Model Zoos**

**Routing Decision Framework:**
```
Query: texture synthesis request (target aesthetic, resolution, real-time constraints)
  ↓
Router Analysis:
  - Quality requirement (photorealistic vs. organic/abstract)
  - Latency constraint (real-time 60fps vs. offline generation)
  - Domain (natural textures vs. abstract patterns vs. specific material types)
  ↓
Model Selection:
  - Real-time organic → μNCA (68-1000 params, 25Hz on CPU)
  - Interactive photorealistic → LCM/Diff-NCA (336k-1.1M params, <1s)
  - Maximum quality → Standard diffusion (millions of params, seconds)
  - Specific texture family → Genomic NCA with appropriate signal
```

**Training Data for NCA Router:**
- User preference annotations: "Which texture better matches request X?"
- Quality metrics: FID, LPIPS, perceptual texture similarity scores
- Performance metrics: Generation time, memory footprint, frame rate
- Domain classification: Material type, aesthetic category, structural complexity

**Expected Behavior:**
- Learn semantic texture space: abstract vs. realistic, organic vs. geometric, smooth vs. rough
- Predict quality-speed Pareto frontier for query
- Transfer across NCA architectures as new models are added to zoo

**Routing Strategies Specific to NCAs:**

**Cascade Routing** (try fast model first, fall back if quality insufficient):
```
Request → μNCA (1000 params, 40ms)
  → Quality check (learned threshold)
  → If pass: return; If fail: Diff-NCA (336k params, 800ms)
  → Quality check
  → If pass: return; If fail: Full diffusion (50M params, 5s)
```

**Achieved Savings**: RouteLLM demonstrates 60-63% cost reduction with cascades; NCAs could achieve similar compute reductions by routing ~60% of requests to ultra-lightweight NCAs, ~30% to mid-weight, ~10% to heavy models.

---

### 5. Open Questions and Research Gaps

#### **Fundamental Capabilities**
- **Can local computation achieve global reasoning?** NCAs excel at pattern formation but struggle with tasks requiring long-range dependencies or abstract reasoning. Hierarchical NCAs and attention mechanisms partially address this, but it's unclear if they can match transformer-scale reasoning.

- **What are NCA scaling laws?** Unlike transformer power laws (Kaplan, Chinchilla), NCAs exhibit fundamentally different scaling: capacity = parameters × iterations × receptive field. Missing: systematic empirical studies relating parameter count, training compute, and generalization across tasks.

- **Optimal routing granularity?** Should routers select entire NCA models, or route at finer granularity (per-layer, per-cell, per-timestep)? MNCA demonstrates per-cell routing works, but computational overhead vs. specialization benefits remain unexplored.

#### **Architecture Design**
- **Mixture vs. Switch in NCAs?** MoE literature debates top-1 (switch) vs. top-k (mixture) routing. For spatial automata, is hard rule selection (MNCA categorical sampling) better than soft attention-weighted mixing (universal NCAs)?

- **Load balancing in spatial domains?** MoE models require auxiliary losses to prevent expert collapse. Do spatial pattern dynamics naturally balance load across NCA rules, or do similar failures occur? What's the NCA equivalent of expert capacity?

- **Hierarchical routing strategies?** Should routers first select domain (texture vs. simulation vs. morphogenesis), then select specialized model within domain? Or end-to-end learned routing?

#### **Training and Specialization**
- **Fine-tuning protocols for NCA zoos?** When new texture families are added, should we: (1) train new specialized NCA from scratch, (2) fine-tune genomic NCA by expanding channel count, (3) add new rule to MNCA mixture? No systematic comparison exists.

- **Catastrophic forgetting in multi-task NCAs?** Continual learning challenge: can we add new tasks to universal NCAs without degrading performance on existing tasks? Hierarchical NCAs may enable adding/refining levels incrementally.

- **Optimal number of experts/rules?** MoE models scale to 128+ experts, but NCA studies use 2-8 rules (MNCA) or small numbers of parallel pathways. What's the sweet spot for pattern formation tasks?

#### **Routing Learning**
- **What training signal for NCA routers?** RouteLLM uses human preference data. For textures: user annotations? Quality metrics (FID/LPIPS)? Success at downstream task? Hybrid approach?

- **Transfer learning across NCA architectures?** RouteLLM routers transfer across model pairs. Can texture-domain routers transfer to morphogenesis domain? Can routing principles learned for NCAs transfer to other spatial models (reaction-diffusion, graph neural networks)?

- **Online learning and adaptation?** Should routers update based on user feedback during deployment, or remain static post-training? Risk of distribution shift vs. benefit of personalization.

#### **Production Deployment**
- **Inference infrastructure for NCA zoos?** MoE models require specialized distributed serving infrastructure. NCAs are tiny (68-8000 params), but serving 100+ specialized models still requires efficient loading, caching, and execution. WebGL? WASM? Model quantization?

- **Benchmarking methodology?** Need standardized evaluation: quality metrics, speed benchmarks, memory footprints, user preference studies across diverse texture families. Currently fragmented across papers.

- **Real-world production adoption?** Which shipped games/applications actually use NCA model zoos? Case studies missing. Most research demonstrates proof-of-concept, not production deployment lessons.

---

## Deep Dive: Architectural Pathways to NCA Model Zoos

### Path 1: Genomic Signal Model Zoos

**Current State**: Demonstrated feasibility for 2-8 textures from single NCA

**Scaling Strategy**:
- Increase genomic channel count: ng=4 → 16 textures, ng=5 → 32 textures
- **Challenge**: Exponential growth means training data requirements scale rapidly
- **Mitigation**: Self-supervised pretraining on pattern diversity; few-shot adaptation to new textures

**Routing Mechanism**:
- **User-side**: Select texture ID (binary encoding) → initialize genomic channels
- **No learned router needed**: User explicitly specifies desired output via genomic signal
- **Limitation**: Cannot handle continuous control or open-ended prompts

**Optimal Use Case**: Applications requiring fixed set of texture families with user selecting from catalog (game asset generation, procedural material libraries)

### Path 2: Mixture-of-NCAs with Probabilistic Routing

**Current State**: MNCA framework demonstrates stochastic rule routing for biological modeling

**Scaling Strategy**:
- Increase expert count: K=2-8 rules → K=16-32 rules (analogous to MoE scaling)
- **Challenge**: Load balancing—ensure all rules specialize rather than collapsing to few
- **Mitigation**: Auxiliary losses from MoE literature; noisy categorical sampling; expert capacity constraints adapted for spatial domains

**Routing Mechanism**:
- **Per-cell learned routing**: Rule Selector MLP processes cell state → categorical distribution over K rules
- **Spatial coherence consideration**: Should neighboring cells prefer similar rules (encourage pattern stability) or diverse rules (encourage boundary formation)?
- **Training signal**: Task loss (texture reconstruction) + auxiliary loss (balanced rule usage) + optional entropy regularization (encourage exploration)

**Optimal Use Case**: Single-image synthesis requiring heterogeneous local behaviors (hybrid textures, complex materials with multiple visual phenomena)

### Path 3: Universal NCA + Model Zoo Ensembles

**Current State**: Attention-based universal NCAs handle multiple computational tasks; no large-scale zoo deployment demonstrated

**Scaling Strategy**:
- **Pretrain universal NCA substrate** on diverse pattern formation tasks (texture synthesis, morphogenesis, simulation)
- **Build specialized zoo**: Fine-tune pretrained NCAs for specific domains (wood grain specialist, marble specialist, fabric specialist)
- **Deploy learned router**: RouteLLM-style classifier selects specialist based on query embedding

**Routing Mechanism**:
- **Query-level routing**: User provides text prompt or image exemplar → CLIP embedding → router network → select 1-3 specialized NCAs from zoo
- **Router architecture options**:
  - Similarity-weighted: Compare CLIP embedding to prototype embeddings for each specialist
  - Matrix factorization: Learn low-rank decomposition of (query × model) preference matrix
  - BERT/transformer classifier: Fine-tune small model on (query, model) → preference labels

**Optimal Use Case**: Open-domain texture synthesis with natural language prompts ("weathered oak planks," "silk fabric with subtle sheen")

### Path 4: Hybrid Cascade Systems

**Proposed Architecture** (not yet demonstrated in literature):

```
Stage 1: Fast Screening
  Input query → Lightweight MLP router (10k params)
    → Route 70% to μNCA (1000 params, real-time)
    → Route 20% to Diff-NCA (336k params, interactive)
    → Route 10% to specialized heavy models

Stage 2: Quality Gating
  Generated texture → Learned quality predictor
    → If quality score > threshold: accept
    → If quality score < threshold: route to next tier

Stage 3: Ensemble Refinement (optional)
  Blend outputs from multiple specialized NCAs for superior quality
```

**Advantages**:
- Achieves 60-63% compute reduction (demonstrated in RouteLLM)
- Graceful degradation: users get instant previews, high-quality results follow
- Learn optimal quality thresholds from user preference data

**Challenges**:
- Requires quality predictor training (FID/LPIPS alone insufficient; need perceptual metrics)
- Latency for multi-tier generation
- Ensemble blending may introduce artifacts at texture boundaries

---

## Connections to Existing Knowledge

### Relationship to Foundation NCA Research
- This research directly builds on prior work: "Large-scale pretraining for NCAs - can foundation NCAs be fine-tuned like diffusion models?" (rq-1739076481001) concluded foundation NCAs don't exist yet
- Model zoo approach offers **pragmatic alternative**: instead of single foundation model, build ecosystem of specialized models with learned routing
- Complements hierarchical NCA research (rq-1739254800002): hierarchical architectures can serve as individual experts in zoo

### Relationship to NCA vs. Diffusion Models
- Prior research (rq-1770852365000) established NCAs achieve 2-4 orders of magnitude faster inference than diffusion models
- Model zoo paradigm **amplifies this advantage**: routing 60%+ of queries to ultra-lightweight NCAs compounds speedup
- Hybrid systems can achieve "best of both worlds": fast NCAs for organic textures, diffusion for photorealism

### Relationship to Mixture-of-Experts in LLMs
- NCAs inherit MoE's core insight: **specialization through learned routing is more efficient than monolithic models**
- Key difference: NCAs operate on spatial grids (cells) vs. sequential tokens; routing can be per-cell (MNCA) or per-query (zoo selection)
- Open question: Do NCAs benefit from fine-grained routing (per-cell like MoE per-token) or coarse-grained routing (per-query)?

### Relationship to Neural Architecture Search
- NAS literature explores automated model selection based on predicted performance
- **Application to NCA zoos**: Router networks are essentially lightweight NAS predictors—given query features, predict which architecture will achieve best quality/speed trade-off
- NAS insight: "Factor-based task-module router that identifies latent graph factors and directs incoming tasks to the most appropriate architecture module" directly applicable to texture-domain routing

### Relationship to Production Model Serving
- Model zoos resemble production ML serving systems: collections of models behind intelligent routing layer
- **Critical lesson**: Infrastructure matters—efficient model loading, caching, batching as important as routing accuracy
- NCAs' extreme compactness (8k params) could enable novel deployment: entire zoo fits in browser via WASM, client-side routing, zero backend infrastructure

---

## Follow-up Questions for Future Research

1. **Can attention-based universal NCAs serve as pre-trained foundation models for fine-tuning specialists in a model zoo?** (Combines Path 3 with transfer learning)

2. **What is the Pareto frontier of (number of experts) × (parameters per expert) for fixed compute budget in NCA zoos?** (Systematic scaling study)

3. **Do spatial pattern dynamics require different load balancing mechanisms than token-based MoE models?** (Theoretical analysis of expert collapse in spatial domains)

4. **Can CLIP embeddings serve as universal routing signals for NCA model zoos across texture, morphogenesis, and simulation domains?** (Transfer learning for routing)

5. **What quality metrics correlate with human preference for textures generated by different NCAs in a zoo?** (Evaluation methodology—feeds into router training)

6. **Can genomic signal NCAs be dynamically expanded (add new genomic bits) without retraining from scratch?** (Continual learning for model zoos)

7. **How do NCA model zoos compare to diffusion model LoRA collections for texture synthesis?** (Direct benchmark vs. alternative approach)

8. **Can model zoos with learned routers achieve compositional generalization (combining multiple specialists) for hybrid textures?** (Emergent capabilities of ensembles)

---

## Sources

### Neural Cellular Automata - Core Research
- [Learning spatio-temporal patterns with Neural Cellular Automata - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11078362/)
- [Neural Cellular Automata (neuralca.org)](https://www.neuralca.org/)
- [Neural Cellular Automata research overview](https://www.emergentmind.com/topics/neural-cellular-automata-nca)
- [Neural cellular automata: applications to biology and beyond classical AI](https://arxiv.org/abs/2509.11131)
- [Neural Cellular Automata: From Cells to Pixels](https://arxiv.org/abs/2506.22899)

### Mixture of Neural Cellular Automata
- [Mixtures of Neural Cellular Automata: A Stochastic Framework for Growth Modelling and Self-Organization](https://arxiv.org/html/2506.20486)
- [Neural cellular automata: Applications to biology and beyond classical AI - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1571064525001757?via=ihub)

### Universal NCAs and Multi-Task Learning
- [A Path to Universal Neural Cellular Automata](https://arxiv.org/html/2505.13058v1)
- [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/)
- [Growing Reservoirs with Developmental Graph Cellular Automata](https://arxiv.org/html/2508.08091)

### Genomic Signals and Specialization
- [Neural Cellular Automata Can Respond to Signals](https://www.researchgate.net/publication/370949760_Neural_Cellular_Automata_Can_Respond_to_Signals)
- [Multi-texture synthesis through signal responsive neural cellular automata](https://www.nature.com/articles/s41598-025-23997-7)
- [Multi-Texture Synthesis through Signal Responsive Neural Cellular Automata](https://arxiv.org/html/2407.05991v2)
- [Neural Cellular Automata Can Respond to Signals | MIT Press](https://direct.mit.edu/isal/proceedings/isal2023/35/5/116835)

### Mixture of Experts - Architecture and Routing
- [Mixture of Experts Explained | Hugging Face](https://huggingface.co/blog/moe)
- [What Is Mixture of Experts (MoE)? | DataCamp](https://www.datacamp.com/blog/mixture-of-experts-moe)
- [What Is Mixture of Experts (MoE) and How It Works? | NVIDIA](https://www.nvidia.com/en-us/glossary/mixture-of-experts/)
- [Mixture of experts - Wikipedia](https://en.wikipedia.org/wiki/Mixture_of_experts)
- [Mixture-of-Experts with Expert Choice Routing | Google Research](https://research.google/blog/mixture-of-experts-with-expert-choice-routing/)
- [Mixture of Experts LLMs: Key Concepts Explained](https://neptune.ai/blog/mixture-of-experts-llms)
- [What is mixture of experts? | IBM](https://www.ibm.com/think/topics/mixture-of-experts)
- [Understanding Mixture of Experts (MoE) Neural Networks](https://intuitionlabs.ai/articles/mixture-of-experts-moe-models)
- [The Rise of MoE: Comparing 2025's Leading Mixture-of-Experts AI Models](https://friendli.ai/blog/moe-models-comparison)

### RouteLLM and Learned Routing
- [RouteLLM: Learning to Route LLMs with Preference Data](https://arxiv.org/abs/2406.18665)
- [GitHub - lm-sys/RouteLLM](https://github.com/lm-sys/RouteLLM)
- [RouteLLM: An Open-Source Framework for Cost-Effective LLM Routing | LMSYS](https://lmsys.org/blog/2024-07-01-routellm/)
- [RouteLLM – UC Berkeley Sky Computing Lab](https://sky.cs.berkeley.edu/project/routellm/)

### Model Zoos and Collections
- [Model Zoo - Deep learning code and pretrained models](https://www.modelzoo.co/)
- [A Model Zoo on Phase Transitions in Neural Networks](https://arxiv.org/html/2504.18072)
- [Model Zoos for Benchmarking Phase Transitions in Neural Networks | OpenReview](https://openreview.net/forum?id=JlkqReTftJ)
- [GitHub - onnx/models: ONNX Model Zoo](https://github.com/onnx/models)

### Foundation Models vs. Specialized Ensembles
- [Foundation model - Wikipedia](https://en.wikipedia.org/wiki/Foundation_model)
- [How Foundational are Foundation Models for Time Series Forecasting?](https://arxiv.org/html/2510.00742v3)
- [Foundation Models vs. Traditional AI: What's the Difference?](https://medium.com/@hakeemsyd/foundation-models-vs-traditional-ai-whats-the-difference-ad9f3f097dec)
- [What Are Foundation Models? | IBM](https://www.ibm.com/think/topics/foundation-models)
- [Beyond the basics: A comprehensive foundation model selection framework | AWS](https://aws.amazon.com/blogs/machine-learning/beyond-the-basics-a-comprehensive-foundation-model-selection-framework-for-generative-ai/)
- [Leveraging a foundation model zoo for cell similarity search](https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2025.1480384/full)
- [Impact of Model Ensemble Techniques on Foundation Model Performance](https://www.algomox.com/resources/blog/model_ensemble_techniques_in_fmops/)
- [Benchmarking foundation models as feature extractors for weakly supervised computational pathology](https://www.nature.com/articles/s41551-025-01516-3)

### Neural Architecture Search and Meta-Learning
- [Advances in neural architecture search - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11389615/)
- [Neural Architecture Search (NAS): Automating Model Design](https://blog.roboflow.com/neural-architecture-search/)
- [Systematic review on neural architecture search](https://link.springer.com/article/10.1007/s10462-024-11058-w)
- [Advances in neural architecture search | National Science Review](https://academic.oup.com/nsr/article/11/8/nwae282/7740455)
- [Introduction to Meta Learning and Neural Architecture Search](https://www.thinkautonomous.ai/blog/meta-learning/)

---

**Conclusion**: NCA model zoos with learned routing represent a viable path toward versatile, efficient pattern generation systems. While true foundation NCAs remain elusive, combining specialized NCAs (genomic signals, MNCA mixtures, attention-based universal models) with RouteLLM-style learned routing can achieve broad capabilities with orders-of-magnitude lower compute than diffusion models. Critical open questions center on optimal routing granularity, load balancing in spatial domains, and whether local computation paradigms can achieve global reasoning. The extreme parameter efficiency of NCAs (68-8000 params) fundamentally changes the economics of model zoos—training and serving hundreds of specialists becomes trivial, suggesting specialized collections may be more appropriate than pursuing single foundation models for this architecture class.
