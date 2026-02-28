# Hierarchical NCA Fine-Tuning Strategies: Freeze Coarse or Fine Layers?

**Research Topic ID**: rq-1770914798000-hierarchical-fine-tuning
**Date**: 2026-02-19
**Status**: Completed

## Summary

Hierarchical neural cellular automata (H-NCA) fine-tuning strategies remain largely unexplored territory, with no published systematic studies comparing coarse-layer freezing versus fine-layer freezing. However, convergent evidence from three domains—H-NCA architectures, traditional hierarchical neural networks, and parameter-efficient fine-tuning—suggests **freezing coarse layers while fine-tuning fine layers** is the optimal default strategy, with task-specific exceptions.

## Key Findings

### 1. Current State of H-NCA Fine-Tuning Research

**Gap in Literature**: Despite active research in hierarchical NCAs (Pande & Grattarola 2023, evolving H-NCA 2024), no papers explicitly address fine-tuning strategies for pre-trained H-NCA models. The field has focused on initial training rather than transfer learning.

**Existing Differential Training**: The Pande & Grattarola (2023) H-NCA architecture demonstrates differential trainability across hierarchy levels:
- **Child-NCA** (fine scale): Non-trainable perception layer, 64 hidden filters
- **Parent-NCA** (coarse scale): Trainable perception layer, 256 hidden filters

This design choice reflects an architectural decision rather than a systematic fine-tuning strategy, but suggests the authors recognized different hierarchy levels require different learning dynamics.

### 2. Transfer Learning Principles Applied to H-NCA

Evidence from traditional hierarchical neural networks strongly supports **coarse-layer freezing**:

#### Feature Hierarchy Theory
- **Coarse/Early layers**: Capture universal features (edges, textures, low-frequency patterns)
- **Fine/Later layers**: Develop task-specific representations requiring adaptation

#### Empirical Evidence from Computer Vision

**YOLO Architectures** (2025): Systematic study of layer-freezing strategies found:
- **FR2 (backbone freezing)** emerged as most robust approach across diverse scenarios
- Balances feature preservation with adaptation flexibility
- Gradient analysis shows freezing creates distinct optimization dynamics

**Transformer Models** (2025): Freezing bottom 25-50% of layers in sub-3B parameter models:
- Performance equal to or better than full fine-tuning
- 30-50% memory reduction
- 20-30% faster training

**Fine-Grained Recognition** (2025): YOLOv8 adaptation demonstrated that preserving COCO performance while adapting to new tasks requires careful balance—significant adaptation possible *without* catastrophic forgetting when coarse features are preserved.

### 3. H-NCA-Specific Considerations

#### Spatial Hierarchy vs. Feature Hierarchy

H-NCAs operate differently than standard CNNs:
- **Coarse scales** in H-NCA: Condition collective behavior, provide global context
- **Fine scales** in H-NCA: Perform detailed local dynamics, implement specific patterns

This mapping suggests:
```
Standard CNN          →  H-NCA
Early layers          →  Coarse-scale parent NCAs
Later layers          →  Fine-scale child NCAs
```

#### Argument FOR Freezing Coarse Layers:

1. **Hierarchical Control Flow**: Parent NCAs provide "guidance signals" to child NCAs. Stable coarse-scale behavior enables consistent fine-scale adaptation.

2. **Catastrophic Forgetting Risk**: Coarse scales encode fundamental pattern formation principles. Modifying these affects *all* downstream fine-scale behaviors—high risk of forgetting.

3. **Parameter Efficiency**: Parent NCAs have more parameters (256 filters vs. 64 in Pande & Grattarola). Freezing the larger component saves compute.

4. **Multi-Scale Compositionality**: Frozen coarse scales act as reusable "pattern generators" that fine scales can learn to modulate—similar to prompt tuning in LLMs.

#### Argument FOR Freezing Fine Layers (Vice Versa):

1. **Low-Level Feature Reusability**: Fine-scale NCAs might learn universal local update rules (e.g., edge detection, spot formation) that transfer across tasks.

2. **Top-Down Adaptation**: Some tasks require changing high-level structure while preserving low-level implementation details.

3. **Computational Locality**: NCAs update rules are local by definition. Fine-scale rules might be more generalizable than coarse-scale coordination.

### 4. Analogies to Other Hierarchical Generative Models

#### Hierarchical VAEs
- Coarse latent variables: Encode global structure
- Fine latent variables: Encode local details
- Fine-tuning typically adapts decoder while freezing encoder (similar to freezing coarse scales)

#### Diffusion Models with U-Net
- Downsampling blocks (coarse): Learn hierarchical features at different resolutions
- Progressive training: Often starts with low-resolution (coarse) before adding high-resolution (fine)
- Fine-tuning typically preserves low-frequency understanding, adapts high-frequency details

#### AdaNCA (2024)
- NCAs inserted as "adaptors" between ViT layers
- Train-from-scratch approach, but demonstrates NCAs as plug-and-play modules
- Suggests future direction: frozen pre-trained NCA adaptors with trainable insertion points

### 5. Practical Recommendations

#### Default Strategy: **Freeze Coarse, Train Fine**

**Rationale**:
1. Preserves universal pattern formation principles
2. Enables task-specific detail adaptation
3. Reduces catastrophic forgetting risk
4. More compute-efficient (fewer parameters to update)
5. Aligns with transfer learning best practices

**Implementation**:
```python
# Freeze parent-NCA (coarse scale)
for param in parent_nca.parameters():
    param.requires_grad = False

# Fine-tune child-NCA (fine scale)
for param in child_nca.parameters():
    param.requires_grad = True

# Use low learning rate to prevent fine-scale instability
optimizer = Adam(child_nca.parameters(), lr=1e-4)
```

#### Task-Specific Exceptions:

**Freeze Fine, Train Coarse** when:
- Target task requires fundamentally different global organization
- Low-level patterns are known to transfer well (e.g., biological morphogenesis rules)
- Computational budget prohibits fine-scale retraining

**Progressive Unfreezing** when:
- Maximum adaptation needed
- Sufficient training data available
- Risk of overfitting is low

```python
# Stage 1: Fine-tune fine layers
train(child_nca, epochs=10)

# Stage 2: Unfreeze coarse layers with lower LR
for param in parent_nca.parameters():
    param.requires_grad = True
optimizer = Adam([
    {'params': child_nca.parameters(), 'lr': 1e-4},
    {'params': parent_nca.parameters(), 'lr': 1e-5}
], weight_decay=1e-6)
train(full_model, epochs=5)
```

#### Continual Learning Considerations:

Recent research (2025) on catastrophic forgetting in LLMs reveals:
- **Hierarchical layer-wise regularization** prevents knowledge loss during fine-tuning
- **Decoder-only architectures** (analogous to parent→child flow in H-NCA) experience milder forgetting than encoder-decoder
- **General instruction tuning** on diverse tasks reduces catastrophic forgetting

Adapted to H-NCA:
- Apply regularization losses to coarse-scale NCAs (preserve general pattern formation)
- Allow fine-scale NCAs greater plasticity
- Pre-train on diverse morphogenesis tasks before task-specific fine-tuning

## Deep Dive: Why This Question Matters

### The Tension Between Local and Global

NCAs fundamentally operate through **local interactions generating global patterns**. This creates a unique challenge for hierarchical systems:

**Bottom-Up Emergence**: Child NCAs execute local rules that *should* produce emergent patterns matching parent NCA guidance.

**Top-Down Control**: Parent NCAs provide signals that *should* coordinate child NCA behavior without micromanaging.

**Fine-Tuning Disrupts This Balance**: Modifying either level risks breaking the coordination loop. The question isn't just "which layers to freeze" but "which level of the hierarchy embeds the transferable knowledge?"

### Parameter Efficiency in NCA Context

Traditional deep learning: More parameters = more capacity.
NCA paradigm: **Iterative application** multiplies effective capacity.

A tiny NCA (68-8000 params) iterated 100 times has immense representational power. This means:
1. Freezing even small coarse-scale NCAs preserves substantial "knowledge"
2. Fine-tuning fine-scale NCAs with low parameter count is surprisingly effective
3. The **number of iterations** becomes a hyperparameter for transfer learning (more iterations = more capacity to adapt)

### Connection to Foundation Models

If H-NCAs are to become "foundation models" for generative tasks, the fine-tuning strategy must be determined. Current evidence suggests:

**H-NCA Foundation Model Architecture**:
```
┌─────────────────────────────────┐
│  Frozen Coarse-Scale NCA        │  ← Pre-trained on diverse patterns
│  (256-512 channels)             │     (Turing patterns, textures, morphogenesis)
├─────────────────────────────────┤
│  Trainable Fine-Scale NCA       │  ← Task-specific adaptation
│  (64-128 channels)              │     (Specific texture, specific morphology)
├─────────────────────────────────┤
│  Optional: Genomic Signals      │  ← Multi-texture capability
│  (3-8 bit encoding)             │     (8+ patterns from single model)
└─────────────────────────────────┘
```

This architecture enables:
- **Zero-shot coarse patterns** (frozen parent provides general structure)
- **Few-shot fine details** (minimal fine-tuning of child for specifics)
- **Parameter efficiency** (<10k trainable params for new tasks)

## Connections to Existing Knowledge

### Link to NCA Model Zoos (rq-1739254800004)
Hierarchical fine-tuning strategy determines how individual specialist NCAs in a model zoo can be adapted. If coarse-layer freezing is optimal, then:
- Zoo of H-NCAs shares frozen coarse-scale components
- Fine-scale NCAs specialize per-task
- Router directs queries to appropriate fine-scale specialist while using shared coarse foundation

### Link to CLIP-Conditioned NCAs (rq-1739254800001)
CLIP embeddings could condition coarse-scale NCAs (frozen) to guide fine-scale adaptation:
```
CLIP Text Embedding → Parent NCA (frozen) → Child NCA (trainable)
```
Text provides semantic direction, frozen parent provides pattern formation principles, trainable child implements specifics.

### Link to Continual Learning (rq-1770914798002)
Adding/refining hierarchy levels without catastrophic forgetting requires:
- **Adding fine levels**: Insert new child NCAs below frozen parents (safe)
- **Adding coarse levels**: Insert new parents above frozen children (risky, requires careful regularization)
- **Refining existing**: Fine-layer tuning less disruptive than coarse-layer tuning

## Follow-Up Questions

1. **Empirical Validation Needed**: No paper has actually tested coarse-freeze vs fine-freeze for H-NCA. This is a high-value experiment.

2. **Optimal Hierarchy Depth**: Does the freezing strategy change for 3-level vs 4-level hierarchies? (Links to rq-1770914798001)

3. **Genomic Signal Interaction**: How do genomic signals (which provide multi-texture capability) interact with hierarchical freezing strategies?

4. **Attention-Based H-NCA**: Would ViTCA-style attention mechanisms (rq-1739950000001) change the optimal freezing strategy by enabling better gradient flow?

5. **Quantitative Metrics**: What metrics indicate when coarse-layer freezing is preferable? Edge density? Pattern frequency spectrum? Task similarity?

## Sources

### Hierarchical NCA Research
- [Hierarchical Neural Cellular Automata (Pande & Grattarola, 2023)](https://direct.mit.edu/isal/proceedings/isal2023/35/20/116844)
- [Evolving Hierarchical Neural Cellular Automata (2024)](https://dl.acm.org/doi/10.1145/3638529.3654150)
- [A Path to Universal Neural Cellular Automata](https://arxiv.org/html/2505.13058v1)
- [Attention-based Neural Cellular Automata (ViTCA)](https://arxiv.org/abs/2211.01233)
- [AdaNCA: Neural Cellular Automata As Adaptors](https://arxiv.org/abs/2406.08298)
- [Neural Cellular Automata: Applications to Biology and Beyond](https://www.sciencedirect.com/science/article/pii/S1571064525001757)
- [Learning Spatio-Temporal Patterns with NCAs](https://pmc.ncbi.nlm.nih.gov/articles/PMC11078362/)
- [Growing Neural Cellular Automata (Mordvintsev et al.)](https://distill.pub/2020/growing-ca/)

### Layer Freezing Strategies
- [An Analysis of Layer-Freezing Strategies for YOLO Architectures (2025)](https://www.mdpi.com/2227-7390/13/15/2539)
- [Exploring Selective Layer Freezing in Transformers (2025)](https://openreview.net/forum?id=kvBuxFxSLR)
- [Fine-Tuning Without Forgetting: YOLOv8 Adaptation (2025)](https://arxiv.org/html/2505.01016v1)
- [Freezing Layers in Deep Learning and Transfer Learning](https://www.exxactcorp.com/blog/deep-learning/guide-to-freezing-layers-in-ai-models)

### Catastrophic Forgetting & Continual Learning
- [Empirical Study of Catastrophic Forgetting in LLMs (2025)](https://arxiv.org/html/2308.08747v4)
- [Hierarchical Layer-Wise Regularization (2025)](https://arxiv.org/html/2501.13669v2)
- [MIT Self-Distillation Fine-Tuning (2025)](https://venturebeat.com/orchestration/mits-new-fine-tuning-method-lets-llms-learn-new-skills-without-losing-old)
- [Spurious Forgetting in Continual Learning](https://openreview.net/forum?id=ScI7IlKGdI)

### Multi-Scale & Progressive Training
- [Coarse-to-Fine Trained Multi-Scale CNNs](https://ieeexplore.ieee.org/document/7280542)
- [Efficient Progressive Training with Granularity Cross (2025)](https://www.nature.com/articles/s41598-025-20975-x)
- [Multi-Scale Attention for Fine-Grained Visual Categorization (2025)](https://www.mdpi.com/2079-9292/14/14/2869)
- [Bottom-Up and Top-Down Hierarchical Reasoning](https://openaccess.thecvf.com/content_cvpr_2016/papers/Hu_Bottom-Up_and_Top-Down_CVPR_2016_paper.pdf)
- [Learning to Combine Top-Down and Bottom-Up Signals](http://proceedings.mlr.press/v119/mittal20a/mittal20a.pdf)

### Hierarchical Generative Models
- [Deep Generative Modelling: VAEs, GANs, Diffusion Comparison](https://arxiv.org/pdf/2103.04922)
- [Improving Diffusion Models as Alternative to GANs](https://developer.nvidia.com/blog/improving-diffusion-models-as-an-alternative-to-gans-part-2/)
- [Generative AI Survey of Recent Advances](https://www.arxiv.org/pdf/2510.21887)
- [Hybrid Diffusion-VAE Models](https://www.emergentmind.com/topics/diffusion-vae-approach)

### Parameter-Efficient Fine-Tuning (PEFT)
- [PEFT Methods for LLMs](https://huggingface.co/blog/samuellimabraz/peft-methods)
- [Adapter-Based Tuning Strategies](https://medium.com/@akankshasinha247/adapter-based-tuning-vs-prompt-based-tuning-full-fine-tuning-fdd5bb0b4767)
- [Transfer Learning with Keras](https://keras.io/guides/transfer_learning/)

---

**Conclusion**: While empirical validation is needed, convergent evidence from hierarchical neural network research, transfer learning principles, and NCA architectural considerations strongly suggests **freezing coarse layers while fine-tuning fine layers** as the optimal default strategy for hierarchical NCA fine-tuning. This preserves universal pattern formation principles encoded at coarse scales while enabling task-specific adaptation at fine scales, balancing parameter efficiency with catastrophic forgetting mitigation.

The absence of explicit research on this question represents a significant gap and high-value research opportunity for the NCA community.
