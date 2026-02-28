# Learned Prompt-to-Genome Mapping for Neural Cellular Automata

**Research Topic:** Can networks map CLIP text embeddings to genomic signals, combining zero-shot capability with stability?

**Research Date:** February 19, 2026

**Research ID:** rq-1770914400001-prompt-to-genome

---

## Summary

Learned prompt-to-genome mapping represents an **emerging but unexplored frontier** in NCA research that could bridge CLIP's zero-shot generalization with genomic signals' stability. The challenge is extreme dimensionality reduction (512D → 3-8 bits, a **64-170× compression**) while preserving semantic information. Three architectural approaches exist: **learned adapters** (autoencoder-based projection), **hardware conditioning** (Universal NCA's attention-based routing), and **hybrid systems** (AdaNCA-style plug-and-play modules). No implementations exist yet, but recent advances in binary quantization (2024-2025) and sparse autoencoders (2025-2026) demonstrate feasibility. Critical challenges include gradient flow through long NCA iterations, semantic preservation during extreme compression, and training data requirements. The approach offers potential for **zero-shot multi-texture synthesis** with sub-20k parameters if successfully implemented.

---

## Key Findings

### 1. The Dimensionality Challenge: 512D → 3-8 Bits

**CLIP Text Embeddings:** CLIP models (ViT-B/32) produce **512-dimensional continuous embeddings** that encode rich semantic information about text prompts. These embeddings capture nuanced relationships between concepts through their position in high-dimensional space.

**Genomic Signals in NCAs:** Current state-of-the-art multi-texture NCAs use **binary genomic encoding** with 3-8 bits to specify texture identity. For example:
- 3 bits = 8 textures (2³)
- 4 bits = 16 textures (2⁴)
- 8 bits = 256 textures (2⁸)

The **compression ratio** is staggering: 512 × 32 bits (float32) = **16,384 bits → 3-8 bits** represents a **2,048× to 546× compression** just in bit count, or **64× to 170× compression** if we consider going from 512 continuous dimensions to 3-8 discrete bits.

### 2. Binary Quantization: Lessons from Embedding Compression

Recent 2024-2025 research on CLIP embedding quantization demonstrates that extreme compression is possible while maintaining semantic relationships:

**Binary Quantization (1-bit per dimension):**
- Converts float32 values to 1-bit: `if value > 0 then 1 else 0`
- Achieves **32× memory reduction** (512 float32 → 512 bits)
- Performance: 1024D models retain **>96% performance** after 32× reduction
- Speed: Hamming distance on 512-bit vectors is **~20× faster** than float operations

**Multi-bit Quantization:**
- 2-bit quantization: **16× compression**, intermediate accuracy
- 8-bit quantization: **4× compression**, minimal quality loss
- Matryoshka learning enables **nested embeddings** where first k dimensions form valid k-dimensional embedding

**Critical Insight:** The 512D → 512 bits (1 bit per dimension) reduction suggests that **512D → 8 bits** (64× further compression to 8 binary channels) is an **extreme but not impossible** target, though it requires sophisticated learned mappings rather than simple thresholding.

### 3. Sparse Autoencoders for Semantic Preservation

**Recent Advances (2025-2026):**

Sparse autoencoders have emerged as a powerful tool for mapping high-dimensional embeddings to low-dimensional representations while preserving semantic structure:

**Regularized Autoencoders (RAE):**
- Achieve **superior k-NN preservation** through norm distortion regularization
- Provide **provable guarantees** for neighborhood structure preservation
- Critical for maintaining semantic similarity in compressed space

**Sparse Linear Concept Embeddings (SpLiCE):**
- Decomposes CLIP representations into **sparse linear combinations** of interpretable concepts
- Enables **adjustable interpretability-accuracy tradeoff**
- Demonstrates that CLIP's latent space is highly structured

**Hierarchical Sparse Autoencoders:**
- Applied to CLIP's representation space to decompose complex distributed representations
- Trade-off: sparsity vs reconstruction fidelity
- Recent work (Feb 2025) explores this specifically for CLIP embeddings

**Key Challenge:** Existing sparse autoencoders typically reduce to **50-256 dimensions**, not the **3-8 bits** required for genomic signals. The gap remains substantial.

### 4. Three Architectural Paradigms for Prompt-to-Genome Mapping

#### A. Learned Adapter Networks

**Architecture:** Train a small neural network to project CLIP embeddings into genomic space:
```
CLIP(text) → 512D embedding → Adapter Network → 3-8 bit genome → NCA
```

**Training Strategy:**
1. Pre-train NCA with genomic signals on multiple textures (known ground truth)
2. Freeze NCA weights
3. Train adapter to map CLIP embeddings → genomes that produce desired textures
4. Loss: VGG perceptual loss + CLIP similarity between generated texture and target

**Advantages:**
- Modular design, NCA remains unchanged
- Adapter can be small (few thousand parameters)
- Clear separation of concerns

**Challenges:**
- Requires differentiating through NCA iterations (gradient flow challenge)
- Training data: need (text prompt, target texture) pairs
- Risk of mode collapse to limited genome subspace

#### B. Hardware Conditioning (Universal NCA Approach)

**Architecture:** Based on Gabriel Béna's 2025 "Universal NCA" work, use attention-based conditioning:

```
Each cell receives:
- Perception vector P (spatial neighborhood information)
- Hardware vector I (immutable conditioning information, derived from CLIP)

Update function: attention mechanism conditions P on I
```

**Key Innovation:** The hardware vector acts as a **fixed scaffold** that guides computation. CLIP embeddings are projected to this hardware space (could be 16-32D instead of 3-8 bits, more feasible).

**Advantages:**
- More expressive than binary genomic signals
- Attention mechanism naturally handles variable-length conditioning
- Proven to achieve "universal computation" in continuous NCAs

**Challenges:**
- Requires NCA architecture redesign
- Still needs learned projection CLIP → hardware space
- Computational overhead from attention mechanism

#### C. AdaNCA-Style Plug-and-Play Adaptors

**Architecture:** Inspired by AdaNCA (NeurIPS 2024), insert small NCA modules as adaptors:

```
CLIP(text) → Adapter NCA → genomic signal → Primary NCA → texture
```

**Key Findings from AdaNCA:**
- <3% parameter increase yields 10% accuracy improvement on adversarial tasks
- Dynamic Interaction mechanism reduces computational overhead
- Plug-and-play design works with frozen base models

**Adaptation for Prompt-to-Genome:**
- Adapter NCA takes CLIP embedding as additional channels
- Evolves for small number of steps to "distill" CLIP information
- Final state mapped to genomic signal via learned linear projection

**Advantages:**
- Leverages proven adaptor design
- Small parameter overhead
- Can be trained independently of main NCA

**Challenges:**
- Two-stage computation may be slower
- Still requires extreme dimensionality reduction at final projection

### 5. Training Data Requirements and Stability

**Genomic NCA Training:**
Current multi-texture NCAs with genomic signals require:
- **1,500-10,000 parameters** for 8 textures
- **100-500 training images** per texture class
- **Stable training** through VGG perceptual loss

**CLIP-Conditioned Systems:**
From existing CLIP-guided NCA work (though not genomic):
- **Gradient noise** from CLIP's complex loss landscape
- **Dead cell problem** where cells stop updating
- **Long-horizon gradient flow** through 100-1000 NCA iterations

**Proposed Training Strategy for Prompt-to-Genome:**

1. **Phase 1: Supervised Pre-training**
   - Dataset: Pairs of (text descriptions, texture images)
   - Train adapter to map CLIP(text) → genome such that NCA produces texture
   - Size: ~10,000 texture-text pairs across diverse categories

2. **Phase 2: Self-Supervised Fine-tuning**
   - Generate synthetic textures with known genomes
   - Create text descriptions via CLIP interrogation or GPT-4V
   - Train adapter on (synthetic text, known genome) pairs
   - Advantage: unlimited training data

3. **Phase 3: Zero-Shot Evaluation**
   - Test on unseen text prompts
   - Measure: texture quality (FID, LPIPS), text-image alignment (CLIP score)

**Stability Techniques:**
- Gradient clipping (essential for long NCA iterations)
- Curriculum learning: start with short iteration counts, gradually increase
- Regularization on genome space to prevent collapse
- Possibly freeze NCA weights during adapter training

### 6. Comparison with Existing Approaches

| Approach | Dimensionality | Training | Zero-Shot | Stability | Status |
|----------|---------------|----------|-----------|-----------|---------|
| **Binary Genomic Signals** | 3-8 bits | Low | ❌ No | ✅ High | ✅ Implemented |
| **CLIP Direct Loss** | 512D (implicit) | Medium | ✅ Yes | ⚠️ Noisy gradients | ⚠️ Experimental |
| **Latent Space NCAs** | 16-64D continuous | Medium | ❌ No | ✅ High | ✅ Implemented |
| **Hardware Conditioning (Universal NCA)** | 16-32D continuous | High | ⚠️ Partial | ✅ High | ✅ Implemented |
| **Prompt-to-Genome Adapters** | 3-8 bits | **High** | **✅ Yes** | **❓ Unknown** | **❌ Not Implemented** |

**Key Insight:** Prompt-to-genome mapping would be the **first approach** to combine:
1. Ultra-compact genomic representation (3-8 bits)
2. Zero-shot generalization from text prompts
3. Stable training (inherited from pre-trained NCA)

### 7. Feasibility Assessment

**Technical Feasibility: ⚠️ HIGH RISK, HIGH REWARD**

**Proven Components:**
✅ Binary quantization of 512D CLIP embeddings (96% performance retention)
✅ Multi-texture NCAs with genomic signals (1,500-10,000 params, 8+ textures)
✅ AdaNCA-style adaptor architectures (<3% overhead)
✅ Sparse autoencoders for semantic preservation (RAE, SpLiCE)

**Missing Links:**
❌ No demonstrated 512D → 3-8 bit projection maintaining semantic relationships
❌ No CLIP-conditioned genomic NCA implementations
❌ Unknown gradient flow behavior through adapter + NCA pipeline
❌ Unclear training data requirements for zero-shot generalization

**Risk Factors:**

1. **Extreme Compression:** 64-170× compression may lose too much semantic information
   - Mitigation: Use 16-32D hardware vectors instead of 3-8 bits (Universal NCA approach)

2. **Gradient Flow:** Backprop through 100-1000 NCA iterations is challenging
   - Mitigation: Freeze NCA weights, train only adapter

3. **Mode Collapse:** Adapter may learn to map all prompts to small genome subset
   - Mitigation: Diversity regularization, contrastive loss on genome space

4. **Training Data:** May need 10,000+ (text, texture, genome) triplets
   - Mitigation: Self-supervised generation of synthetic training data

**Estimated Implementation Effort:**
- Small-scale proof-of-concept: **2-3 months** (single researcher)
- Full system with benchmark: **6-12 months** (research team)
- Production-ready: **12-18 months**

### 8. Potential Impact and Applications

**If Successfully Implemented:**

**Zero-Shot Multi-Texture Synthesis:**
- User types "zebra stripes" → instant texture generation
- No per-texture training required
- Semantic interpolation: "zebra stripes" + "tiger fur" blends

**Extreme Parameter Efficiency:**
- Current: 8 textures = 1,500-10,000 NCA params + 8 fixed genomes
- With adapters: 8 textures = 1,500-10,000 NCA params + 5,000-20,000 adapter params
- Scaling: 1000+ textures with same adapter (amortized cost)

**Real-Time Creativity Tools:**
- Game developers: text-to-texture in engine
- Artists: semantic texture exploration
- VR/AR: dynamic environment generation from descriptions

**Research Contributions:**
- First bridge between CLIP zero-shot and genomic stability
- New benchmark for extreme dimensionality reduction
- Framework for injecting semantic control into cellular automata

---

## Deep Dive

### The Core Technical Challenge: Information Bottleneck

The fundamental question is: **How much semantic information can 3-8 bits encode?**

**Information Theory Perspective:**

CLIP embeddings live in a **512-dimensional continuous space** with effectively infinite entropy. Genomic signals are **3-8 discrete bits** with entropy:
- 3 bits = log₂(8) = **3 bits of information**
- 8 bits = log₂(256) = **8 bits of information**

This creates a **massive information bottleneck**. The entire universe of CLIP's semantic understanding must compress into one of 8-256 discrete states.

**But wait—is this actually the constraint?**

**Insight 1: Genomic Signals are NOT the Final Representation**

The genome is an **input** to the NCA, which then iterates 100-1000 times to produce the texture. The final texture is effectively a **much higher-dimensional representation** (e.g., 256×256×3 = 196,608 values).

**The genome + NCA dynamics = decoder in a VAE-like system:**
```
CLIP embedding (512D) → Genome (3-8 bits) → NCA iterations → Texture (196,608D)
```

This is analogous to:
```
Data → Compressed latent code → Decoder → Reconstruction
```

**Insight 2: The Genome is a "Seed" not a "Representation"**

In biology, a genome doesn't encode every detail of an organism—it encodes a **developmental program** that unfolds through complex dynamics. Similarly, an NCA genome is:
- A **developmental instruction**
- A **parameter to a dynamic system**
- A **discrete switch** that selects between learned behaviors

**Analogy:** Like selecting between piano pieces with a 3-bit code. The 3 bits don't "contain" the music—they select which program to run.

### Can 8 Genomic States Cover "Semantic Texture Space"?

**Pessimistic View: No**

CLIP's semantic space is continuous and infinitely divisible. Text prompts span:
- Materials: "wood", "metal", "stone", "fabric"...
- Patterns: "stripes", "spots", "waves", "fractals"...
- Styles: "realistic", "abstract", "impressionist"...
- Combinations: "rusty metal with moss growing"...

No 8-state or even 256-state discrete system can cover this space.

**Optimistic View: Maybe**

But do we need to cover **all** of semantic space? Consider:
- **Discrete clustering:** Maybe texture space naturally clusters into ~100-1000 prototypical patterns
- **Interpolation:** CLIP embeddings could map to **continuous genomic representations** (not discrete bits) which then get quantized
- **Hierarchical encoding:** First 3 bits = coarse category, NCA learns to interpret remaining CLIP dimensions as fine-grained modulation

**Hybrid Approach: Continuous Hardware Vectors**

This is where **Universal NCA's hardware conditioning** becomes compelling:

Instead of:
```
CLIP (512D) → Adapter → Discrete genome (3-8 bits) → NCA
```

Do:
```
CLIP (512D) → Adapter → Continuous hardware vector (16-32D) → Hardware-conditioned NCA
```

**Advantages:**
- More expressive: 16-32 continuous dimensions >> 3-8 discrete bits
- Still compact: 16-32D is 16× smaller than CLIP's 512D
- Gradient-friendly: continuous optimization throughout

**Trade-off:** Requires modifying NCA architecture to accept hardware conditioning.

### Gradient Flow: The Long Iteration Problem

**Challenge:** NCAs iterate 100-1000 times. Gradients must flow backward through this entire computational graph.

**Standard NCA Training:**
- Loss is computed on final output texture
- Gradients backpropagate through all iterations
- Memory: O(iterations × state_size)
- Gradient magnitude decay: exponential in iteration count

**CLIP-Conditioned Challenge:**
Adding an adapter in front makes it worse:
```
CLIP text → Adapter → Genome → NCA (1000 iters) → Texture → Loss
```

Gradients must flow through:
1. VGG perceptual loss (10+ layers)
2. NCA iterations (1000 steps of convolution + MLP)
3. Adapter network (3-5 layers)
4. CLIP encoder (12-24 transformer layers)

This is an **extremely long computational graph**.

**Solution 1: Freeze the NCA**

Most promising approach:
1. Pre-train NCA with fixed genomic signals on multiple textures
2. **Freeze all NCA weights**
3. Only train the adapter: CLIP → genome
4. Loss: Final texture quality + CLIP alignment

**Advantages:**
- No gradients through NCA iterations
- Much faster training
- Leverages stable pre-trained NCA

**Challenge:** How to get gradients w.r.t. genome if it's discrete?

**Solution 1A: Straight-Through Estimator**

During forward pass: Quantize to discrete genome
During backward pass: Treat as continuous (gradient passes through unchanged)

This is standard in binary neural networks and has been shown to work well.

**Solution 1B: Gumbel-Softmax**

Use Gumbel-Softmax trick for differentiable sampling from discrete distribution:
- Adapter outputs logits over genome space
- Sample with Gumbel noise + temperature annealing
- As temperature → 0, becomes discrete
- Gradients flow through softmax operation

**Solution 2: Multi-Resolution Supervision**

Add losses at intermediate NCA iterations:
```
Iteration 100: Coarse texture loss
Iteration 300: Medium detail loss
Iteration 1000: Final quality loss
```

This provides "shortcut" gradients and stabilizes training.

**Solution 3: Detached CLIP Loss**

Only compute CLIP alignment loss on final texture, not on adapter:
```
genome = adapter(clip_embedding).detach()
texture = nca(genome)
loss = vgg_loss(texture, target) + clip_loss(clip_embedding, texture)
```

The `.detach()` prevents CLIP gradients from flowing to adapter.

### Training Data: The Synthetic Generation Strategy

**Problem:** Need large datasets of (text prompt, texture image, optimal genome) triplets.

**Solution: Self-Supervised Generation**

1. **Generate Textures from Known Genomes**
   - For each of the 8-256 trained genomes
   - Run NCA to produce texture
   - Introduce variation: different random seeds, perturbations

2. **Generate Text Descriptions**
   - Use CLIP Interrogator or GPT-4V to caption textures
   - Example: Genome 0b101 → texture → "organic cellular pattern with dark boundaries"

3. **Create Training Triplets**
   - (Generated text, generated texture, known genome)
   - Advantage: **unlimited training data**
   - Cost: Synthetic distribution may not match real prompts

4. **Fine-Tune on Real Data**
   - Collect smaller set of human-written (prompt, texture) pairs
   - Fine-tune adapter on real distribution

**Data Augmentation:**
- Paraphrase prompts with LLM
- Vary NCA iteration count
- Perturb initial states for diversity

### Architectural Design Choice: Where to Inject CLIP?

**Option A: CLIP → Genome (Proposed Above)**
```
CLIP embedding → Adapter → Genome → NCA
```
- Most compact
- Hardest to train (discrete bottleneck)

**Option B: CLIP → Per-Cell Conditioning**
```
CLIP embedding → Broadcast to all cells → NCA update uses CLIP features
```
- More expressive
- Higher memory cost
- Used in some CLIP-guided NCA work

**Option C: CLIP → Genome + Per-Cell**
```
CLIP embedding → {Genome, Per-cell features} → NCA
```
- Hybrid approach
- Genome provides coarse identity
- Per-cell features provide fine-grained control

**Option D: Hierarchical CLIP Injection**
```
CLIP embedding → Multi-scale features → Inject at different NCA layers
```
- Inspired by U-Net skip connections
- Coarse CLIP features at early iterations
- Fine CLIP features at late iterations

**Recommendation:** Start with Option A (simplest, most challenging). If it fails due to information bottleneck, move to Option C or D.

---

## Connections to Existing Knowledge

### 1. Relation to Diffusion Models

Diffusion models also map text → images via CLIP embeddings, but with fundamentally different architecture:

**Diffusion:**
- CLIP embedding conditions UNet at each denoising step
- Typically 50-1000 denoising steps
- 500M-8B parameters
- No explicit "genome" or compressed latent

**Prompt-to-Genome NCA:**
- CLIP embedding → discrete genome → deterministic NCA evolution
- 100-1000 NCA iterations (similar iteration count)
- 5,000-30,000 total parameters (100,000× smaller)
- Explicit compressed "genome" representation

**Key Difference:** NCAs have a **discrete bottleneck** (genome), while diffusion models maintain high-dimensional representations throughout.

**Learning Opportunity:** Diffusion's success shows that CLIP → visual generation is viable. NCAs' challenge is doing it through extreme compression.

### 2. Relation to Genomic Signal NCAs

Current genomic signal NCAs use **fixed, manually-assigned** genomes:
- Texture 0 → genome 0b000
- Texture 1 → genome 0b001
- ...
- Texture 7 → genome 0b111

**Proposed prompt-to-genome adds:**
- **Learned mapping** from semantics to genomes
- **Zero-shot generalization** to new prompts
- **Continuous interpolation** in prompt space → genome space

**Relationship:** Prompt-to-genome is a **generalization** of fixed genomic signals. If adapter learns to map similar prompts to the same genome, it recovers fixed assignment.

### 3. Relation to Universal NCA (Hardware Conditioning)

Gabriel Béna's Universal NCA (2025) uses **continuous hardware vectors** to condition each cell's computation. Key insights:

**Architectural Similarity:**
- Universal NCA: Hardware vector I → conditions NCA update
- Prompt-to-genome: CLIP → genome → conditions NCA rules

**Key Difference:**
- Universal NCA: 16-32D continuous hardware vectors
- Genomic signals: 3-8 bits discrete

**Potential Integration:**
Could combine both approaches:
```
CLIP embedding → Adapter → {Discrete genome, Continuous hardware vector}
```

Where:
- Genome selects coarse behavior (8-256 modes)
- Hardware vector provides fine-grained semantic control

This would be a **hybrid architecture** leveraging both paradigms.

### 4. Relation to AdaNCA (Vision Transformer Adaptors)

AdaNCA (NeurIPS 2024) inserts small NCA modules into ViTs:

**Lessons for Prompt-to-Genome:**

1. **Plug-and-play design works:** <3% parameters, 10% improvement
2. **Frozen base models are okay:** Can train adaptor independently
3. **Dynamic mechanisms reduce overhead:** Don't need full NCA at every step

**Potential Application:**
```
Frozen pre-trained CLIP encoder
    ↓
AdaNCA-style adaptor (small NCA)
    ↓
Genome projection layer
    ↓
Frozen pre-trained texture NCA
    ↓
Output texture
```

This would train **only** the small adaptor NCA + projection layer.

### 5. Relation to Latent Space NCAs

Latent NCAs operate in compressed 16-64D space instead of RGB:

**Comparison:**

| Approach | Dimensionality | Continuous/Discrete | Semantic Control |
|----------|---------------|---------------------|------------------|
| Latent NCA | 16-64D | Continuous | ❌ No |
| Genomic Signal | 3-8 bits | Discrete | ✅ Yes (fixed mapping) |
| Prompt-to-Genome | 3-8 bits | Discrete | ✅ Yes (learned mapping) |

**Could combine approaches:**
```
CLIP → Adapter → Latent space (16-64D) → Latent NCA → RGB
```

With genomic signals injected into the latent NCA as additional channels. This would be a **hybrid latent-genomic architecture**.

---

## Follow-Up Questions

### Research Questions Answered:

**Q: Can networks map CLIP embeddings to genomic signals?**
A: **Technically feasible but unproven.** The 512D → 3-8 bit compression is extreme, but recent binary quantization research suggests 32-64× compression is possible with high semantic preservation. The missing link is a working implementation.

**Q: Can this combine zero-shot capability with stability?**
A: **Potentially yes.** By freezing pre-trained NCAs and training only the adapter, stability of genomic NCAs is preserved while adding zero-shot via CLIP. The architecture cleanly separates these concerns.

### New Research Questions Generated:

1. **Optimal Architecture:** Should adapters map CLIP → discrete genomes, or CLIP → continuous hardware vectors, or hybrid?

2. **Training Data Scale:** What's the minimum dataset size for zero-shot generalization? 1K pairs? 10K? 100K?

3. **Semantic Interpolation:** If CLIP embedding A → genome 0b101 and embedding B → genome 0b110, does embedding (A+B)/2 → genome 0b101.5? (requires continuous representation)

4. **Multi-Modal Conditioning:** Can the same adapter map both text AND image CLIP embeddings to genomes? (Enables texture transfer)

5. **Genome Space Structure:** Do learned genome assignments form semantic clusters? (E.g., all "organic" textures map to genomes 0b0XX)

6. **Failure Modes:** Under what conditions does the adapter collapse to outputting the same genome for all prompts?

7. **Scaling Laws:** How does zero-shot performance scale with:
   - Adapter size (parameters)
   - NCA base model capacity
   - Number of pre-trained genomic categories
   - Training dataset size

8. **Real-Time Performance:** Can adapter inference run at interactive rates? (target: <10ms for CLIP → genome)

---

## Sources

### Multi-Texture Synthesis & Genomic Signals
- [Multi-texture synthesis through signal responsive neural cellular automata | Scientific Reports](https://www.nature.com/articles/s41598-025-23997-7)
- [Multi-Texture Synthesis through Signal Responsive Neural Cellular Automata](https://arxiv.org/html/2407.05991)
- [Neural Cellular Automata Can Respond to Signals | Artificial Life Conference Proceedings](https://direct.mit.edu/isal/proceedings/isal2023/35/5/116835)

### CLIP + Neural Cellular Automata
- [GitHub - Mainakdeb/text-2-cellular-automata: Neural Cellular Automata + CLIP](https://github.com/Mainakdeb/text-2-cellular-automata)
- [Mesh Neural Cellular Automata | ACM Transactions on Graphics](https://dl.acm.org/doi/10.1145/3658127)
- [Mesh Neural Cellular Automata](https://meshnca.github.io/)

### Universal NCA & Hardware Conditioning
- [A Path to Universal Neural Cellular Automata | Gabriel Béna](https://gabrielbena.github.io/blog/2025/bena2025unca/)
- [A Path to Universal Neural Cellular Automata | Proceedings of GECCO](https://dl.acm.org/doi/10.1145/3712255.3734310)

### AdaNCA (Adaptor Architecture)
- [AdaNCA: Neural Cellular Automata As Adaptors For More Robust Vision Transformer](https://arxiv.org/abs/2406.08298)
- [AdaNCA: Neural Cellular Automata as Adaptors for More Robust Vision Transformer](https://arxiv.org/html/2406.08298v5)
- [NeurIPS 2024 Poster: AdaNCA](https://neurips.cc/virtual/2024/poster/96193)

### Binary Quantization & Dimensionality Reduction
- [Binary and Scalar Embedding Quantization for Significantly Faster & Cheaper Retrieval](https://huggingface.co/blog/embedding-quantization)
- [Matryoshka 🤝 Binary vectors: Slash vector search costs with Vespa](https://blog.vespa.ai/combining-matryoshka-with-binary-quantization-using-embedder/)
- [Embedding Quantization — Sentence Transformers documentation](https://www.sbert.net/examples/applications/embedding-quantization/README.html)
- [Learn to Binarize CLIP for Multimodal Retrieval and Ranking](https://www.marqo.ai/blog/learn-to-binarize-clip-for-multimodal-retrieval-and-ranking)

### Sparse Autoencoders
- [RAE: A Neural Network Dimensionality Reduction Method for Nearest Neighbors Preservation](https://arxiv.org/html/2509.25839v1)
- [Interpreting CLIP with Hierarchical Sparse Autoencoders](https://arxiv.org/html/2502.20578v1)
- [A Survey on Sparse Autoencoders: Interpreting the Internal Mechanisms of Large Language Models](https://arxiv.org/html/2503.05613v1)
- [Sparse Autoencoders, Again?](https://arxiv.org/html/2506.04859)

### CLIP Embedding Structure
- [Quantifying Structure in CLIP Embeddings: A Statistical Framework for Concept Interpretation](https://arxiv.org/html/2506.13831v1)
- [Interpreting CLIP with Sparse Linear Concept Embeddings (SpLiCE)](https://arxiv.org/html/2402.10376v1)
- [CLIP Embeddings for AI-Generated Image Detection](https://arxiv.org/html/2505.10664v1)

### Neural Cellular Automata Foundations
- [Neural cellular automata: applications to biology and beyond classical AI](https://arxiv.org/abs/2509.11131)
- [Parameter-efficient diffusion with neural cellular automata | npj Unconventional Computing](https://www.nature.com/articles/s44335-025-00026-4)
- [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/)
- [DyNCA: Real-Time Dynamic Texture Synthesis Using Neural Cellular Automata](https://dynca.github.io/)

### Training Stability
- [Stabilizing LLM Training: Techniques and Insights](https://www.rohan-paul.com/p/stabilizing-llm-training-techniques)
- [Mastering Gradient Clipping: Enhancing Neural Networks for Optimal Training](https://www.lunartech.ai/blog/mastering-gradient-clipping-enhancing-neural-networks-for-optimal-training)

### Genomic Deep Learning (Methodology Transfer)
- [NCAE: data-driven representations using a deep network-coherent DNA methylation autoencoder](https://academic.oup.com/bib/article/24/5/bbad293/7243028)
- [Understanding the LLM-based gene embeddings](https://www.biorxiv.org/content/10.64898/2025.12.19.695582v1.full)

---

## Conclusion

**Learned prompt-to-genome mapping represents a high-risk, high-reward research direction.** The core technical challenge—compressing CLIP's 512-dimensional semantic space into 3-8 discrete bits—is extreme, but not insurmountable. Recent advances in binary quantization, sparse autoencoders, and adaptor architectures provide building blocks.

**Three viable paths forward:**

1. **Discrete Genomic Adapters** (most challenging, most compact)
   - 512D → 3-8 bits via learned projection + straight-through estimators
   - Freeze NCA, train only adapter
   - Risk: information bottleneck

2. **Continuous Hardware Conditioning** (more expressive, requires NCA redesign)
   - 512D → 16-32D continuous vectors
   - Integrate with Universal NCA architecture
   - Lower risk, higher parameter count

3. **Hybrid Approach** (balanced)
   - 512D → {discrete genome (3-8 bits) + continuous features (8-16D)}
   - Best of both worlds: discrete switching + continuous modulation
   - Moderate complexity

**If successful, this would enable:**
- Zero-shot multi-texture synthesis from text
- Extreme parameter efficiency (<20k params for 1000+ textures)
- Real-time creative tools with semantic control
- New benchmark for neural-symbolic integration in NCAs

**Critical next step:** Small-scale proof-of-concept with frozen 8-texture NCA and simple adapter network. If discrete genome bottleneck proves too restrictive, pivot to continuous hardware vectors.

The field is ready for this exploration—all components exist, awaiting integration.
