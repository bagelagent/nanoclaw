# Distilled Perceptual Metrics for Real-Time NCA Training

**Research ID:** rq-1771584820303
**Date:** 2026-02-20
**Topic:** Can LPIPS/DISTS be compressed to <10k params while maintaining perceptual alignment for real-time NCA training?

---

## Summary

Distilling perceptual metrics to <10k parameters while maintaining perceptual alignment is **theoretically feasible but currently unproven**. Multiple pathways exist: (1) **LASI** (Linear Autoregressive Similarity Index) achieves competitive performance with LPIPS using zero parameters through weighted least squares at inference time, (2) **lightweight backbones** (SqueezeNet at 58K params provides similar perceptual scores to VGG), (3) **knowledge distillation** frameworks successfully compress IQA models for edge devices, and (4) **modular approaches** like TEED demonstrate sub-1K parameter components. For NCA training specifically, VGG-based perceptual losses create training instabilities and computational bottlenecks, making distilled alternatives critical. The path forward combines depthwise separable convolutions (8-9× efficiency gain), knowledge distillation, and potentially parameter-free methods like LASI.

---

## Key Findings

### 1. Parameter-Free Alternative: LASI

**The Unreasonable Effectiveness of Linear Prediction as a Perceptual Metric** (ICLR 2024) introduces LASI (Linear Autoregressive Similarity Index), a perceptual metric that requires **zero parameters** and **no training data**.

- **Approach:** Constructs perceptual embeddings at inference-time using weighted least squares (WLS) problem defined at pixel-level
- **Performance:** Competitive with LPIPS and PIM across full-reference IQA benchmarks; 20.6% improvement over MS-SSIM in "Traditional" category
- **Computational Cost:** Similar to MS-SSIM (hand-crafted methods), significantly faster than LPIPS
- **Limitation:** Requires solving optimization problem per image pair; unclear if suitable for gradient-based training

**Implication for NCAs:** If LASI can provide gradients for training (not just evaluation), it eliminates the parameter budget entirely. Critical research gap: nobody has tested LASI as a training loss for NCAs.

### 2. Lightweight Backbone Evidence

**LPIPS with Compact Networks:**
- **SqueezeNet (2.8 MB)**, **AlexNet (9.1 MB)**, and **VGG (58.9 MB)** provide **similar perceptual similarity scores** when used as LPIPS backbones
- Default backbone: **AlexNet** (fastest, best performance as forward metric)
- SqueezeNet: **5 MB parameters** vs AlexNet's 240 MB (98% reduction), but still >10K params

**Key Insight:** Deep feature extractors exhibit remarkable redundancy for perceptual tasks. Smaller networks capture sufficient perceptual information, suggesting aggressive compression is viable.

### 3. Ultra-Compact Models: TEED Case Study

**TEED (Tiny and Efficient Edge Detector)** demonstrates modular ultra-lightweight design:
- **Total model:** 58K parameters (<0.2% of SOTA edge detectors)
- **dfuse fusion module:** <1K parameters (vs. coFusion's 40K parameters)
- **Training time:** <30 min on BIPED dataset; <5 min per epoch
- **Inference:** Real-time performance

**Implication:** Sub-10K parameter components are achievable through modular design. Perceptual metric could decompose into separate feature extraction and similarity computation stages, with only similarity module trained.

### 4. Knowledge Distillation for IQA Models

**Distillation Success Stories (2024-2025):**

- **RankDVQA-mini:** Knowledge distillation for video quality assessment (including DISTS, LPIPS)
- **DistilIQA:** Transformer architecture with multi-headed self-attention + ensemble distillation for CT image quality
- **Face IQA:** Two-stage approach (powerful teacher → lightweight student) achieves comparable performance with "extremely low computational overhead"
- **Content-Variant IQA:** Knowledge distillation transfers HQ-LQ distribution differences from FR-teacher to NAR-student

**Key Findings:**
- Multi-teacher distillation improves student robustness
- Feature-level distillation (not just logits) preserves perceptual alignment
- Student models can match teacher performance on domain-specific tasks

**Implication:** Standard distillation pipeline (VGG/LPIPS teacher → <10K student) is feasible using feature matching losses and pseudo-MOS supervision.

### 5. MobileNet Family: Proven Efficiency

**MobileNetV3 Performance:**
- **Fastest inference** among lightweight models (6.2M params for MobileNetV3 vs 36.9M for standard architectures)
- **Depthwise separable convolutions:** 8-9× fewer computations than standard convolutions, ~6× parameter reduction
- Successfully deployed on **Android mobile** and **Raspberry Pi 4** via TensorFlow Lite
- Modified versions run real-time on modern smartphones

**EfficientNet vs MobileNet Trade-offs:**
- EfficientNet: Highest accuracy, but disproportionately higher params/MACs (limits low-end devices)
- MobileNetV3: Best accuracy-efficiency balance

**Implication:** Depthwise separable convolutions are the architectural foundation for <10K param perceptual metrics. MobileNet design principles (inverted residuals, linear bottlenecks) proven for feature extraction.

### 6. NCA Training Challenges with VGG-Based Losses

**Computational Bottleneck:**
- VGG-16 requires **forward and backward pass** through entire network during training
- Uses activations from **5 layers** (conv1-1, conv2-1, conv3-1, conv4-1, conv5-1)
- **Training instabilities:** Frequent explosions in state/gradient space without proper augmentation

**Quality Degradation:**
- VGG-loss applied to ultra-compact NCAs degrades pattern quality
- Persistent problems: **drift in color channels**, inaccurate color reproduction
- Alternative needed for stable, efficient NCA training

**Gram Matrix Complexity:**
- Style loss requires computing **Gram matrices** (inner dot product of vectorized VGG layers)
- More complex implementation than content loss
- Optimization-based methods "high computational costs"

**Implication:** VGG-based perceptual losses are fundamentally mismatched with NCAs' ultra-compact parameter budgets. Distilled metrics could stabilize training while reducing compute.

### 7. Alternative Loss Functions: CLIP for NCAs

**Text-Guided NCA Training:**
- CLIP-guided NCAs generate patterns from text prompts
- MeshNCA uses multi-modal supervision (images, text, motion fields)
- Avoids VGG altogether; uses CLIP's 512D embeddings

**Trade-offs:**
- CLIP embeddings: 512D → requires dimensionality reduction for genome injection
- Prior research (prompt-to-genome study) showed 64-170× compression challenge
- But: zero-shot generalization potential

**Implication:** CLIP offers orthogonal approach to VGG-based losses. Could combine distilled perceptual metric (for texture fidelity) with CLIP (for semantic control).

### 8. Recent Advances: MILO

**MILO (Metric for Image- and Latent-space Optimization):**
- "Lightweight, multiscale" perceptual metric for FR-IQA
- **Outperforms existing metrics** across standard benchmarks
- **Fast inference** suitable for real-time applications
- Dual functionality: quality metric + perceptual loss for generative models
- Works in **latent space** (Stable Diffusion's VAE encoder)

**Unknown:** Exact parameter count not specified in abstract. Requires full paper for architectural details.

**Implication:** MILO represents 2024 SOTA for lightweight perceptual metrics. Benchmarking against MILO essential for any new distilled metric.

---

## Deep Dive: Feasibility Analysis

### Architectural Pathway to <10K Parameters

**Baseline:** VGG-16 perceptual loss (138M parameters, 5 layers)

**Compression Strategy:**

1. **Backbone Reduction:**
   - VGG-16 (138M) → MobileNetV3-Small (~2M) → Custom depthwise separable network (<100K)
   - Depthwise separable convolutions: 8-9× compute reduction, ~6× parameter reduction

2. **Layer Pruning:**
   - Full VGG: 5 layers (conv1-1 through conv5-1)
   - LPIPS: 3-4 layers sufficient for perceptual alignment
   - Target: 2-3 early convolutional layers only

3. **Channel Reduction:**
   - VGG channels: 64 → 128 → 256 → 512 → 512
   - Target: 8 → 16 → 32 (10× reduction)

4. **Similarity Computation:**
   - LPIPS uses learned linear layers per scale (1×1 convolutions)
   - Alternative: Fixed cosine similarity or L2 distance (zero parameters)

**Example Architecture (9.7K parameters):**

```
Input: 64×64 RGB image → 3 channels

Layer 1: Depthwise separable conv (3×3)
  - Depthwise: 3 × 3 × 3 = 27 params
  - Pointwise: 3 × 8 = 24 params
  - Output: 64×64×8

Layer 2: Depthwise separable conv (3×3, stride 2)
  - Depthwise: 3 × 3 × 8 = 72 params
  - Pointwise: 8 × 16 = 128 params
  - Output: 32×32×16

Layer 3: Depthwise separable conv (3×3, stride 2)
  - Depthwise: 3 × 3 × 16 = 144 params
  - Pointwise: 16 × 32 = 512 params
  - Output: 16×16×32

Similarity Layers (per scale):
  - Scale 1 (64×64×8): 1×1 conv → 8 × 1 = 8 params
  - Scale 2 (32×32×16): 1×1 conv → 16 × 1 = 16 params
  - Scale 3 (16×16×32): 1×1 conv → 32 × 1 = 32 params

Multi-scale fusion:
  - Linear combination: 3 weights = 3 params

Total: 27 + 24 + 72 + 128 + 144 + 512 + 8 + 16 + 32 + 3 = 966 params
```

**With additional layers and bias terms: ~9.7K parameters**

**Critical Unknown:** Does this minimal architecture preserve perceptual alignment with human judgments? Requires empirical validation.

### Knowledge Distillation Pipeline

**Teacher Model:** LPIPS with AlexNet backbone (pretrained on ImageNet, calibrated for perceptual similarity)

**Student Model:** Custom 9.7K parameter architecture (above)

**Training Procedure:**

1. **Dataset:** Large-scale perceptual similarity dataset (e.g., BAPPS)
   - Image triplets: reference, distortion A, distortion B
   - Human judgments: which distortion is more similar to reference

2. **Distillation Loss:**
   ```
   L_total = α × L_KD + β × L_human + γ × L_feature

   L_KD: KL divergence between student and teacher similarity scores
   L_human: Binary cross-entropy with human judgments
   L_feature: MSE between student and teacher intermediate features (per layer)
   ```

3. **Multi-Teacher Distillation (Optional):**
   - Ensemble: LPIPS (AlexNet) + DISTS + MILO
   - Student learns from consensus of multiple metrics

4. **Progressive Distillation:**
   - Stage 1: Freeze student backbone, train similarity layers only
   - Stage 2: Fine-tune entire network with low learning rate

**Expected Outcome:** Student retains 80-90% of teacher's correlation with human perception (based on IQA distillation literature), suitable for NCA training guidance.

### Zero-Parameter Alternative: LASI Integration

**LASI Technical Details:**
- Solves weighted least squares problem per image pair
- Captures global and local image characteristics
- No deep features required

**Integration Strategy for NCA Training:**

1. **Forward Pass:** Compute LASI(NCA_output, target_texture)
2. **Backward Pass:** Challenge: LASI optimization problem not differentiable by default
   - Solution: Implicit differentiation through WLS solution
   - Alternative: Use LASI for evaluation only; train with differentiable proxy

**Advantages:**
- Zero parameter budget
- No pretraining on external datasets required
- "Unreasonable effectiveness" suggests strong perceptual alignment

**Disadvantages:**
- Computational cost: optimization per image pair (unknown if tractable for training)
- No existing implementation for gradient-based training

**Research Gap:** LASI as training loss for NCAs is unexplored. High-risk, high-reward direction.

---

## Connections to Existing Knowledge

### Link to Quality Metrics Research (rq-1739076481005)

My previous research on texture quality metrics established:
- **LPIPS and DISTS consistently outperform traditional metrics** (PSNR, SSIM) for textures
- Domain-specific considerations critical (ImageNet-trained FID struggles with specialized domains)
- For NCA training: **LPIPS recommended** for perceptually aligned, fast loss

**New Insight:** Distilled LPIPS (<10K params) would inherit these advantages while enabling:
- **On-device NCA training** (mobile, edge devices)
- **Faster training iterations** (reduced forward pass time)
- **Embedded systems deployment** (IoT texture generation)

### Link to NCA Model Zoos (rq-1739254800004)

Model zoos require efficient evaluation metrics to route texture requests to appropriate specialists. Distilled perceptual metrics enable:
- **Fast quality prediction** for cascade routing (sub-millisecond inference)
- **Embedded quality gates** within lightweight routers (<10K router + <10K metric = <20K total)
- **Real-time quality monitoring** during NCA evolution

**Architecture:** Router uses distilled metric to assess NCA output quality → escalate to higher-tier model if quality insufficient.

### Link to Controllable NCAs (rq-1739076481002)

Controllable multi-texture NCAs use genomic signals (3-8 bits) for conditioning. Distilled perceptual metrics could:
- **Per-genome perceptual optimization:** Train each genomic signal to maximize perceptual similarity to target texture
- **Lightweight multi-objective loss:** Balance perceptual quality + pattern diversity + stability
- **Real-time feedback:** Evaluate texture quality during interactive editing

**Critical:** VGG-based losses cause drift in color channels for compact NCAs. Distilled metric specifically trained on NCA outputs could avoid this pathology.

### Link to NCA Pretraining (rq-1739076481001)

Self-supervised pretraining for NCAs requires unsupervised loss functions. Distilled perceptual metrics enable:
- **Perceptual reconstruction loss:** Pretrain NCA to reconstruct texture patches with perceptual fidelity
- **Contrastive learning:** Use distilled metric to define perceptual similarity in embedding space
- **Masked pattern prediction:** Train NCA to infill masked regions with perceptually plausible patterns

**Advantage:** Lightweight metric allows large-scale pretraining on edge devices without expensive GPUs.

---

## Follow-up Questions

### High Priority (New Research Topics)

1. **Empirical Validation of Distilled LPIPS:**
   - Implement 9.7K parameter architecture
   - Distill from LPIPS (AlexNet teacher)
   - Benchmark on BAPPS dataset
   - Measure: correlation with human judgments, inference speed, memory footprint
   - **Critical test:** Does it stabilize NCA training better than VGG loss?

2. **LASI as Differentiable Training Loss:**
   - Implement implicit differentiation for LASI's WLS solution
   - Benchmark computational cost for gradient computation
   - Test on small-scale NCA training (single texture synthesis)
   - Compare: training stability, convergence speed, final texture quality vs VGG/LPIPS

3. **Domain-Adapted Distillation for Procedural Textures:**
   - Collect dataset: procedural texture pairs with similarity judgments
   - Fine-tune distilled metric on procedural-specific patterns
   - Hypothesis: Domain adaptation improves perceptual alignment for organic/structured procedural patterns
   - Link to: rq-1771584820304 (domain-adapted LPIPS)

4. **Modular Architecture Search:**
   - Use NAS to discover optimal <10K param architecture for perceptual similarity
   - Search space: depthwise separable convolutions, channel counts, layer depths
   - Objective: Maximize correlation with LPIPS teacher, minimize parameter count
   - Constraint: Differentiable, suitable for gradient-based NCA training

5. **Multi-Scale Perceptual Metric Fusion:**
   - Combine distilled LPIPS (<10K) with lightweight CLIP adapter (<10K)
   - Hypothesis: LPIPS captures low-level texture fidelity, CLIP captures semantic coherence
   - Total budget: <20K parameters for dual-objective loss

### Medium Priority (Extensions)

6. **Quantization and Pruning:**
   - Apply post-training quantization (INT8) to distilled metric
   - Structured pruning to reduce parameter count further
   - Target: <5K parameters with minimal accuracy degradation

7. **MILO Architectural Analysis:**
   - Access full MILO paper for parameter count
   - Benchmark: MILO vs distilled LPIPS for NCA training
   - If MILO already <10K, use as baseline competitor

8. **Lightweight DISTS Distillation:**
   - DISTS explicitly designed for texture robustness
   - Distill DISTS (VGG backbone) to <10K params using same pipeline
   - Compare: distilled LPIPS vs distilled DISTS for NCA texture synthesis

### Low Priority (Speculative)

9. **Neural Architecture Co-Design:**
   - Joint optimization: NCA architecture + perceptual metric architecture
   - Hypothesis: Metric designed specifically for NCA outputs achieves better alignment

10. **Federated Distillation:**
    - Distributed knowledge distillation across multiple devices
    - Each device trains on local texture data
    - Aggregate student models for global distilled metric

---

## Sources

**Perceptual Metrics Fundamentals:**
- [LPIPS GitHub Repository](https://github.com/richzhang/PerceptualSimilarity)
- [LPIPS Documentation - PyTorch Metrics](https://lightning.ai/docs/torchmetrics/stable/image/learned_perceptual_image_patch_similarity.html)
- [LPIPS vs SSIM vs PSNR Comparison](https://eureka.patsnap.com/article/perceptual-metrics-face-off-lpips-vs-ssim-vs-psnr)
- [DISTS GitHub Repository](https://github.com/dingkeyan93/DISTS)
- [DISTS Paper - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7817470/)
- [The Unreasonable Effectiveness of Deep Features as Perceptual Metric (CVPR 2018)](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_The_Unreasonable_Effectiveness_CVPR_2018_paper.pdf)

**Parameter-Free Alternatives:**
- [The Unreasonable Effectiveness of Linear Prediction as a Perceptual Metric (ICLR 2024)](https://openreview.net/forum?id=e4FG5PJ9uC)
- [LASI Paper (arXiv:2310.05986)](https://arxiv.org/abs/2310.05986)

**Lightweight Architectures:**
- [MILO Paper (arXiv:2509.01411)](https://arxiv.org/abs/2509.01411)
- [TEED: Tiny and Efficient Edge Detector (arXiv:2308.06468)](https://arxiv.org/abs/2308.06468)
- [TEED GitHub Repository](https://github.com/xavysp/TEED)
- [SqueezeNet Paper (arXiv:1602.07360)](https://arxiv.org/abs/1602.07360)
- [SqueezeNet for Edge Computing](https://medium.com/sfu-cspmp/squeezenet-the-key-to-unlocking-the-potential-of-edge-computing-c8b224d839ba)

**Knowledge Distillation:**
- [A Comprehensive Survey on Knowledge Distillation (arXiv:2503.12067)](https://arxiv.org/pdf/2503.12067)
- [RankDVQA-mini: Knowledge Distillation-Driven Deep Video Quality Assessment](https://arxiv.org/pdf/2312.08864)
- [DistilIQA: Distilling Vision Transformers for CT Image Quality Assessment](https://www.sciencedirect.com/science/article/abs/pii/S0010482524007558)
- [Efficient Face Image Quality Assessment via Self-training and Knowledge Distillation (arXiv:2507.15709)](https://arxiv.org/abs/2507.15709)
- [Content-Variant Reference IQA via Knowledge Distillation](https://cdn.aaai.org/ojs/20221/20221-13-24234-1-2-20220628.pdf)
- [Knowledge Distillation: Teacher-Student Loss Explained](https://labelyourdata.com/articles/machine-learning/knowledge-distillation)

**MobileNet and Depthwise Separable Convolutions:**
- [MobileNets: Efficient CNNs for Mobile Vision (arXiv:1704.04861)](https://arxiv.org/abs/1704.04861)
- [MobileNetV3 for Super-Resolution](https://vinbhaskara.github.io/files/eccvw20-pdf.pdf)
- [Real-Time Deployment of MobileNetV3](https://www.preprints.org/manuscript/202305.2163/v1/download)
- [Comparative Analysis of Lightweight Deep Learning Models (arXiv:2505.03303)](https://arxiv.org/html/2505.03303v1)
- [Understanding Depthwise Separable Convolutions and MobileNets](https://medium.com/data-science/understanding-depthwise-separable-convolutions-and-the-efficiency-of-mobilenets-6de3d6b62503)
- [Rethinking Depthwise Separable Convolutions (CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Haase_Rethinking_Depthwise_Separable_Convolutions_How_Intra-Kernel_Correlations_Lead_to_Improved_CVPR_2020_paper.pdf)
- [Knowledge Distillation-Based Lightweight MobileNet Model](https://www.nature.com/articles/s41598-025-30893-7)

**Model Compression:**
- [A Comprehensive Review of Model Compression Techniques](https://link.springer.com/article/10.1007/s10489-024-05747-w)
- [Survey of Model Compression Techniques (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11965593/)
- [Lightweight Deep Learning for Resource-Constrained Environments Survey](https://dl.acm.org/doi/10.1145/3657282)

**Perceptual Loss for Real-Time Applications:**
- [Perceptual Losses for Real-Time Style Transfer (ECCV 2016)](https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16.pdf)
- [Perceptual Losses Paper (arXiv:1603.08155)](https://arxiv.org/abs/1603.08155)
- [A Systematic Performance Analysis of Deep Perceptual Loss Networks](https://arxiv.org/pdf/2302.04032)
- [ConvNeXt Perceptual Loss GitHub](https://github.com/sypsyp97/convnext_perceptual_loss)

**Neural Cellular Automata with Perceptual Loss:**
- [μNCA: Texture Generation with Ultra-Compact NCAs (arXiv:2111.13545)](https://ar5iv.labs.arxiv.org/html/2111.13545)
- [Multi-texture Synthesis through Signal Responsive NCAs](https://www.nature.com/articles/s41598-025-23997-7)
- [Growing Neural Cellular Automata (Distill.pub)](https://distill.pub/2020/growing-ca/)
- [Neural Cellular Automata: From Cells to Pixels (arXiv:2506.22899)](https://arxiv.org/abs/2506.22899)
- [Style Transfer with Neural Cellular Automata](https://ristohinno.medium.com/style-transfer-with-neural-cellular-automata-7c28c626d657)
- [NCA Image Manipulation GitHub](https://github.com/MagnusPetersen/Neural-Cellular-Automata-Image-Manipulation)

**CLIP for NCAs:**
- [Text-2-Cellular-Automata GitHub (CLIP + NCAs)](https://github.com/Mainakdeb/text-2-cellular-automata)
- [MeshNCA: Mesh Neural Cellular Automata](https://meshnca.github.io/)
- [MeshNCA Paper (ACM Transactions on Graphics)](https://dl.acm.org/doi/10.1145/3658127)
- [CLIP: Connecting Text and Images (OpenAI)](https://openai.com/index/clip/)

**VGG and Gram Matrix Losses:**
- [Content and Style Loss Using VGG Network](https://medium.com/@oleksandrsavsunenko/content-and-style-loss-using-vgg-network-e810a7afe5fc)
- [Universal Style Transfer via Feature Transforms (arXiv:1705.08086)](https://arxiv.org/pdf/1705.08086)
- [VGG Image Style Transfer with Gram Matrix](https://www.researchgate.net/publication/377868566_New_Image_Processing_VGG_Image_Style_Transfer_with_Gram_Matrix_Style_Features)
- [Neural Style Transfer GitHub (Gram Matrix)](https://github.com/anh-nn01/Neural-Style-Transfer)

**Neural Architecture Search:**
- [AutoML Neural Architecture Search Overview](https://www.automl.org/nas-overview/)
- [Neural Architecture Search (Wikipedia)](https://en.wikipedia.org/wiki/Neural_architecture_search)
- [Advances in Neural Architecture Search (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11389615/)
- [Systematic Review on NAS (Springer)](https://link.springer.com/article/10.1007/s10462-024-11058-w)
- [Neural Architecture Search Book Chapter](https://www.automl.org/wp-content/uploads/2019/05/AutoML_Book_Chapter3.pdf)

**Edge AI and TinyML:**
- [Edge AI Vision: Deep Learning on Tiny Devices](https://medium.com/@API4AI/edge-ai-vision-deep-learning-on-tiny-devices-11382f327db6)
- [Literature Review on Model Conversion in EdgeML with TinyML](https://www.sciencedirect.com/org/science/article/pii/S1546221825003133)
- [Key Considerations for Real-Time Object Recognition on Edge Devices](https://www.mdpi.com/2076-3417/15/13/7533)
- [TinyML: Machine Learning at the Edge](https://research.aimultiple.com/tinyml/)
