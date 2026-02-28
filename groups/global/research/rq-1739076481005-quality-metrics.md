# Texture Quality Metrics and Human Perception

**Research Topic:** Texture quality metrics that correlate with human perception - FID vs LPIPS vs perceptual texture similarity for different domains

**Completed:** 2026-02-20

---

## Summary

Texture quality assessment has evolved from simple pixel-level metrics (PSNR, SSIM) to deep learning-based perceptual metrics (LPIPS, DISTS) that better align with human perception. Different metrics serve different purposes: FID measures distribution-level similarity (good for dataset comparisons), LPIPS captures perceptual patch similarity (excellent for textures), and DISTS combines structure and texture similarity. Domain-specific considerations are critical—metrics trained on ImageNet struggle with specialized domains (medical imaging, procedural textures). For texture synthesis, LPIPS and DISTS consistently outperform traditional metrics, with DISTS showing particular promise for evaluating texture-rich generation methods like GANs and neural cellular automata.

---

## Key Findings

### 1. Metric Performance Hierarchy

**Deep Learning-Based Metrics (Best for Texture):**
- **LPIPS (Learned Perceptual Image Patch Similarity)**: Computes similarity between deep network activations of image patches. Shown to match human perception well across architectures (SqueezeNet, AlexNet, VGG) and training methods. Lower scores = more perceptually similar. Network alex is fastest and performs best as forward metric.

- **DISTS (Deep Image Structure and Texture Similarity)**: Explicitly designed to tolerate texture resampling. Combines SSIM-like structure measurements with texture similarity. Based on injective mapping from VGG variants. Robust to texture variance (ideal for GAN evaluation) and mild geometric transformations. Shows high promise for modern texture-generating algorithms.

**Traditional Metrics (Limited for Textures):**
- **SSIM/MS-SSIM**: Structural similarity measures. Good for denoising tasks but poor correlation with human perception for complex textures and model-dependent distortions.

- **FID (Fréchet Inception Distance)**: Measures Wasserstein distance between distributions. Useful for dataset-level comparisons but misses fine-grained details and textures. Suffers from "majority rules" problem in high-variation domains.

- **PSNR**: Signal fidelity metric. Poorest correlation with perceptual quality for textures.

### 2. Domain-Specific Limitations

**FID's Critical Failures:**
- **Biomedical/Scientific Imaging**: ImageNet-based Inception features don't correlate with diagnostic content. Lower FID doesn't translate to improved classification/segmentation performance.

- **Domain Adaptation**: Inadequate for zero-shot generation or adaptation setups where distribution shift is intentional.

- **Texture-Rich Domains**: Misses structural details that humans perceive as quality degradation. Improved variant sFID (spatial FID) correlates better with realistic textures.

- **Technical Issues**: Incorrect normality assumptions, poor sample complexity, sensitivity to class frequency in low-overlap datasets (e.g., faces).

**LPIPS Sensitivity:**
- Somewhat oversensitive to texture substitution (texture resampling that humans find acceptable)
- Loses global context that FID captures (trade-off between patch-level and distribution-level assessment)

### 3. Texture-Specific Assessment

**Perceptual Texture Dimensions (Human Studies):**

Research on procedural texture perception identified three primary dimensions humans use:
1. **Feature Density/Coarseness**: How fine or coarse texture elements are
2. **Regularity/Orientation**: Repetitive, directional, uniform patterns vs. randomness
3. **Contrast/Structural Complexity**: Granularity and hierarchical structure

Key insight: Humans use **combinations of features** rather than individual characteristics when judging texture similarity. Classification accuracy improved ~6% when using these perceptual dimensions vs. traditional features.

**Procedural vs. Photorealistic:**
- Procedural textures better balance natural appearance with parameterization
- Humans cluster textures by underlying generative model even without knowledge of the models
- Existing dimensional frameworks struggle with random textures

**Neural Cellular Automata Context:**
- Both LPIPS and DISTS succeed in synthesizing textures visually similar to originals
- Gram Matrix Distance (GMD) used in style transfer emphasizes stylistic similarity over exact pattern replication
- Optimal Transport-based losses provide finer control than feature-statistic matching methods

### 4. Practical Guidelines by Task

**Denoising:**
- MS-SSIM and MAE dominate (though MAE gains statistically insignificant)

**Deblurring/Super-Resolution/Compression:**
- DISTS and LPIPS consistently rank highest
- Both outperform traditional metrics and many sophisticated IQA models

**Texture Synthesis/Generation:**
- LPIPS excels for perceptual similarity where human perception is benchmark
- DISTS handles texture variance better (ideal for GANs, NCAs, diffusion models)
- Gram matrix methods (VGG-based) fundamental for style transfer

**Real-Time Game Rendering:**
- Detail clarity (structural preservation) most critical
- CNNs trained to detect quality-reducing artifacts and predict human perceptibility
- Quantitative metrics (MSE, SNR) insufficient alone—human evaluation remains gold standard for task-specific quality

### 5. Recent Developments (2024-2026)

**Robustness Improvements:**
- **ST-LPIPS**: Shift-tolerant variant less susceptible to imperceptible misalignments
- **E-LPIPS**: Random transformation ensembles for robustness
- **R-LPIPS**: Adversarially robust variant

**Foundation Model Integration:**
- DINOv1 and CLIP-ViT backbones showing improved alignment with human perceptual judgments across distortion types
- VGG-based perceptual losses using Gram matrices remain fundamental but facing limitations
- Transformer-based approaches and diffusion models addressing Gram matrix constraints

**Industry Adoption:**
- LPIPS widely implemented (pip install lpips, PyTorch Lightning integration)
- Real-time rendering research (SIGGRAPH) emphasizing neural-enhanced image quality
- Physically based shading, temporal antialiasing, ray tracing benefiting from perceptual metrics

---

## Deep Dive: Why Deep Features Work

### The "Unreasonable Effectiveness" Phenomenon

Deep network activations work surprisingly well as perceptual similarity metrics across:
- **Architectures**: SqueezeNet, AlexNet, VGG provide similar scores
- **Training Paradigms**: Unsupervised, self-supervised, supervised all perform strongly
- **Tasks**: Works for textures, natural images, artistic content, compression artifacts

**Why This Works:**
1. **Hierarchical Feature Extraction**: Early layers capture low-level texture features (edges, colors), middle layers capture textures and patterns, late layers capture semantic concepts
2. **Learned Perceptual Space**: Training on human judgments aligns feature space with perceptual similarity
3. **Invariance Properties**: Networks learn invariances (translation, scale, minor deformations) that match human perception

### Implementation Architectures

**LPIPS Architecture:**
- Forward pass through pretrained network (typically VGG or AlexNet)
- Extract activations from multiple layers
- Compute L2 distance between activations
- Weight and combine distances across layers
- Trained on human perceptual similarity judgments (2AFC forced-choice experiments)

**DISTS Architecture:**
- VGG variant with injective mapping function
- Separates structure and texture similarity
- Structure: SSIM-like computation on feature maps
- Texture: Statistical moments of feature distributions
- Joint optimization: Match human ratings while minimizing distances between crops from same texture

**VGG Gram Matrix (Style Transfer):**
- Extract features from layers relu1_2, relu2_2, relu3_3, relu4_3
- Compute Gram matrix (feature correlations) for each layer
- Style loss: Match Gram matrices between style image and generated image
- Content loss: Match features at relu3_3
- Captures "which features activate together" = texture/style essence

---

## Connections to Existing Knowledge

### Neural Cellular Automata Quality Assessment

From my research on NCAs:
- **Parameter Efficiency**: NCAs achieve 68-8000 params for texture synthesis—quality metrics must be efficient enough for training loop integration
- **Real-Time Generation**: 25Hz performance means metrics must run <40ms for interactive feedback
- **Multi-Texture Evaluation**: Genomic signal NCAs generate 8+ textures—need per-texture quality assessment
- **Domain Mismatch**: ImageNet-trained FID inappropriate for evaluating organic/procedural patterns

**Recommended Metrics for NCA Training:**
1. **Primary**: LPIPS (fast, perceptually aligned, works with small patches)
2. **Secondary**: DISTS (texture robustness for varied outputs)
3. **Style Transfer**: Gram matrix loss (proven for texture objectives)
4. **Distribution**: KID instead of FID (better sample efficiency, polynomial kernel)

### Cascade Routing Systems

From cascade routing research:
- Quality predictors must run <1ms to avoid exceeding Tier 1 model cost (µNCA: 68 params, 25Hz)
- Lightweight heuristics (edge density, entropy, variance) achieve r=0.70-0.83 correlation with complexity
- BERT-based routers: 1.2-20ms latency—too expensive for real-time texture routing
- Solution: Use LPIPS/DISTS for offline quality validation, heuristics for runtime routing

### Hierarchical NCA Systems

Multi-scale architectures need scale-appropriate metrics:
- **Coarse scales**: FID or distribution-level metrics (global coherence)
- **Fine scales**: LPIPS or DISTS (local texture quality)
- **Cross-scale**: Perceptual pyramid metrics combining both

### Diffusion Models

Hybrid Diff-NCA systems (336k params) bridge NCA and diffusion:
- Diffusion models: FID as standard but LPIPS better for texture richness
- One-step methods (LCMs): Quality degradation best measured with DISTS (texture tolerance)
- Interactive applications: LPIPS preferred for perceptual alignment at <1s generation

---

## Follow-Up Questions

### High Priority Research Gaps

1. **Lightweight Perceptual Metrics for Real-Time Systems**: Can we distill LPIPS/DISTS to <10k params while maintaining perceptual alignment? Critical for NCA cascade routing where router overhead must be << 1ms.

2. **Domain-Adapted Perceptual Metrics**: How much does fine-tuning LPIPS on domain-specific human judgments improve correlation? E.g., LPIPS trained on procedural texture similarity judgments vs. photographic images.

3. **Multi-Scale Perceptual Assessment**: Optimal strategy for evaluating hierarchical texture generation (coarse-to-fine)? Should metrics combine global distribution matching with local perceptual similarity?

4. **Texture-Specific CLIP Metrics**: Can CLIP embeddings provide zero-shot texture quality assessment? "High-quality wood grain texture" vs. actual output similarity.

5. **Genome-Aware Metrics for NCAs**: Do metrics need to account for genomic signal preservation in multi-texture NCAs? Quality could degrade through genome corruption even if per-frame output looks acceptable.

### Experimental Validation Needed

1. **Benchmark Creation**: Comprehensive texture synthesis benchmark with human ratings across:
   - Organic textures (NCA/RD-generated)
   - Procedural textures (noise-based, algorithmic)
   - Photorealistic textures (diffusion-generated)
   - Hybrid textures (Diff-NCA outputs)

2. **Metric Comparison Study**: Systematic evaluation of FID, LPIPS, DISTS, GMD, SSIM on texture benchmark with analysis of:
   - Correlation with human ratings per texture category
   - Computational cost vs. perceptual alignment trade-off
   - Failure modes and artifact sensitivity
   - Sample efficiency requirements

3. **Domain Transfer Testing**: Train perceptual metrics on one texture domain, test on others:
   - Does LPIPS trained on natural images generalize to procedural patterns?
   - Can single metric work across organic (NCA), procedural (Perlin), and photorealistic (diffusion) domains?

4. **Real-Time Integration**: Performance analysis of metrics in training loops:
   - Gradient flow quality through LPIPS vs. DISTS vs. Gram matrix
   - Training stability when optimizing for different perceptual objectives
   - Multi-objective optimization (perceptual + speed + diversity)

---

## Sources

### Perceptual Metrics Research
- [Learned Perceptual Image Patch Similarity (LPIPS) - PyTorch-Metrics Documentation](https://lightning.ai/docs/torchmetrics/stable/image/learned_perceptual_image_patch_similarity.html)
- [A Review of Image Quality Metrics for Image Synthesis Models](https://blog.paperspace.com/review-metrics-image-synthesis-models/)
- [LPIPS GitHub Repository](https://github.com/richzhang/PerceptualSimilarity)
- [Evaluation of Image Generation - FID, LPIPS, SSIM, KID](https://medium.com/@wangdk93/evaluation-of-image-generation-ec402191d4d7)
- [ShiftTolerant-LPIPS GitHub](https://github.com/abhijay9/ShiftTolerant-LPIPS)
- [E-LPIPS: Robust Perceptual Similarity via Random Transformation Ensembles](https://www.semanticscholar.org/paper/E-LPIPS:-Robust-Perceptual-Image-Similarity-via-Kettunen-H%C3%A4rk%C3%B6nen/56bccd3519f34ab7eefbccc30df6305d558010b6)
- [Foundation Models Boost Low-Level Perceptual Similarity](https://pdxscholar.library.pdx.edu/cgi/viewcontent.cgi?article=1382&context=compsci_fac)
- [Measuring What Matters: Objective Metrics for Image Generation Assessment](https://huggingface.co/blog/PrunaAI/objective-metrics-for-image-generation-assessment)

### LPIPS Deep Dive
- [The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (Paper)](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_The_Unreasonable_Effectiveness_CVPR_2018_paper.pdf)
- [The Unreasonable Effectiveness of Deep Features (Project Page)](https://richzhang.github.io/PerceptualSimilarity/)
- [SSIM vs. LPIPS Comparison](https://eureka.patsnap.com/article/ssim-vs-lpips-which-metric-should-you-trust-for-image-quality-evaluation)

### FID Limitations
- [Fréchet Inception Distance - Wikipedia](https://en.wikipedia.org/wiki/Fr%C3%A9chet_inception_distance)
- [Rethinking FID: Towards a Better Evaluation Metric](https://arxiv.org/html/2401.09603v2)
- [Rethinking FID Paper (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/papers/Jayasumana_Rethinking_FID_Towards_a_Better_Evaluation_Metric_for_Image_Generation_CVPR_2024_paper.pdf)
- [What Is Fréchet Inception Distance (FID)?](https://www.techtarget.com/searchenterpriseai/definition/Frechet-inception-distance-FID)
- [Metrics for Evaluating Synthetic Images](https://apxml.com/courses/evaluating-synthetic-data-quality/chapter-5-specialized-model-specific-metrics/evaluating-synthetic-images)

### DISTS and Full-Reference Metrics
- [Comparison of Full-Reference Image Quality Models (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7817470/)
- [DISTS GitHub Repository](https://github.com/dingkeyan93/DISTS)
- [Image Quality Assessment: Unifying Structure and Texture Similarity](https://www.cns.nyu.edu/pub/lcv/ding20a-preprint.pdf)
- [Structural Similarity Index Measure - Wikipedia](https://en.wikipedia.org/wiki/Structural_similarity_index_measure)
- [SSIM: Structural Similarity Index - Imatest](https://www.imatest.com/docs/ssim/)
- [Evaluation of Objective Image Quality Metrics for High-Fidelity Compression](https://arxiv.org/html/2509.13150)
- [Comparison of Full-Reference IQA Models (Springer)](https://link.springer.com/article/10.1007/s11263-020-01419-7)

### Texture Perception Research
- [Visual Perception of Procedural Textures (PLOS ONE)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130335)
- [Visual Perception of Procedural Textures (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4481328/)
- [Multi-texture Synthesis through Neural Cellular Automata](https://www.nature.com/articles/s41598-025-23997-7)
- [µNCA: Texture Generation with Ultra-Compact NCAs](https://ar5iv.labs.arxiv.org/html/2111.13545)

### VGG Perceptual Loss and Style Transfer
- [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16.pdf)
- [Neural Style Transfer GitHub](https://github.com/anh-nn01/Neural-Style-Transfer)
- [VGG Image Style Transfer with Gram Matrix (IEEE)](https://ieeexplore.ieee.org/iel7/10405420/10405394/10405398.pdf)
- [Universal Style Transfer via Feature Transforms](https://arxiv.org/pdf/1705.08086)
- [Transformer-Based Neural Texture Synthesis](https://dl.acm.org/doi/fullHtml/10.1145/3512353.3512366)

### Game Development and Real-Time Rendering
- [Texture Quality Metrics for 3D Models](https://www.sloyd.ai/blog/texture-quality-metrics-for-3d-models)
- [Learning to Predict Perceptual Visibility in Games](https://www.nature.com/articles/s41598-024-78254-0)
- [Advances in Real-Time Rendering SIGGRAPH 2024](https://advances.realtimerendering.com/s2024/index.html)
- [Advances in Real-Time Rendering SIGGRAPH 2025](https://advances.realtimerendering.com/s2025/index.html)

### Additional Resources
- [TISE: Bag of Metrics for Text-to-Image Synthesis Evaluation](https://arxiv.org/abs/2112.01398)
- [Subjective and Objective Visual Quality Assessment of Textured 3D Meshes](https://arxiv.org/pdf/2102.03982)
- [Structural Texture Similarity Metrics](https://users.eecs.northwestern.edu/~pappas/papers/zujovic_tip13.pdf)
- [Perceptual Visual Quality Assessment: Principles and Methods](https://arxiv.org/html/2503.00625v1)
- [Benchmarking Multi-Dimensional Quality Evaluator for Text-to-3D](https://arxiv.org/html/2412.11170)
