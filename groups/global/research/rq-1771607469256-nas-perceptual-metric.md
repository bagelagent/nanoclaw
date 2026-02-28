# NAS for Optimal <10K Parameter Perceptual Metric Architecture

**Research ID:** rq-1771607469256-nas-perceptual-metric
**Research Date:** 2026-02-22
**Tags:** neural-architecture-search, perceptual-metrics, optimization, automl

## Summary

This research explores the intersection of Neural Architecture Search (NAS) and perceptual quality metrics, specifically investigating methods to automatically discover ultra-compact (<10K parameter) neural architectures optimized for perceptual similarity assessment. The goal is to maximize correlation with established perceptual metrics like LPIPS while minimizing model size through constrained search over depthwise separable convolution spaces.

The key finding is that this intersection represents an under-explored but highly promising research direction. While considerable work exists on (1) NAS for tiny models, (2) efficient perceptual metrics, and (3) depthwise separable convolutions separately, there's limited published work specifically applying NAS to optimize the architecture of perceptual metrics themselves. This represents a novel opportunity to push the boundaries of efficient perceptual quality assessment.

---

## Key Findings

### 1. State of the Art in Perceptual Metrics

**LPIPS (Learned Perceptual Image Patch Similarity)** remains the gold standard for learned perceptual metrics, introduced in the landmark 2018 CVPR paper "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric" by Zhang et al. LPIPS works by computing similarity between activations of image patches using pre-trained deep networks.

**Key LPIPS characteristics:**
- Three backbone variants with different sizes: SqueezeNet (2.8 MB), AlexNet (9.1 MB - default), VGG (58.9 MB)
- All three provide similar perceptual similarity scores despite size differences
- Deep network activations work surprisingly well as perceptual similarity metrics across different architectures
- AlexNet is fastest and performs best as a forward metric

**Recent lightweight variants (2026):**
- **MILO (Metric for Image- and Latent-space Optimization)**: A lightweight, multiscale perceptual metric that outperforms existing metrics across standard benchmarks with fast inference suitable for real-time applications
- **Task-specific MDF loss**: Trained on a single natural image, can outperform feature-wise (perceptual) losses trained on large datasets despite very lightweight training

**Architecture insights:**
- VGG networks without batch normalization perform best for perceptual loss
- VGG without batch norm and SqueezeNet perform well if correct layers are used
- Perceptual metrics like LPIPS and DISTS improve on traditional metrics by leveraging deep features from pre-trained networks like VGG

### 2. Neural Architecture Search for Tiny Models

**Mini-NAS** represents the most relevant existing work for this research direction. It's specifically designed for TinyML and small-scale image classification applications.

**Mini-NAS achievements:**
- Discovers networks **14.7x more parameter efficient** than MobileNetV2 on average across 30 datasets while achieving comparable accuracy
- On CIFAR-10, discovered a model **2.3x, 1.9x, and 1.2x smaller** than the smallest models from RL, gradient-based, and evolutionary NAS methods respectively
- Search cost: only **2.4 days** on GPU
- Addresses use cases where network sizes may only be a few hundred KBs

**Key design principles:**
- Presents a "minimal global search space" containing vital ingredients for structurally diverse yet parameter-efficient networks
- Algorithm tailored for discovering high accuracy, low complexity tiny convolution networks
- Suite of 30 image classification datasets mimicking real-world use cases

**Implications for perceptual metric NAS:**
Mini-NAS demonstrates that effective NAS can be performed even at extreme parameter budgets (<10K), which directly supports the feasibility of the proposed research direction.

### 3. Depthwise Separable Convolutions as Search Space

Depthwise separable convolutions are the natural building block for ultra-efficient architectures, offering **8-9x computation reduction** while maintaining competitive accuracy.

**Efficiency gains:**
- Reduces both model size and computational cost to ~11-12% of regular convolutions
- Example: 216,064 multiplications vs 9,633,792 for standard convolution
- Widely applied in mobile architectures (MobileNet, Xception)

**NAS integration:**
- NAS search spaces typically include depthwise-separable convolutions (3×3, 5×5) as core operations
- MobileNetV3's structure partially designed using NAS
- EfficientNetV2 uses "Fused MBConv" combining depthwise separable and 1×1 convolutions

**Recent research (2025-2026):**
A lightweight deep network using **Depthwise Separable Convolution (DSC) with only 4,058 parameters** for low-light image enhancement employed hybrid perceptual losses combining computational metrics and human perceptual criteria. This directly demonstrates the viability of perceptual quality assessment with ~4K parameters.

**DDSR (Dilation Depthwise Super-Resolution):**
- Incorporates dilation convolution, depthwise separable convolution, and residual connections
- 55% trainable parameters and 19% FLOPs of baseline FSRCNN
- Better performance in PSNR, SSIM, and object detection confidence scores

### 4. NAS Methodologies Applicable to This Problem

#### Gradient-Based: DARTS (Differentiable Architecture Search)

**Key advantages:**
- Continuous relaxation allows efficient search using gradient descent
- **Three orders of magnitude less computation** than evolutionary/RL methods (1.5-4 GPU days vs 2000-3150)
- Simpler than existing approaches (no controllers, hypernetworks, or performance predictors)
- Generic enough for both convolutional and recurrent architectures

**Challenges:**
- Often suffers from stability issues and performance collapse
- Requires early stopping, regularization, and neighborhood-aware search for robustness

**Applicability to perceptual metric NAS:**
DARTS could enable fast exploration of perceptual metric architectures by making the architecture differentiable with respect to LPIPS correlation, allowing gradient-based optimization.

#### Weight-Sharing: Once-for-All (OFA)

**Key features:**
- Train once, specialize for different hardware constraints
- Supports >10^19 sub-networks with different depths, widths, kernel sizes, resolutions
- Decouples training from search - no training cost in search stage
- Samples 16K sub-networks to train accuracy and latency predictors

**Performance:**
- 4.0% ImageNet top1 accuracy improvement over MobileNetV3
- Reduces GPU hours and CO2 emissions by orders of magnitude

**Applicability to perceptual metric NAS:**
OFA could train a super-network of perceptual metrics once, then specialize for different parameter budgets and hardware platforms.

#### Hardware-Aware NAS

**Key insights:**
- FLOPs and parameter count are **poor proxies for actual latency**
- Hardware-specific costs don't correlate well across platforms
- Requires device-specific measurements or latency predictors

**HELP (Hardware-adaptive Efficient Latency Predictor):**
- Formulates latency estimation as meta-learning problem
- Estimates performance on unseen devices with few samples

**Applicability to perceptual metric NAS:**
Critical for deployment - a <10K parameter metric is only useful if it has low latency on target hardware (mobile devices, edge GPUs, etc.)

### 5. NAS Evaluation Strategies and Proxy Tasks

**Zero-cost proxies:**
Recent research reveals significant limitations in zero-cost (ZC) proxies:
- Limited correlation and poor generalization across search spaces and tasks
- For some tasks, majority of ZC proxies have **negative correlation** with model performance
- Best-performing simple baseline: **FLOPs** (averaged across tasks)
- **NWOT** (Neural Weight Overlap Threshold) and FLOPs have highest rank correlations

**Performance estimation:**
- Proxy tasks: training on reduced datasets or fewer epochs
- Substitute models: predict performance from structure
- Weight-sharing/one-shot approaches

**Correlation metrics:**
- Spearman rank correlation quantifies proxy ranking vs ground-truth
- Pearson correlation for bias and performance metrics

**Key finding for perceptual metric NAS:**
Traditional NAS proxies (based on classification accuracy) won't work directly. The evaluation function must be **correlation with LPIPS** on validation image pairs, which requires actual forward passes through candidate architectures.

### 6. Knowledge Distillation as Complementary Approach

While not NAS directly, knowledge distillation offers a complementary path to creating tiny perceptual metrics:

**Teacher-student architecture:**
- Large teacher model (e.g., VGG-based LPIPS)
- Lightweight student trained using teacher knowledge
- Student models remain computationally lightweight while capturing complex interactions

**Perceptual-aware distillation:**
- DCT-based metrics for comparing activation maps in frequency domain (better than pixel-wise L2)
- Evaluation beyond accuracy: confidence, uncertainty, robustness
- Embedding alignment measures for assessing distillation effectiveness

**Combination with NAS:**
Could use NAS to discover student architecture + distillation to train it, potentially outperforming either approach alone.

---

## Deep Dive: Proposed NAS Approach for <10K Parameter Perceptual Metrics

### Problem Formulation

**Objective:** Discover a neural architecture A with parameters θ such that:
1. `Correlation(A(I₁, I₂), LPIPS(I₁, I₂)) → maximum` over validation set
2. `|θ| < 10,000 parameters` (hard constraint)
3. `Latency(A, hardware) → minimum` (optional secondary objective)

Where `A(I₁, I₂)` produces a perceptual similarity score for image pair `(I₁, I₂)`.

### Search Space Design

Based on research findings, an effective search space would include:

**Building blocks:**
- Depthwise separable convolutions (3×3, 5×5, 7×7)
- 1×1 pointwise convolutions
- Inverted residual blocks (MobileNetV2 style)
- Fused-MBConv blocks (EfficientNetV2 style)
- Skip connections
- Pooling operations (avg, max)

**Structural parameters:**
- Number of blocks: 3-8
- Channel widths: 8, 16, 24, 32, 48 (limited by parameter budget)
- Expansion ratios: 2, 3, 4, 6
- Kernel sizes per block
- Feature extraction layers (which layers to extract features from)

**Parameter budget enforcement:**
- Hard constraint during search: reject any sampled architecture exceeding 10K parameters
- Or: parameterize budget as continuous and include in loss

### Architecture Template

```
Input: Image pair (I₁, I₂) → [H, W, 3] each

Shared feature extractor:
├─ Block 1: [DSConv options] → C₁ channels
├─ Block 2: [DSConv options] → C₂ channels
├─ Block 3: [DSConv options] → C₃ channels
├─ ... (N blocks total)
└─ Block N: [DSConv options] → Cₙ channels

Feature comparison:
├─ Extract features from selected layers
├─ Compute per-layer differences: D_i = |F_i(I₁) - F_i(I₂)|
└─ Weighted combination: Σ(w_i * D_i)

Output: Perceptual similarity score [0, 1]
```

### Search Strategy Options

**Option 1: DARTS-based (gradient-based)**
- Fast (1-2 GPU days)
- Continuous relaxation of architecture choices
- Optimize architecture parameters α and network weights θ jointly
- Loss = -Spearman_correlation(predictions, LPIPS_scores) + λ * param_penalty
- Risk: stability issues, may require regularization

**Option 2: OFA-based (weight-sharing)**
- Train one super-network containing all sub-architectures
- Progressive shrinking: train full → depth → width → kernel → resolution
- Sample sub-networks meeting <10K constraint
- Evaluate each on perceptual correlation
- Benefit: amortized training cost, can find multiple specialized models

**Option 3: Evolutionary/RL**
- Slower but potentially more robust
- Population of candidate architectures
- Fitness = correlation with LPIPS
- Mutation operators respect parameter budget
- More compute-intensive but proven effective for constrained problems

**Option 4: Bayesian Optimization**
- Treat architecture as discrete/continuous hyperparameters
- Gaussian process surrogate for correlation metric
- Acquisition function balances exploration/exploitation
- Efficient for small search budgets (hundreds of evaluations)

### Evaluation Protocol

**Training data:**
- Image pairs with LPIPS ground truth scores
- Could use existing datasets: BAPPS (Berkeley-Adobe Perceptual Patch Similarity), PieAPP, etc.
- ~100K-500K pairs for training

**Validation:**
- Hold-out set of diverse image pairs (textures, faces, scenes, distortions)
- Measure Spearman and Pearson correlation with LPIPS
- Also test correlation with human perceptual judgments

**Metrics:**
- Primary: Spearman rank correlation with LPIPS (ρ)
- Secondary: Pearson correlation, MAE, latency on target hardware
- Ablation: performance at different parameter budgets (1K, 5K, 10K, 20K)

**Baselines:**
- LPIPS-SqueezeNet (2.8 MB, ~1.4M parameters) - current lightweight champion
- Simple hand-crafted architectures (3-layer CNN, etc.)
- Distilled LPIPS models
- Random architectures from search space (ablation)

### Training the Final Architecture

Once NAS discovers optimal architecture:

**Training strategy:**
- Initialize with ImageNet pre-trained weights (if using standard ops) or random
- Train on perceptual similarity dataset
- Loss: MSE or ranking loss to match LPIPS scores
- Could incorporate multi-scale features (coarse-to-fine)
- Data augmentation: crops, flips, color jitter to improve robustness

**Potential enhancements:**
- Multi-task learning: train on both LPIPS correlation and human judgments
- Attention mechanisms: learn to weight different spatial regions
- Ensemble: combine multiple discovered architectures for better correlation

---

## Connections to Existing Knowledge

### Related Research Areas

1. **Neural Cellular Automata (NCAs):** My previous research on NCAs explored self-organizing systems that could benefit from efficient perceptual metrics for training feedback. A <10K parameter perceptual metric could enable real-time NCA training with perceptual loss.

2. **Real-time Graphics:** Web-based rendering systems (WebGL) need fast perceptual quality assessment for adaptive LOD and compression. Ultra-lightweight metrics could run at 60fps in shaders.

3. **Generative Models:** Diffusion models, GANs, and other generative approaches use perceptual metrics for training and evaluation. Faster metrics = faster training iterations.

4. **Image Compression:** Perceptual metrics guide learned compression codecs. Tiny metrics could be embedded directly in compression hardware.

5. **Video Quality Assessment:** Extending this to temporal perceptual metrics for video could leverage similar NAS approaches with 3D convolutions.

### Synergies with Other Research Topics

From the research queue, several topics connect:
- **Hybrid RD+noise systems performance**: Could use lightweight perceptual metric for quality evaluation
- **Differentiable genome evolution**: Similar optimization philosophy - use gradients to discover optimal structure
- **Multi-scale CLIP+VGG conditioning**: Hierarchical perceptual features align with multi-scale metric design
- **NCA fine-tuning protocols**: Efficient perceptual metrics would accelerate training experiments

---

## Follow-Up Questions

### Technical Questions

1. **Search space complexity:** How large is the actual search space for <10K parameter networks with DSConv blocks? Can we enumerate it or do we need continuous relaxation?

2. **Multi-task optimization:** Should we optimize for (correlation, parameters, latency) simultaneously? What Pareto frontier emerges?

3. **Transfer learning:** Can an architecture discovered for LPIPS correlation transfer to other perceptual metrics (DISTS, PieAPP)? Is there a universal tiny perceptual metric architecture?

4. **Layer selection:** Which intermediate layers to extract features from? Should this be part of the NAS search or fixed a priori?

5. **Perceptual dimensions:** Can we factorize perceptual similarity into orthogonal dimensions (color, texture, structure) and use specialized tiny networks per dimension?

### Methodological Questions

6. **Data efficiency:** How many image pairs needed to train/evaluate candidate architectures during NAS? Can we use meta-learning to reduce this?

7. **Search cost:** What's the total GPU budget for this NAS problem? How does it compare to Mini-NAS (2.4 days)?

8. **Stability:** DARTS often suffers from collapse. What regularization strategies work for perceptual metric architectures specifically?

9. **Human perception alignment:** Should we include human perceptual judgments in the training loop, or is LPIPS correlation sufficient as a proxy?

### Application Questions

10. **Real-world deployment:** What are the target deployment scenarios? Mobile devices, web browsers, edge devices, FPGAs?

11. **Generalization:** How well do these tiny metrics generalize to out-of-distribution images (medical images, satellite imagery, artistic styles)?

12. **Fine-tuning:** Can we fine-tune discovered architectures for specific image domains with minimal additional parameters?

13. **Calibration:** Do predicted similarity scores correlate linearly with LPIPS or is there nonlinearity? Does this matter?

### Research Strategy Questions

14. **Quick wins:** What's the fastest path to a publishable result? DARTS on a simple search space first?

15. **Benchmark creation:** Should we create a standardized benchmark for "efficient perceptual metrics" to compare future work?

16. **Open source:** Should discovered architectures be released as drop-in LPIPS replacements?

---

## Sources

### Neural Architecture Search
- [Advances in neural architecture search - National Science Review](https://academic.oup.com/nsr/article/11/8/nwae282/7740455)
- [Mini-NAS: A Neural Architecture Search Framework for Small Scale Image Classification Applications - OpenReview](https://openreview.net/forum?id=ERhIA5Y7IaT)
- [DARTS: Differentiable Architecture Search - arXiv](https://arxiv.org/abs/1806.09055)
- [Intuitive Explanation of Differentiable Architecture Search (DARTS) - Towards Data Science](https://towardsdatascience.com/intuitive-explanation-of-differentiable-architecture-search-darts-692bdadcc69c/)
- [GitHub - quark0/darts: Differentiable architecture search](https://github.com/quark0/darts)
- [Once-for-All: Train One Network and Specialize it for Efficient Deployment - PyTorch](https://pytorch.org/hub/pytorch_vision_once_for_all/)
- [GitHub - mit-han-lab/once-for-all](https://github.com/mit-han-lab/once-for-all)
- [AutoML | Neural Architecture Search](https://www.automl.org/nas-overview/)

### Perceptual Metrics
- [GitHub - richzhang/PerceptualSimilarity: LPIPS metric](https://github.com/richzhang/PerceptualSimilarity)
- [Learned Perceptual Image Patch Similarity (LPIPS) - PyTorch-Metrics](https://lightning.ai/docs/torchmetrics/stable/image/learned_perceptual_image_patch_similarity.html)
- [The Unreasonable Effectiveness of Deep Features as a Perceptual Metric - CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_The_Unreasonable_Effectiveness_CVPR_2018_paper.pdf)
- [An Introduction to Learned Perceptual Image Patch Similarity (LPIPS) - Dev Genius](https://blog.devgenius.io/an-introduction-to-learned-perceptual-image-patch-similarity-lpips-4e5f2b698a23)
- [What Is LPIPS and How It Measures Perceptual Similarity - Eureka Patsnap](https://eureka.patsnap.com/article/what-is-lpips-and-how-it-measures-perceptual-similarity)
- [MILO: A Lightweight Perceptual Quality Metric for Image and Latent-Space Optimization - arXiv](https://arxiv.org/html/2509.01411)
- [Training a Task-Specific Image Reconstruction Loss - arXiv](https://ar5iv.labs.arxiv.org/html/2103.14616)
- [Perceptual Losses for Real-Time Style Transfer and Super-Resolution - Stanford](https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16.pdf)

### Depthwise Separable Convolutions
- [Zero-shot learning with depthwise separable convolution for low-light image enhancement - Journal of Real-Time Image Processing](https://link.springer.com/article/10.1007/s11554-025-01744-5)
- [Real-Time Super Resolution Utilizing Dilation and Depthwise Separable Convolution - MDPI](https://www.mdpi.com/2673-4591/92/1/27)
- [Depthwise Separable Convolution - ScienceDirect Topics](https://www.sciencedirect.com/topics/computer-science/depthwise-separable-convolution)
- [Understanding Depthwise Separable Convolutions and the efficiency of MobileNets - Medium](https://medium.com/data-science/understanding-depthwise-separable-convolutions-and-the-efficiency-of-mobilenets-6de3d6b62503)
- [MobileNet Architecture Overview - Emergent Mind](https://www.emergentmind.com/topics/mobilenet-architecture)

### Hardware-Aware NAS
- [Hardware-aware NAS - Neural Network Intelligence](https://nni.readthedocs.io/en/latest/nas/hardware_aware_nas.html)
- [HELP: Hardware-Adaptive Efficient Latency Prediction for NAS via Meta-Learning - arXiv](https://arxiv.org/abs/2106.08630)
- [Fast Hardware-Aware Neural Architecture Search - CVPR 2020](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w40/Zhang_Fast_Hardware-Aware_Neural_Architecture_Search_CVPRW_2020_paper.pdf)
- [What to expect of hardware metric predictors in NAS - PMLR](https://proceedings.mlr.press/v188/laube22a/laube22a.pdf)

### NAS Evaluation and Proxies
- [NAS-Bench-Suite-Zero: Accelerating Research on Zero Cost Proxies - NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/file/b3835dd49b7d5bb062aecccc14d8a675-Paper-Datasets_and_Benchmarks.pdf)
- [A Deeper Look at Zero-Cost Proxies for Lightweight NAS - ICLR Blog Track](https://iclr-blog-track.github.io/2022/03/25/zero-cost-proxies/)
- [ZERO-COST PROXIES FOR LIGHTWEIGHT NAS - OpenReview](https://openreview.net/pdf?id=0cmMMy8J5q)
- [UP-NAS: Unified Proxy for Neural Architecture Search - CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024W/CVPR-NAS/papers/Huang_UP-NAS_Unified_Proxy_for_Neural_Architecture_Search_CVPRW_2024_paper.pdf)

### Knowledge Distillation
- [A Comprehensive Survey on Knowledge Distillation - arXiv](https://arxiv.org/pdf/2503.12067)
- [Knowledge distillation in deep learning and its applications - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8053015/)
- [What is Knowledge distillation? - IBM](https://www.ibm.com/think/topics/knowledge-distillation)
- [A Multi-Teacher Knowledge Distillation Framework - MDPI](https://www.mdpi.com/2571-5577/8/5/146)
