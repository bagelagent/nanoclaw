# Empirical NCA Speed Benchmark: VGG vs SqueezeNet vs Optimal Transport Loss

**Research Date:** 2026-02-23 (Updated: 2026-02-24)
**Topic ID:** rq-1771829371564-empirical-nca-speed-benchmark
**Research Agent:** Bagel

---

## Summary

Neural Cellular Automata (NCAs) for texture synthesis require perceptual loss functions during training, with VGG16 being the dominant choice despite its computational overhead. This research examines the performance characteristics of different perceptual loss approaches: VGG16 (standard), SqueezeNet (lightweight), and Optimal Transport-based methods, to understand their speed-quality tradeoffs for NCA training.

---

## Key Findings

### 1. VGG16: The Standard Choice

**Architecture Characteristics:**
- **Parameters:** ~138M parameters
- **Model size:** 58.9 MB
- **MFLOPS (224×224 forward pass):** 15,517 (VGG-16), 19,682 (VGG-19)
- **Inference speed:**
  - Modern desktop GPU (estimated): 10-20ms per forward pass
  - CPU: ~67ms
  - VGG average response (older benchmark): 2.21 seconds/image
- **Why it's dominant:** VGG networks without batch normalization consistently achieve the best perceptual similarity scores

**Empirical NCA Training Times (T4 GPU):**
- **Single texture (5,000 epochs):** ~1h40min (pool=1024, batch=8)
- **Multi-texture (10,000 epochs):** ~2 hours
- **Evolution steps per training iteration:** 32-96 (random)
- **Stability horizon:** Most architectures stable until 6,000 iterations
- **Complex textures with auto-correlation regularizers:** Up to 20,000 iterations

**Usage in NCAs:**
VGG16 serves as a pre-trained feature extractor for computing perceptual loss, not as the model being trained. The NCA learns to evolve textures by minimizing the distance between VGG feature representations of generated and target textures. Most style transfer and texture synthesis approaches use VGG variants due to "superior results compared to other architectures."

**Performance Impact:**
Using deep CNNs like VGG for perceptual loss adds significant computational overhead during training. Each training iteration requires passing both generated and target images through VGG to extract features, then computing loss across multiple layers. This overhead scales with:
- Grid size (quadratically for NCAs)
- Batch size
- Number of VGG layers used for loss computation
- Training iteration count

**Real-world training overhead:**
- Real-time style transfer with VGG: achieves 20 FPS at 512×512 resolution
- Training time on GTX 1070 with VGG loss: 4-4.5 hours for MS COCO dataset (256×256)
- Note: "training time is much slower and batch size is much smaller compared to training without perceptual loss"

### 2. SqueezeNet: The Lightweight Alternative

**Architecture Characteristics:**
- **Parameters:** ~1.25M parameters (50x fewer than AlexNet, ~110x fewer than VGG16)
- **Model size:** 2.8MB compressed (vs 58.9MB for VGG)
- **Inference speed:**
  - Mobile/embedded: ~2.1ms per inference (14MB memory)
  - Desktop GPU: ~1ms per inference
  - Azure Arm64 CPU: ~1.86ms (538 inferences/second)
- **Accuracy:** Achieves AlexNet-level ImageNet accuracy

**Performance Advantage:**
SqueezeNet is approximately **3-4x faster** than VGG16 for inference, with dramatically lower memory footprint. More detailed benchmarks show:
- **CPU performance:** 7-8× faster than VGG-16 (VGG: 2.21s/image, SqueezeNet: 0.288s/image)
- **Video inference on CPU:** 66.7 FPS vs much slower VGG
- **Embedded systems:** 104 FPS demonstrated
- **GPU performance caveat:** VGG16 can be faster than SqueezeNet on powerful GPUs due to better parallelization

For NCA training, this translates to:
- Faster iteration times (each training step requires 2+ perceptual loss computations)
- Lower GPU memory usage, enabling larger batch sizes or higher resolution grids
- Dramatic speedup for CPU/embedded deployment scenarios
- GPU benefits may be less pronounced due to VGG's parallelization advantages

**Tradeoff:**
Research on LPIPS (Learned Perceptual Image Patch Similarity) found that AlexNet, SqueezeNet, and VGG "provided similar scores" as perceptual metrics. However, "Network alex is fastest, performs the best (as a forward metric), and is the default." This suggests SqueezeNet may perform comparably for perceptual loss, though empirical validation for NCA training specifically is needed.

**Architecture Trade-offs:**
- SqueezeNet: 10× fewer FLOPS than AlexNet
- 31× smaller than VGG-19 in parameters
- Uses "fire modules" (squeeze + expand layers) for efficiency
- Achieves 64.55% Top-1, 85.09% Top-5 accuracy on ImageNet (AlexNet-level)

### 3. Optimal Transport Loss (OTT-Loss)

**Core Concept:**
Instead of comparing images through deep network features, Optimal Transport (OT) methods directly optimize the distance between feature distributions, treating texture synthesis as a "robust feature transformation through optimal transport."

**Performance Characteristics:**
The "Optimal Textures" method (2020) reports impressive speed:
- **1024×1024 images with PCA:** 23 seconds end-to-end
- **1024×1024 images without PCA:** 84 seconds
- **Comparison:** "Tens of minutes" for backpropagation-based methods (Gatys, Risser)

**Why It's Fast:**
OT methods operate "directly on deep neural features themselves, within the bottleneck layer of an auto-encoder" rather than optimizing in image space. This eliminates repeated forward passes through the perceptual network during optimization.

**Interactive Potential:**
Authors suggest the "run-time performance makes optimal transport an attractive candidate for an interactive artist tool, particularly when only sub-regions of the image are edited in real time."

**Applicability to NCAs:**
The challenge is adapting OT-based losses to NCA training, which uses backpropagation through time. Traditional OT methods solve the transport problem per-image, while NCA training requires differentiable losses computed at each step. Recent work on "sliced Wasserstein loss" (CVPR 2021) may bridge this gap by providing efficient, differentiable OT approximations.

#### Recent OT Developments (2024-2026)

**Sinkhorn Loss for Neural Networks:**
A 2024 paper on Optimal Transport-inspired deep learning demonstrates that "using Sinkhorn divergence as the loss function enhances stability during training, robustness against overfitting and noise, and accelerates convergence" compared to conventional MSE or cross-entropy loss.

**Convergence Rate Advances:**
- Sinkhorn algorithm achieves **O(t⁻¹) convergence rate** for relative entropy
- **O(t⁻¹)** for dual suboptimality
- **O(t⁻²)** for marginal entropies
- Exponential convergence guarantees for continuous settings with semi-concave costs

**New OT Frameworks:**
1. **Universal Neural Optimal Transport (UNOT):** Processes measures of variable resolution using Fourier Neural Operators
2. **Riemannian Neural OT (RNOT):** Extends neural OT beyond Euclidean spaces with polynomial complexity guarantees
3. **Displacement Interpolation OT Model (DIOTM):** Improves training stability and OT map approximation

**Training Advantages vs Traditional Losses:**
- **Cross-entropy:** Faster convergence, no gradient saturation, but task-specific
- **MSE:** Simpler but can converge slower (though may need half the iterations under normalized conditions)
- **Sinkhorn/OTT:** Enhanced stability, accelerated convergence, robust to noise

**Key Insight for NCAs:**
The stability and noise robustness of Sinkhorn loss could be particularly valuable for NCA training, which involves stochastic cell updates and can suffer from training instability. The faster convergence rate may offset any computational overhead from the OT computation itself.

---

## Deep Dive: NCA Training Bottlenecks

### The Quadratic Scaling Problem

NCAs face fundamental performance challenges:

> "Training time and memory requirements grow **quadratically** with grid size... confined them primarily to low-resolution outputs. Current NCAs are typically trained on grids, volumes, or meshes with at most 10^4 to 10^5 cells, which translates to spatial resolutions around **a few hundred pixels**."

This quadratic scaling applies to:
1. **Forward pass:** Each cell computes local updates based on neighbors
2. **Backpropagation through time:** Gradients must be propagated through T timesteps
3. **Perceptual loss computation:** Must evaluate loss at each (or select) timesteps

### Iteration Count Impact

NCAs are unrolled for multiple iterations during training:
- More iterations → better long-range structure
- More iterations → longer training time and more GPU memory (gradient accumulation)
- More iterations → harder optimization (vanishing/exploding gradients)

The perceptual loss network (VGG, SqueezeNet, etc.) is called multiple times per training batch:
- At least once per sample for computing loss
- Possibly multiple times if loss is computed at intermediate NCA steps
- The loss network dominates the computational cost for texture synthesis NCAs

### Memory Hierarchy

For a typical NCA training setup:
1. **NCA parameters:** Small (typically <1M parameters for the update rule)
2. **Grid state:** Medium-large (depends on resolution and channels)
3. **Gradient accumulation:** Large (storing activations for backprop through time)
4. **Perceptual network:** Large (VGG16: 2GB+, SqueezeNet: ~15MB)

---

## Empirical Benchmark Design

To properly evaluate VGG vs SqueezeNet vs OTT-Loss for NCA training, the following protocol is recommended:

### Controlled Variables
- **Same NCA architecture:** Identical update rule network (e.g., 16-channel state, 3x3 conv, 128 hidden units)
- **Same target texture:** Use a standard test texture (e.g., from DTD or Describable Textures Dataset)
- **Same resolution:** Train on 64×64 or 128×128 grids
- **Same iteration count:** Fixed T=64 or T=128 steps per training sample
- **Same batch size:** Largest batch size that fits in GPU memory for VGG (then use same for others)
- **Same optimization:** Adam optimizer with same learning rate schedule

### Loss Function Variants
1. **VGG16-Loss:** Standard multi-layer VGG perceptual loss (layers relu1_2, relu2_2, relu3_3, relu4_3)
2. **SqueezeNet-Loss:** Equivalent multi-layer loss using SqueezeNet features
3. **OTT-Loss (if feasible):** Sliced Wasserstein distance on VGG features or learned features

### Metrics to Measure

**Speed Metrics:**
- **Wall-clock time per epoch:** Total training time for 1000 samples
- **Iteration time (ms/iter):** Time per training batch
- **Forward pass time:** Isolated NCA forward pass (without loss)
- **Loss computation time:** Isolated perceptual loss computation
- **Total training time to convergence:** Time to reach target quality threshold

**Memory Metrics:**
- **Peak GPU memory:** Maximum VRAM usage during training
- **Steady-state memory:** Typical VRAM usage during training
- **Maximum achievable batch size:** For fixed grid resolution

**Quality Metrics:**
- **LPIPS:** Learned perceptual image patch similarity (using VGG to be fair)
- **FID:** Fréchet Inception Distance (if training generative model)
- **Human evaluation:** Side-by-side quality comparison
- **Training curve:** Loss vs epoch plots to compare convergence speed

### Expected Results

**Hypothesis:**
- **VGG16:** Slowest training, highest quality, highest memory
- **SqueezeNet:** 3-4x faster training, comparable quality (?), lower memory
- **OTT-Loss:** Potentially fastest if properly implemented, quality unclear for NCA application

**Key Questions:**
1. Does SqueezeNet maintain perceptual quality for NCA texture synthesis?
2. Is the speed gain worth any quality loss?
3. Can OT-based losses be made differentiable enough for NCA backprop through time?
4. What's the memory-speed-quality Pareto frontier?

---

## Connections to Existing Knowledge

### Distilled LPIPS
Related research (rq-1771607469255-distilled-lpips-empirical) explores distilling LPIPS into <10K parameter networks. This is complementary: instead of swapping VGG for SqueezeNet, train a tiny student network to mimic VGG's perceptual judgments. This could push even further toward real-time NCA training.

### Real-Time NCA Synthesis
Several papers address real-time *inference* for NCAs:
- **DyNCA (CVPR 2023):** Real-time dynamic texture synthesis using NCAs
- **Hierarchical NCAs:** Multi-scale approaches to reduce iteration count

But training speed remains a bottleneck. Faster perceptual losses directly enable:
- Faster experimentation and iteration
- Interactive training interfaces
- Training larger/higher-resolution NCAs

### Multi-Task and Pretrained NCAs
If NCA training becomes 3-4x faster with SqueezeNet, it becomes feasible to:
- Train multi-texture NCAs (single automaton for multiple textures)
- Pretrain NCAs on large texture datasets
- Fine-tune NCAs for specific textures with few-shot learning

---

## Follow-Up Questions

1. **Hybrid approaches:** Can we use SqueezeNet during early training (fast exploration) then switch to VGG for fine-tuning (quality)?

2. **Layer selection:** Which SqueezeNet layers best correspond to VGG's perceptual hierarchy? Does layer choice matter as much as architecture?

3. **Knowledge distillation:** Can we distill VGG's perceptual loss into SqueezeNet by training SqueezeNet to match VGG's feature distances?

4. **Mobile deployment:** If SqueezeNet-trained NCAs run well, can the entire pipeline (NCA + loss) run on mobile GPUs?

5. **Beyond texture synthesis:** Do these findings generalize to other NCA applications (morphogenesis, pattern formation, video textures)?

6. **Differentiable OT libraries:** Are there PyTorch/JAX libraries with fast, differentiable sliced Wasserstein losses ready for NCA training?

7. **VGG layer selection:** Research shows "selecting an earlier extraction layer can potentially reduce computation requirements by orders of magnitude" - can we achieve 80% of quality with 20% of compute?

8. **Multi-loss training:** Can simultaneous optimization using VGG + SqueezeNet + OTT losses improve both speed and quality through ensemble effects?

9. **Texture-specific optimization:** Do certain texture classes (geometric vs organic, high vs low frequency) benefit more from specific loss functions?

10. **Training schedule mixing:** Start with fast SqueezeNet exploration, transition to VGG refinement, then use OTT for final stabilization?

---

## Sources

- [Multi-texture synthesis through signal responsive neural cellular automata (Nature Scientific Reports, 2025)](https://www.nature.com/articles/s41598-025-23997-7)
- [Neural Cellular Automata: From Cells to Pixels (arXiv 2506.22899, 2026)](https://arxiv.org/abs/2506.22899)
- [Neural Cellular Automata: From Cells to Pixels (HTML version)](https://arxiv.org/html/2506.22899v2)
- [Optimal Textures: Fast and Robust Texture Synthesis through Optimal Transport (arXiv 2010.14702)](https://ar5iv.labs.arxiv.org/html/2010.14702)
- [A Systematic Performance Analysis of Deep Perceptual Loss Networks (arXiv 2302.04032)](https://arxiv.org/abs/2302.04032)
- [Perceptual Losses for Real-Time Style Transfer and Super-Resolution (ECCV 2016)](https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16.pdf)
- [SqueezeNet Architecture Guide (Medium)](https://medium.com/@avidrishik/squeezenets-architecture-compressed-neural-network-7741d24ca56f)
- [More than Real-Time FPS using SqueezeNet (PyTorch)](https://debuggercafe.com/more-than-real-time-fps-using-squeezenet-for-image-classification-in-pytorch/)
- [LPIPS GitHub Repository](https://github.com/richzhang/PerceptualSimilarity)
- [DyNCA: Real-Time Dynamic Texture Synthesis Using Neural Cellular Automata (CVPR 2023)](https://dynca.github.io/)
- [VGG-16 TensorFlow Benchmark (LeaderGPU)](https://www.leadergpu.com/tensorflow_vgg16_benchmark)
- [GPU vs CPU Inference Speed Comparison (GMI Cloud)](https://www.gmicloud.ai/blog/gpu-inference-vs-cpu-inference-speed-cost-and-scalability)

**Additional Sources (2026-02-24 Update):**
- [Optimal Transport-inspired Deep Learning Framework with Sinkhorn Loss (arXiv 2308.13840)](https://arxiv.org/abs/2308.13840)
- [Universal Neural Optimal Transport (OpenReview)](https://openreview.net/forum?id=t10fde8tQ7&noteId=EojJABLREV)
- [Riemannian Neural Optimal Transport (arXiv 2602.03566)](https://arxiv.org/html/2602.03566v1)
- [Improving Neural OT via Displacement Interpolation (OpenReview)](https://openreview.net/forum?id=CfZPzH7ftt)
- [On the Convergence Rate of Sinkhorn's Algorithm (Mathematics of Operations Research 2024)](https://pubsonline.informs.org/doi/abs/10.1287/moor.2024.0427)
- [Semi-Discrete Optimal Transport for Long-Tailed Classification (JCST 2025)](https://jcst.ict.ac.cn/article/doi/10.1007/s11390-023-3086-0)
- [Optimal Transport for Machine Learners (arXiv 2505.06589)](https://arxiv.org/abs/2505.06589)
- [VGG16 vs VGG19 Detailed Comparison (Medium)](https://medium.com/@sandhrabijoy/vgg16-vs-vgg19-a-detailed-comparison-of-the-popular-cnn-architectures-cae5ba404352)
- [SqueezeNet vs VGG CPU/GPU Performance (GitHub Issues)](https://github.com/forresti/SqueezeNet/issues/51)
- [SqueezeNet: Key to Edge Computing (Medium)](https://medium.com/sfu-cspmp/squeezenet-the-key-to-unlocking-the-potential-of-edge-computing-c8b224d839ba)
- [Neural Network Techniques for Image Style Transfer 2025 (ITM Conferences)](https://www.itm-conferences.org/articles/itmconf/pdf/2025/04/itmconf_iwadi2024_02027.pdf)
- [Scaling Painting Style Transfer (Computer Graphics Forum 2024)](https://onlinelibrary.wiley.com/doi/10.1111/cgf.15155)

**NCA GitHub Implementations:**
- [0xekez/neural-cellular-automata](https://github.com/0xekez/neural-cellular-automata) - VGG-16 tinygrad/PyTorch implementation for texture synthesis
- [MagnusPetersen/Neural-Cellular-Automata-Image-Manipulation](https://github.com/MagnusPetersen/Neural-Cellular-Automata-Image-Manipulation) - VGG19 style transfer for NCAs
- [IVRL/MeshNCA](https://github.com/IVRL/MeshNCA) - Official Mesh Neural Cellular Automata with VGG-based style loss
- [MonashDeepNeuron/Neural-Cellular-Automata](https://github.com/MonashDeepNeuron/Neural-Cellular-Automata) - General NCA training framework
- [MECLabTUDA/awesome-nca](https://github.com/MECLabTUDA/awesome-nca) - Curated list of NCA research and frameworks
- [dwoiwode/awesome-neural-cellular-automata](https://github.com/dwoiwode/awesome-neural-cellular-automata) - Paper and resource collection

---

## Tags

`nca`, `benchmarking`, `optimization`, `empirical-study`, `perceptual-loss`, `vgg`, `squeezenet`, `optimal-transport`, `training-speed`, `neural-cellular-automata`, `texture-synthesis`
