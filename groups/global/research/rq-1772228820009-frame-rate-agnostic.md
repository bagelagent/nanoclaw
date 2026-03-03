# Frame-Rate-Agnostic Video Representations Through Multi-Scale Temporal Decomposition

## Summary

Frame-rate-agnostic video representations aim to build models that can process, generate, or understand video at arbitrary temporal resolutions without being locked to a specific frame rate during training. This is achieved through multi-scale temporal decomposition strategies—hierarchical processing of video at coarse-to-fine temporal granularities—combined with continuous temporal representations that treat time as a continuous variable rather than discrete frame indices. The field has matured rapidly from 2022–2026, spanning video understanding, generation, super-resolution, and compression, with approaches ranging from implicit neural representations to hierarchical diffusion cascades and multi-scale attention transformers.

## Key Findings

### 1. The Frame Rate Problem in Video AI

Traditional video models are brittle to frame rate changes. Most architectures are trained on fixed temporal sampling rates (e.g., 8 frames at 30fps), and performance degrades substantially when deployed at different frame rates. Research shows:

- **Action recognition models** are not robust to frame rate variations—state-of-the-art approaches experience significant accuracy drops when the input frame rate changes from the training distribution.
- **Frame sampling strategy matters**: A 2025 benchmark of small vision-language models found that uniform-FPS sampling benefits most from larger frame counts (peaking ~256 frames), but excessive frames introduce redundancy. The optimal strategy is model-specific and task-dependent.
- **Variable frame rate content** is increasingly common (smartphone video, screen recordings, surveillance), making FPS invariance a practical necessity.

### 2. Foundational Architectures for Multi-Scale Temporal Processing

#### SlowFast Networks (ICCV 2019)
The seminal dual-pathway architecture from Facebook AI Research processes video at two temporal resolutions simultaneously:
- **Slow pathway**: Low frame rate (e.g., 2fps), high channel capacity → captures spatial semantics
- **Fast pathway**: High frame rate (e.g., 16fps), lightweight (1/8 channels) → captures fine temporal dynamics

Inspired by the primate visual system (P-cells for spatial detail, M-cells for motion), SlowFast demonstrated that asymmetric temporal processing is both more efficient and more accurate than single-scale approaches. The Fast pathway uses only 20% of total FLOPs while providing crucial temporal information.

#### Multiscale Vision Transformers – MViT (ICCV 2021) and MViTv2 (CVPR 2022)
MViT builds a spatiotemporal feature hierarchy within the transformer backbone:
- **Progressive resolution**: Fine spacetime resolution in early layers (high spatial/temporal detail) → coarse spacetime in later layers (high channel complexity)
- **Pooling Attention (MHPA)**: Applies pooling kernels to Q, K, V tensors along temporal and spatial dimensions, creating the multi-scale effect within attention
- **No external pretraining required**: Unlike ViT-based video models, MViT trains from scratch

MViTv2 achieved 86.1% on Kinetics-400, demonstrating that multi-scale temporal modeling provides strong temporal understanding without the spatial bias pitfall of prior methods.

#### MeMViT (CVPR 2022)
Extended MViT with memory augmentation for long-term video recognition, treating long videos as sequences of short clips and maintaining temporal memory across iterations.

### 3. Implicit Neural Representations for Continuous Time

A breakthrough approach to frame-rate agnosticism comes from **Implicit Neural Representations (INRs)**, which model video as a continuous function mapping (x, y, t) coordinates to RGB values.

#### VideoINR (CVPR 2022)
The foundational work in continuous space-time video representation:
- Maps any 3D space-time coordinate to an RGB value
- Enables decoding at **arbitrary spatial resolution and frame rate**
- Architecture: Encoder extracts features from consecutive frames → SpatialINR decodes spatial coordinates → TemporalINR computes motion flow → warped features decoded to output
- Achieves competitive performance on standard scales and **significantly outperforms** prior works on out-of-training-distribution scales

#### NeRV and FANeRV
Neural Representations for Videos (NeRV) represents video as a function from frame index to pixel grid. FANeRV addresses spectral bias by using Discrete Wavelet Transform for explicit frequency separation, enabling better high-frequency temporal detail capture.

#### TeNeRV (2026) – Hierarchical Temporal Neural Representation
Addresses the limitation of INR-based video compression by integrating short- and long-term temporal dependencies, enabling motion-aware compression that better handles dynamic content than pure memorization approaches.

#### V3: Continuous Space-Time Super-Resolution with Fourier Fields (2025)
Combines Video Fourier Fields with neural encoders for unified spatio-temporal super-resolution at arbitrary scaling factors, outperforming competing methods by >1 dB PSNR.

### 4. Hierarchical Frame-Rate Generation

#### TempoMaster (CVPR 2025)
Formulates video generation as **next-frame-rate prediction**:
1. Generate low-frame-rate clip as a coarse blueprint
2. Progressively increase frame rate to refine details
3. Bidirectional attention within each frame-rate level + autoregression across frame rates
4. Continuous positional encoding prevents overfitting to specific temporal indices

Training requires ~1500 H100 GPU-days across two stages (single-resolution → multi-resolution). The hierarchical coarse-to-fine approach enables both long-range temporal coherence and fine-grained motion detail.

#### Imagen Video Temporal Cascade
Google's approach uses interleaved spatial and temporal super-resolution diffusion models—3 TSR stages progressively increase frame rate by inserting intermediate frames between existing ones.

#### SemFi: Semantic Frame Interpolation (2025)
Uses **Mixture-of-LoRA** for multi-frame-rate interpolation:
- Base adapter captures interpolation-invariant features (motion consistency, semantic preservation across scales)
- Expert LoRAs specialize for different discrete frame counts
- Graceful generalization to unseen frame counts via automatic expert selection

### 5. Variable Frame Rate Tokenization and Understanding

#### VFRTok (2025) – Variable Frame Rate Video Tokenizer
Proposes the **Duration-Proportional Information Assumption** for video tokenization:
- Transformer-based tokenizer handles variable input/output frame rates
- Successfully interpolates 12fps video to 30, 60, and 120fps
- Treats frame rate as a continuous variable rather than a fixed architectural constraint

#### F-16: High-Frame-Rate Video LLM (2025)
First multimodal LLM designed for 16fps video understanding:
- Compresses visual tokens within 1-second clips
- **Variable-frame-rate decoding**: trained at 16fps but deployable at lower FPS for efficiency
- Lower FPS for general comprehension, higher FPS for fine-grained temporal analysis

### 6. Training Strategies for Frame Rate Robustness

| Strategy | Example Work | Mechanism |
|----------|-------------|-----------|
| Multi-temporal-resolution training | TempoMaster | Two-stage training with varying FPS |
| Frame dropout augmentation | Depth Any Video | Randomly drop frames to simulate variable FPS |
| Continuous positional encoding | TempoMaster | Prevents overfitting to fixed temporal indices |
| Mixture-of-LoRA | SemFi | Frame-count-specific adapters with shared base |
| Adversarial augmentation | Action recognition | Frame rate conversion as data augmentation |
| Multi-clip embedding | TWLV-I | Divide video into M clips × N frames each |
| Content-aware sampling | STEC metric | Sample proportional to information density |

### 7. Wavelet and Multi-Resolution Approaches

The classical multi-resolution analysis from wavelet theory has been integrated into deep video architectures:
- **3D wavelet decomposition** separates temporal subbands (TLL, TLH, THL, THH) which are further decomposed spatially
- **Multi-level Wavelet CNNs (MWCNN)** treat wavelet packet transform filters as learnable convolutions
- **Fully learnable wavelets** (PNAS) allow automatic extraction of meaningful multi-scale representations, learning optimal high/low-pass filters at each decomposition level
- **FANeRV** uses DWT for explicit frequency separation in neural video representations

## Deep Dive: Why Multi-Scale Temporal Decomposition Works

The effectiveness of multi-scale temporal decomposition stems from a fundamental property of natural video: **temporal information is distributed across multiple frequency bands**.

**Low-frequency temporal information** captures:
- Scene composition and layout
- Slow camera movements
- Background consistency
- Semantic content (what is happening)

**High-frequency temporal information** captures:
- Rapid motions and actions
- Flickering and transients
- Fine-grained temporal detail
- Motion blur and interlacing artifacts

By processing these bands separately (either explicitly via wavelet decomposition or implicitly via multi-scale architectures like SlowFast), models can:

1. **Allocate compute efficiently**: Low-frequency content needs fewer samples; high-frequency content needs dense sampling but less channel capacity
2. **Generalize across frame rates**: A model that separately understands slow and fast dynamics can recombine them at any target frame rate
3. **Handle long videos**: Coarse temporal resolution enables long-range context while fine resolution captures local dynamics (the TempoMaster principle)

This mirrors how the human visual system processes temporal information—with separate neural pathways for different temporal frequencies (magnocellular vs. parvocellular).

## Connections to Existing Research

### NCA and Frame-Rate Agnosticism
This research directly connects to the NCA texture synthesis queue items:
- **NoiseNCA** demonstrates that NCAs can learn continuous dynamics with frame-rate-independent behavior when initialized with uniform noise, enabling continuous control over pattern formation speed
- **DyNCA** tackles real-time dynamic texture synthesis with post-training control over motion speed—essentially frame-rate-agnostic texture generation
- **TeNCA** applies temporal NCA specifically for sequential consistency, relevant to temporal stability in NCA pipelines
- For NCA-based texture synthesis, multi-scale temporal decomposition could enable training at one frame rate and deploying at another—important for WebGL/WebGPU real-time applications

### Perceptual Loss and Temporal Metrics
The temporal consistency challenge connects to several perceptual loss research items:
- Streaming video perceptual metrics need to handle varying frame rates
- Multi-scale temporal decomposition could improve perceptual loss computation by evaluating at multiple temporal scales simultaneously

### Video Diffusion Models
Connects to existing research on:
- Lipschitz constraints in video diffusion (temporal smoothness across scales)
- ODE solvers for video flow matching (continuous temporal trajectories)
- Adaptive step-count models (which implicitly handle multi-scale temporal generation)

## Follow-up Questions

1. **NCA continuous-time dynamics**: Can NCAs be reformulated as neural ODEs to achieve true continuous-time evolution, enabling frame-rate-agnostic texture animation?
2. **Wavelet-guided NCA**: Would a wavelet temporal decomposition of NCA update rules improve temporal coherence while enabling multi-frame-rate synthesis?
3. **Perceptual metrics for variable FPS**: How should perceptual loss functions (LPIPS, etc.) be adapted when comparing videos at different frame rates—is temporal warping needed?
4. **Efficient multi-scale inference**: What are the Pareto frontiers of compute vs. temporal quality for hierarchical vs. single-pass frame-rate-agnostic approaches?

## Sources

- SlowFast Networks: https://arxiv.org/abs/1812.03982
- MViT: https://arxiv.org/abs/2104.11227
- MViTv2: https://arxiv.org/abs/2112.01526
- VideoINR: https://arxiv.org/abs/2206.04647 / https://zeyuan-chen.com/VideoINR/
- TempoMaster: https://arxiv.org/abs/2511.12578
- FAPS (Frame Rate Agnostic MOT): https://link.springer.com/article/10.1007/s11263-023-01943-2
- NoiseNCA: https://noisenca.github.io/
- DyNCA (CVPR 2023): https://openaccess.thecvf.com/content/CVPR2023/papers/Pajouheshgar_DyNCA_Real-Time_Dynamic_Texture_Synthesis_Using_Neural_Cellular_Automata_CVPR_2023_paper.pdf
- TeNCA (MICCAI 2025): https://papers.miccai.org/miccai-2025/paper/4096_paper.pdf
- NCA: From Cells to Pixels: https://cells2pixels.github.io/
- F-16 Video LLM: https://arxiv.org/html/2503.13956v1
- VFRTok: https://arxiv.org/html/2505.12053v2
- AceVFI Survey: https://arxiv.org/abs/2506.01061
- EDEN VFI: https://arxiv.org/abs/2503.15831
- V3 Fourier Fields: https://arxiv.org/html/2509.26325v1
- Frame Sampling Strategies Benchmark: https://arxiv.org/html/2509.14769v1
- TWLV-I Foundation Model: https://www.twelvelabs.io/blog/twlv-i
- TSMS-SAM2: https://arxiv.org/pdf/2508.05829
- STEC Metric: https://arxiv.org/html/2601.13974
- Long-Short Temporal Contrastive Learning: https://arxiv.org/pdf/2106.09212v2
- Multi-Scale Temporal Difference Transformer: https://arxiv.org/html/2406.16111v1
- Temporal-Invariant Video Contrastive Learning: https://www.sciencedirect.com/science/article/abs/pii/S0031320325009173
- FANeRV: https://www.sciencedirect.com/science/article/abs/pii/S095741742503550X
- Multi-level Wavelet CNN: https://arxiv.org/pdf/1907.03128
- Wavelet-integrated DNNs Review: https://www.sciencedirect.com/science/article/abs/pii/S0925231225023203
- Depth Any Video: https://arxiv.org/html/2410.10815
- SemFi: https://arxiv.org/html/2507.05173v1
- MeMViT: https://openaccess.thecvf.com/content/CVPR2022/papers/Wu_MeMViT_Memory-Augmented_Multiscale_Vision_Transformer_for_Efficient_Long-Term_Video_Recognition_CVPR_2022_paper.pdf
