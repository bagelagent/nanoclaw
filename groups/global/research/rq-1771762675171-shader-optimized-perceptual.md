# Perceptual Metrics Optimized for Real-Time GPU Shader Evaluation

**Research Topic:** Shader-optimized perceptual metrics for real-time GPU evaluation - DISTS, SSIM+, custom learned metrics, and alternatives to LPIPS

**Research Date:** February 22, 2026
**Priority:** 8/10
**Source:** Research follow-up from shader texture synthesis study

---

## Summary

Real-time perceptual metrics for GPU shader evaluation represent a critical gap between training-time quality assessment and runtime performance constraints. While LPIPS and VGG-based perceptual losses are standard for training neural texture synthesis models, their computational demands (50-100+ GFLOPS) make 60 FPS deployment challenging. This research identifies four promising categories of shader-optimized perceptual metrics: (1) **traditional metrics with GPU acceleration** (SSIM, MS-SSIM, PSNR) achieving 30-80x speedups on modern hardware; (2) **modern perceptual metrics designed for efficiency** (SSIMULACRA2, MILO, Butteraugli) balancing quality and speed; (3) **GPU-accelerated neural metrics** (VMAF-CUDA, distilled lightweight networks) providing 4-37x speedups; and (4) **ultra-lightweight approximations** (perceptual hashing, simplified CNN proxies) trading some accuracy for real-time performance. The landscape in 2026 shows convergence toward CUDA/compute shader implementations achieving real-time (60+ FPS) perceptual assessment at 1080p on mid-range GPUs, with WebGPU emerging as the browser-based deployment path.

---

## Key Findings

### 1. Traditional Metrics with GPU Acceleration

**SSIM (Structural Similarity Index)**

SSIM has become the baseline for GPU-accelerated perceptual metrics due to its relative computational simplicity and strong correlation with human perception.

**Performance Characteristics:**
- **CUDA Implementation:** ~30x speedup on NVIDIA GTX275, ~80x on C2050 over Intel single-core
- **OpenCV GPU Module:** Nearly 100% performance increase (2x) vs CPU for SSIM
- **Computational Complexity:** Moderate - requires Gaussian filtering and local statistics computation
- **Real-time Capability:** Easily achieves 60+ FPS at 1080p on modern GPUs

**Architecture:**
```
SSIM(x, y) = [l(x,y)]^α · [c(x,y)]^β · [s(x,y)]^γ

where:
  l(x,y) = (2μ_x μ_y + C1) / (μ_x² + μ_y² + C1)  [luminance]
  c(x,y) = (2σ_x σ_y + C2) / (σ_x² + σ_y² + C2)  [contrast]
  s(x,y) = (σ_xy + C3) / (σ_x σ_y + C3)           [structure]
```

**GPU Optimization Strategies:**
- Gaussian kernels are separable → two 1D convolutions instead of 2D
- Integral images for O(1) local statistics computation
- Strided computation for additional speedup with minimal quality loss
- Half-float precision (FP16) reduces memory bandwidth by 50%

**Shader Implementations:**
- **GLSL Examples:** SSimDownscaler.glsl, SSimSuperRes.glsl for mpv video player
- **OpenGL ES:** ARM SDK provides compute shader tutorials for mobile
- **WebGL2:** Feasible but requires manual implementation (no standard library)
- **MATLAB:** Built-in GPU array support for SSIM function

**MS-SSIM (Multi-Scale SSIM)**

MS-SSIM extends SSIM across multiple scales (typically 5 levels), providing better perceptual correlation at the cost of increased computation.

**Implementation Details:**
- PyTorch: `pytorch-msssim` achieves fast, differentiable computation
- TensorFlow: `tf.image.ssim_multiscale` with GPU support
- C++: Single and multi-scale implementations available
- **Performance:** 3-5x slower than single-scale SSIM due to multi-resolution processing

**Practical Consideration:**
MS-SSIM is **not recommended for Super Resolution tasks** according to practitioners working with these metrics, as it can score destructive filtering higher than perceptually-preserving methods.

**PSNR (Peak Signal-to-Noise Ratio)**

While PSNR is primarily pixel-based and not perceptual, it's included due to its ubiquity and extreme speed.

**Advantages:**
- Trivial to implement in shaders (simple MSE + log10)
- Runs at hundreds to thousands of FPS
- Useful as a quick sanity check alongside perceptual metrics

**Limitations:**
- Poor correlation with human perception
- Absolute errors don't account for structural similarity or masking effects
- Not suitable as sole metric for quality assessment

**Formula:**
```glsl
float MSE = dot(diff, diff) / (width * height);
float PSNR = 20.0 * log2(255.0) / log2(sqrt(MSE));
```

### 2. Modern Perceptual Metrics (2022-2026)

**SSIMULACRA2 (Cloudinary, 2022-2023)**

SSIMULACRA2 represents a significant evolution of SSIM-based metrics, combining multi-scale analysis with perceptually-relevant color space transformations.

**Architecture:**
- Based on MS-SSIM computed in **XYB color space** (perceptually uniform)
- Two additional asymmetric error maps beyond SSIM
- Aggregation using two different norms
- Weights tuned on large subjective quality datasets (CID22, TID2013, Kadid10k, KonFiG-IQA)

**Training Data:**
- Images compressed with JPEG, JPEG 2000, JPEG XL, WebP, AVIF, HEIC
- Various artificial distortions
- Validated against human subjective scores

**Scoring System:**
```
-inf...100 scale:
  10  = very low quality (MOS ~1, "bad")
  30  = low quality (MOS ~2, "poor")
  50  = medium quality (MOS ~3, "fair")
  70  = high quality (MOS ~4, "good")
  90+ = excellent quality (MOS ~5, "excellent")
```

**GPU Performance (TurboMetrics CUDA Implementation, Nov 2024):**

| Hardware | 1080p H.264 Performance | Speedup vs CPU |
|----------|------------------------|----------------|
| GTX 1060 | 46.70 FPS (stride 1) | 14x vs ssimulacra2_rs |
| RTX 4060 | 85.64 FPS (stride 1) | 25.7x vs ssimulacra2_rs |
| RTX 4060 | 245.73 FPS (stride 3) | Better than real-time |

**Implementation Optimizations:**
- Explicit Fused Multiply-Add (FMA) operations
- Rearranged calculation sequences for parallel processing
- Approximated math functions (`powf`) using GPU native instructions
- YUV-to-linear RGB conversion optimized for CUDA

**Bottlenecks:**
- Video decoding becomes primary bottleneck at high computation strides
- GTX 1060 decode limit: ~696 FPS for 1080p H.264
- RTX 4060 decode limit: ~883 FPS H.264, ~1005 FPS AV1

**Accuracy Considerations:**
- Floating-point precision differences accumulate through computation
- GPU scores show statistically different results vs CPU reference
- Preliminary analysis suggests potential systematic under-scoring on GPU
- **Recommendation:** Use stride 3-5 frames for practical assessment

**Status:** Currently the most reputable visual quality metric according to correlation with subjective results, gaining popularity for video assessment due to reliability.

**MILO (Metric for Image and Latent-space Optimization, Sept 2025)**

MILO represents the latest generation of learned lightweight perceptual metrics.

**Key Characteristics:**
- **Lightweight, multiscale** perceptual metric for full-reference IQA
- Outperforms existing metrics across standard FR-IQA benchmarks
- **Fast inference suitable for real-time applications**
- Compact architecture with fewer parameters than LPIPS

**Training Methodology:**
- Pseudo-MOS (Mean Opinion Score) supervision
- Reproducible distortions applied to diverse images
- Scored via ensemble of recent quality metrics
- Accounts for visual masking effects

**Dual-Purpose Design:**
1. **Quality Prediction:** State-of-the-art image quality metric
2. **Perceptual Loss:** Functions in both image and latent domains
   - Spatial masking applied to latent representations
   - Compatible with VAE encoders (e.g., Stable Diffusion)

**Advantages:**
- More efficient than LPIPS while maintaining quality
- Designed specifically for optimization tasks
- Balances perceptual accuracy with computational efficiency

**Status:** Emerging metric (2025), not yet widely adopted but showing promise for real-time applications.

**Butteraugli (Google)**

Butteraugli is a perceptual similarity metric from Google designed to estimate how noticeable differences between images will be to the human eye.

**Architecture:**
- Models aspects of human vision:
  - Color sensitivity (opponent color channels)
  - Spatial masking (texture-dependent sensitivity)
  - Contrast perception (Weber-Fechner law)
- Outputs single "distance" score
- Per-pixel/per-region maps showing most objectionable artifacts

**Implementations:**
- **C++ Reference:** Google/butteraugli (official)
- **Rust:** Pure Rust implementation from libjxl
  - Validated against C++ with <0.0003% difference on real photos
- **GPU-Accelerated:** HIP/CUDA implementation (vship)

**Use Cases:**
- Quality metric for lossy image/video compression
- Used by **Guetzli** JPEG encoder as optimization feedback
- Can define quality level settings for JPEG compressors

**Advantages:**
- Specifically designed for compression artifact detection
- Highlights perceptually-important differences
- GPU implementations available for speed

**Limitations:**
- Less widely adopted than LPIPS/SSIM
- Primarily focused on compression artifacts rather than general distortions

**DISTS (Deep Image Structure and Texture Similarity)**

DISTS is an adaptation of LPIPS that emphasizes texture similarity.

**Architecture:**
- VGG-based convolutional neural network variant
- Combines structure and texture similarity measurements
- Feature maps from multiple layers
- Weighted aggregation tuned for texture perception

**Recent Research (2024-2025):**
- Analyzed in January 2025 study on MR image-to-image translation
- Examined sensitivity to various distortions and MR artifacts
- Recommendations provided for effective usage in translation models

**Comparison to LPIPS:**
- LPIPS: Focuses on general perceptual similarity
- DISTS: More emphasis on texture similarity
- Both use deep feature activation distances

**Availability:**
- GitHub: `dingkeyan93/DISTS` (official implementation)
- Accessible for research and practice

**Status:** Considered alongside LPIPS as a modern perceptual metric, particularly valuable when texture preservation is critical.

### 3. GPU-Accelerated Neural Metrics

**VMAF-CUDA (Netflix/NVIDIA Collaboration, 2023)**

VMAF (Video Multi-Method Assessment Fusion) is Netflix's perceptual video quality metric, now with official CUDA acceleration.

**Performance Benchmarks (1080p/4K):**

| Metric | Speedup | Latency Reduction |
|--------|---------|-------------------|
| 4K throughput | 4.4x | 37x lower latency |
| 4K single-frame | 36.9x faster | vs Dual Xeon 8480 |
| 1080p single-frame | 26.1x faster | vs Dual Xeon 8480 |

**Integration:**
- Officially part of VMAF 3.0 and FFmpeg v6.1
- GPU frames support for hardware-accelerated decoding
- ~6x speedup at same power draw vs Dual Xeon 8480

**Real-Time Applications:**
- Quality monitoring during encoding/transcoding
- V-Nova exploring benefits for LCEVC (MPEG-5 Part 2) encoding
- Real-time decision-making within encoding process

**Implementation:**
- CUDA-based GPU acceleration
- Compute shader approach for parallel processing
- Optimized for NVIDIA hardware

**VMAF-E (MainConcept Alternative)**

VMAF-E is a lightweight fast VMAF estimate:
- **10x faster** than traditional VMAF
- Immediate scoring for live and on-demand content
- Uses encoder-derived data and simple image features
- Ideal when GPU acceleration unavailable or for live streams

**PyTorch VMAF-torch (Unofficial)**

- Speed superior to LPIPS
- Comparable to MS-SSIM
- Unofficial PyTorch implementation based on official C code

**OpenCL Acceleration:**

MSU Video Quality Measurement Tool offers:
- Up to 40 FPS measurement of VMAF and NIQE on FullHD
- GPU support: up to 11.7x faster calculation
- CUDA & OpenCL support

**Distilled Lightweight Perceptual Networks**

**Architecture Candidates:**
- **SqueezeNet:** 2.8 MB, performs well for perceptual loss with low computational needs
- **MobileNetV3 Small:** Best balance between accuracy and efficiency (2025 benchmarks)
- **EfficientNetV2-S:** Highest accuracy among lightweight models
- **MobileNetV2:** Favorable balance of accuracy and latency
- **AlexNet:** 9.1 MB, fastest LPIPS variant for forward scoring

**Knowledge Distillation Approaches:**

1. **Feature-based Distillation:**
   - Transfer knowledge from intermediate layers or attention maps
   - Focus on local perceptual ability and middle layer expressiveness
   - More complex than logit-based but can outperform

2. **Dual Supervision Pathways:**
   - Teacher network: Full VGG/LPIPS
   - Student network: 3-5 layer lightweight CNN
   - KL divergence loss for probability matching
   - Quantization-aware distillation for further efficiency

**Performance Targets:**
- 3-5 layer networks approximating LPIPS
- Target: >0.8 correlation with full LPIPS
- Inference speed: 30-60 FPS at 1080p on mid-range GPUs
- Model size: <10 MB weights

**Recent Developments (2025):**
- Compression techniques: quantization, pruning, distillation
- MobileNetV3 excels in efficiency balance
- SqueezeNet best for severely constrained devices
- EfficientNetV2 maintains high accuracy despite compression

### 4. Ultra-Lightweight Approximations

**Perceptual Hashing (pHash, dHash)**

Perceptual hashes create fingerprints of images based on content features rather than exact pixels.

**Common Algorithms:**

| Algorithm | Method | Speed | Robustness |
|-----------|--------|-------|------------|
| pHash | DCT-based frequency analysis | Moderate | High - handles transformations |
| dHash | Adjacent pixel differences | **Fastest** | Moderate - gradient direction |
| aHash | Average pixel value | Very fast | Low - basic similarity |
| wHash | Wavelet-based | Moderate | Moderate-High |

**Characteristics:**
- Unlike cryptographic hashes, perceptual hashes are "close" for similar images
- Hamming distance measures similarity between hashes
- Extremely lightweight: single pass over image

**Performance:**
- **dHash:** Simplest and fastest, focusing on pixel differences
  - Suitable for real-time applications
  - Lower robustness to transformations
- **pHash:** More robust against significant alterations
  - Better for scenarios with various transformations
  - Slower than dHash but still fast

**GPU/Real-Time Considerations:**
- Most implementations are CPU-based (Python, Java, C libraries)
- Algorithms simple enough for real-time on modern hardware without GPU
- pHash library has multithreading support
- No specific GPU shader implementations found in research

**Limitations:**
- Lower correlation with human perception than learned metrics
- Binary similarity measure, not continuous quality score
- Best suited for duplicate detection rather than quality assessment

**Simplified CNN Proxies**

**Concept:**
Train 3-5 layer convolutional networks to approximate LPIPS scores without full VGG computation.

**Architecture Example:**
```
Input (256×256×3)
  ↓
Conv2D (3×3, 32 channels) + ReLU
  ↓
MaxPool (2×2)
  ↓
Conv2D (3×3, 64 channels) + ReLU
  ↓
MaxPool (2×2)
  ↓
Conv2D (3×3, 128 channels) + ReLU
  ↓
Global Average Pooling
  ↓
Dense (128 → 1) → Perceptual Distance Score
```

**Training:**
- Dataset: Image pairs with ground-truth LPIPS scores
- Loss: MSE or KL divergence between proxy and LPIPS
- Data augmentation: Various distortions, compressions, transformations

**Expected Performance:**
- Inference: 1-2ms per frame at 1080p (30-60 FPS budget)
- Correlation: Target >0.8 with full LPIPS
- Model size: 500KB - 5MB

**Deployment:**
- WebGL2: Feasible with manual CNN implementation in shaders
- WebGPU: More natural with compute shaders
- CUDA: Straightforward with cuDNN or custom kernels

**Challenges:**
- Maintaining perceptual correlation with simplified architecture
- Finding optimal layer count and channel widths
- Generalization across diverse image types and distortions

### 5. WebGL/WebGPU Implementation Strategies

**WebGL2 Limitations for Neural Metrics:**

1. **Texture Format Constraints:**
   - Float textures not filterable by default (nearest interpolation only)
   - Float textures not color-renderable without extensions
   - Extension support varies across devices
   - `OES_texture_float_linear` needed for smooth filtering

2. **Architecture Mismatch:**
   - CNNs prefer buffer operations
   - WebGL2 designed for texture-based graphics
   - Texture packing overhead for multi-channel feature maps
   - Limited texture units (16 minimum) constrains simultaneous access

3. **State Management:**
   - Shader compilation and switching overhead
   - Verbose boilerplate code
   - Ping-pong rendering for iterative processes

**WebGL2 Optimization Techniques:**

1. **Separable Filters:**
   - 2D convolutions → two 1D passes
   - Dramatic performance improvement for Gaussian kernels
   - Essential for SSIM implementations

2. **Half-Float Precision:**
   - `RGBA16F` textures when extensions available
   - Fallback to `RGBA8` with careful normalization
   - 50% memory bandwidth reduction vs FP32

3. **Minimal Wrappers:**
   - **SwissGL:** <1000 lines, single `glsl()` function
   - Eliminates verbose WebGL boilerplate
   - Successfully used for NCA implementations
   - Particle simulations run hundreds of steps/sec on mobile

4. **Fragment vs Compute Shader Hybrid (WebGL2 compute extension):**
   - Fragment shaders: Advantageous for small networks
   - Compute shaders: Better for larger CNNs when available
   - Layer-level shader selection for optimal performance

**WebGPU Advantages:**

1. **Compute-First Design:**
   - Native compute shader support
   - Tensors as buffers, not awkward texture packing
   - Better memory access patterns for ML workloads

2. **Performance Gains:**
   - TensorFlow.js stable diffusion: **3x speedup** WebGL→WebGPU
   - Top AI frameworks support: TensorFlow.js, ONNX Runtime Web, TVM
   - Shared memory and barriers for group-local optimization

3. **Texture Ping-Ponging:**
   - Storage textures as writable buffers
   - Write directly to any pixel within texture
   - Storage buffers allow in-place updates (unlike WebGL)

4. **Workgroup Optimization:**
   - General advice: workgroup size of 64
   - Most GPUs efficiently run 64 operations in lockstep
   - Processing at fraction of canvas size (e.g., 1/4) for more iterations

5. **Real-World Examples:**
   - Reaction-diffusion patterns in compute shaders
   - Neural network inference frameworks
   - Particle systems and flocking simulations

**Browser Support (2026):**
- WebGPU: Chrome, Edge, Firefox (stable)
- WebGL2: Universal support
- SIMD: All major browsers (Firefox 89+, Chrome 114+ with relaxed SIMD)

**WebAssembly SIMD Alternative:**

For perceptual metrics that don't require GPU:

**Performance:**
- SIMD instructions perform same operation on multiple data elements
- Image processing, audio manipulation ideal use cases
- Brings desktop-level performance to browsers

**Benefits:**
- wasm-vips: 5.9x faster than jimp for JPEG
- Relaxed SIMD: 1.5-3x speedup with dot product and FMA instructions
- Shipped in Chrome 114

**Implementation:**
- Compile with emscripten: `-msimd128` flag
- Include `wasm_simd128.h` for intrinsics
- Compiler generates WASM SIMD instructions

**Limitations:**
- CPU-bound, not GPU-accelerated
- Best for operations not requiring thousands of parallel threads
- Complements GPU approaches rather than replacing them

### 6. Practical Implementation Matrix

| Metric | Real-Time (60 FPS) | GPU Accel | Perceptual Quality | WebGL2 Feasible | Best Use Case |
|--------|-------------------|-----------|-------------------|----------------|---------------|
| **PSNR** | ✅ 500+ FPS | ⚠️ Unnecessary | ❌ Poor | ✅ Trivial | Sanity check only |
| **SSIM** | ✅ 100+ FPS | ✅ 30-80x | ⭐⭐⭐ Good | ✅ Moderate effort | Real-time quality baseline |
| **MS-SSIM** | ✅ 60+ FPS | ✅ 20-50x | ⭐⭐⭐⭐ Very Good | ⚠️ Complex | Multi-scale quality |
| **SSIMULACRA2** | ✅ 85 FPS (RTX 4060) | ✅ 25x | ⭐⭐⭐⭐⭐ Excellent | ❌ CUDA only | Best subjective correlation |
| **MILO** | ✅ Claimed real-time | ⚠️ Unspecified | ⭐⭐⭐⭐⭐ Excellent | ⚠️ Details needed | Emerging option |
| **Butteraugli** | ✅ With GPU impl | ✅ HIP/CUDA | ⭐⭐⭐⭐ Very Good | ❌ Complex | Compression artifacts |
| **DISTS** | ⚠️ Moderate | ⚠️ Limited | ⭐⭐⭐⭐ Very Good | ❌ Deep network | Texture similarity |
| **LPIPS** | ❌ 10-30 FPS | ⚠️ Limited | ⭐⭐⭐⭐⭐ Excellent | ❌ Too heavy | Training only |
| **VMAF-CUDA** | ✅ 36x faster | ✅✅ Official | ⭐⭐⭐⭐⭐ Excellent | ❌ CUDA only | Video quality (production) |
| **Distilled CNN** | ✅ 30-60 FPS | ✅ Trainable | ⭐⭐⭐⭐ Very Good (if >0.8 correlation) | ✅ With effort | Custom lightweight proxy |
| **pHash/dHash** | ✅ 1000+ FPS | ⚠️ Unnecessary | ⭐⭐ Fair | ✅ Simple | Duplicate detection |

**Legend:**
- ✅ Yes/Good
- ⚠️ Moderate/Limited
- ❌ No/Poor
- ⭐ Quality rating (5 stars = excellent)

### 7. Recommended Strategies by Use Case

**Use Case 1: Real-Time Browser Texture Synthesis (WebGL2)**

**Goal:** Perceptual guidance during NCA evolution at 60 FPS

**Recommended Approach:**
1. **Primary:** Implement SSIM in fragment shaders
   - Use separable Gaussian kernels
   - Half-float textures (RGBA16F)
   - Target: <2ms per frame
2. **Fallback:** Simplified 3-layer CNN proxy trained to approximate LPIPS
   - Train offline with full LPIPS as teacher
   - Deploy lightweight student network
   - Target: <5ms per frame
3. **Quality Check:** Occasional full LPIPS on server/WebAssembly SIMD
   - Validate every 60-300 frames
   - Async computation doesn't block rendering

**Expected Performance:** 60 FPS with SSIM, 30-45 FPS with CNN proxy

**Use Case 2: Video Quality Assessment (CUDA/Production)**

**Goal:** Accurate perceptual quality for video encoding pipeline

**Recommended Approach:**
1. **First Choice:** VMAF-CUDA (official, proven, 4-37x speedup)
   - Integrated with FFmpeg v6.1
   - Real-time monitoring during transcoding
2. **Alternative:** SSIMULACRA2 with TurboMetrics
   - Best subjective correlation
   - 85+ FPS at 1080p on RTX 4060
   - Use stride 3-5 for balance
3. **Fast Estimate:** VMAF-E for live streaming
   - 10x faster than VMAF
   - Encoder-derived features

**Expected Performance:** Real-time processing for 4K with modern GPUs

**Use Case 3: Game Rendering Quality Monitoring**

**Goal:** Detect perceptible rendering artifacts during gameplay

**Recommended Approach:**
1. **Runtime:** MS-SSIM in compute shaders
   - Multi-scale captures artifacts at different frequencies
   - Compare rendered frame to reference/previous frame
   - Trigger quality adjustments if score drops
2. **Development:** Butteraugli for compression artifact analysis
   - Evaluate texture compression impact
   - Optimize memory vs quality tradeoffs
3. **Optimization:** PSNR for rapid iteration
   - Quick sanity checks during asset pipeline
   - Supplement with MS-SSIM for final validation

**Expected Performance:** 60-120 FPS depending on resolution

**Use Case 4: Research / Experimentation (WebGPU/Cutting Edge)**

**Goal:** Explore novel perceptual guidance architectures

**Recommended Approach:**
1. **Emerging Metric:** MILO for real-time perceptual optimization
   - Latest (2025) lightweight learned metric
   - Designed for both quality assessment and loss function
   - Works in latent spaces (VAE compatibility)
2. **Hybrid Architecture:** Combine multiple metrics
   - SSIM for structure (fast baseline)
   - Lightweight CNN for texture (learned component)
   - Perceptual hash for coarse similarity (ultra-fast)
3. **WebGPU Implementation:** Full CNN in compute shaders
   - Leverage 3x speedup potential
   - Buffer-based tensor operations
   - Test limits of real-time deep perceptual metrics

**Expected Performance:** Pushing boundaries toward full LPIPS at 60 FPS

**Use Case 5: Mobile/Edge Deployment**

**Goal:** Perceptual quality on resource-constrained devices

**Recommended Approach:**
1. **Primary:** SSIM with OpenGL ES compute shaders
   - Proven mobile GPU implementations exist
   - ARM SDK provides tutorials
   - Efficient on battery-powered devices
2. **Lightweight Alternative:** dHash perceptual hashing
   - CPU-based but extremely fast
   - No GPU required
   - Good for coarse similarity checks
3. **Distilled Network:** MobileNetV3-based perceptual proxy
   - Best accuracy/efficiency balance (2025 benchmarks)
   - Optimized for mobile inference
   - Can run on NPU if available

**Expected Performance:** 30-60 FPS on mid-range mobile devices

---

## Deep Dive: Building a Shader-Optimized Perceptual Metric

### Design Principles

1. **Minimize Memory Bandwidth**
   - Use half-float (FP16) whenever possible
   - Exploit texture caching and locality
   - Reduce intermediate feature map sizes

2. **Exploit Parallelism**
   - Separable filters (2D → two 1D passes)
   - Workgroup sizes matching hardware (64 threads)
   - Independent per-pixel operations

3. **Approximate Where Safe**
   - Native GPU math functions (`powf`, `expf` approximations)
   - FMA operations for combined multiply-add
   - Relaxed precision for intermediate calculations

4. **Progressive Quality**
   - Compute cheap metric every frame (SSIM)
   - Run expensive metric every N frames (LPIPS proxy)
   - Async validation on CPU/server for ground truth

### Example: SSIM in WebGL2 Fragment Shader

**Architecture:**
```
Pass 1: Gaussian Blur (Horizontal)
  Input: Original Image A, Original Image B
  Output: Blurred A (horizontal), Blurred B (horizontal)

Pass 2: Gaussian Blur (Vertical)
  Input: Blurred A (horizontal), Blurred B (horizontal)
  Output: μ_A, μ_B (mean images)

Pass 3: Variance & Covariance
  Input: A, B, μ_A, μ_B
  Compute: σ_A², σ_B², σ_AB
  Output: Variance/covariance maps

Pass 4: SSIM Calculation
  Input: μ_A, μ_B, σ_A², σ_B², σ_AB
  Compute: l(x,y) · c(x,y) · s(x,y)
  Output: SSIM map

Pass 5: Pooling (Optional)
  Input: SSIM map
  Output: Mean SSIM score
```

**Fragment Shader Pseudocode (Pass 4):**
```glsl
#version 300 es
precision highp float;

uniform sampler2D u_mu_A;      // Mean of image A
uniform sampler2D u_mu_B;      // Mean of image B
uniform sampler2D u_sigma_A2;  // Variance of A
uniform sampler2D u_sigma_B2;  // Variance of B
uniform sampler2D u_sigma_AB;  // Covariance

in vec2 v_texCoord;
out vec4 fragColor;

const float C1 = 6.5025;   // (K1 * L)^2, K1=0.01, L=255
const float C2 = 58.5225;  // (K2 * L)^2, K2=0.03

void main() {
    float mu_A = texture(u_mu_A, v_texCoord).r;
    float mu_B = texture(u_mu_B, v_texCoord).r;
    float sigma_A2 = texture(u_sigma_A2, v_texCoord).r;
    float sigma_B2 = texture(u_sigma_B2, v_texCoord).r;
    float sigma_AB = texture(u_sigma_AB, v_texCoord).r;

    // Luminance comparison
    float numerator_l = 2.0 * mu_A * mu_B + C1;
    float denominator_l = mu_A * mu_A + mu_B * mu_B + C1;
    float l = numerator_l / denominator_l;

    // Contrast comparison
    float sigma_A = sqrt(sigma_A2);
    float sigma_B = sqrt(sigma_B2);
    float numerator_c = 2.0 * sigma_A * sigma_B + C2;
    float denominator_c = sigma_A2 + sigma_B2 + C2;
    float c = numerator_c / denominator_c;

    // Structure comparison
    float numerator_s = sigma_AB + (C2 / 2.0);
    float denominator_s = sigma_A * sigma_B + (C2 / 2.0);
    float s = numerator_s / denominator_s;

    // SSIM = l * c * s
    float ssim = l * c * s;

    fragColor = vec4(ssim, ssim, ssim, 1.0);
}
```

**Performance Optimization:**
- **Separable Gaussian:** Two passes (horizontal + vertical) instead of 2D convolution
  - 2D: O(w × h × k²) operations
  - Separable: O(w × h × 2k) operations
  - Speedup: k/2 (e.g., 5×5 kernel → 2.5x faster)

- **Integral Images (Advanced):**
  - Precompute prefix sums for O(1) local statistics
  - Requires additional preprocessing pass
  - Worth it for large window sizes

- **Strided Computation:**
  - Compute SSIM every 2-4 pixels instead of every pixel
  - 4-16x speedup with minimal visual quality loss
  - Interpolate between computed values if needed

### Example: Lightweight CNN Proxy

**Training Pipeline:**

```python
import torch
import torch.nn as nn
import lpips

class LightweightPerceptualProxy(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Layer 1: 3 → 32 channels
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 256 → 128

            # Layer 2: 32 → 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 128 → 64

            # Layer 3: 64 → 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
        )

        self.regressor = nn.Linear(128, 1)

    def forward(self, img_a, img_b):
        # Concatenate along channel dimension
        x = torch.cat([img_a, img_b], dim=1)  # (B, 6, H, W)

        # Extract features
        feat = self.features(x)  # (B, 128, 1, 1)
        feat = feat.view(feat.size(0), -1)  # (B, 128)

        # Predict distance
        distance = self.regressor(feat)  # (B, 1)
        return distance

# Training
model = LightweightPerceptualProxy().cuda()
lpips_model = lpips.LPIPS(net='alex').cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(num_epochs):
    for img_a, img_b in dataloader:
        img_a, img_b = img_a.cuda(), img_b.cuda()

        # Teacher: LPIPS ground truth
        with torch.no_grad():
            target = lpips_model(img_a, img_b)

        # Student: Lightweight proxy prediction
        prediction = model(img_a, img_b)

        # Knowledge distillation loss
        loss = nn.MSELoss()(prediction, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

**Deployment to WebGL:**
1. Export trained weights to JSON/binary format
2. Implement 3-layer CNN in fragment shaders (similar to SSIM multi-pass)
3. Load weights as textures
4. Run inference with ping-pong rendering between layers

**Expected Model Size:** 500KB - 2MB (far smaller than VGG's 58MB or AlexNet's 9MB)

### Hybrid Multi-Metric Approach

**Rationale:** Different metrics capture different perceptual aspects. Combining them can provide more robust quality assessment.

**Proposed Combination:**
```
Perceptual Score = α·SSIM + β·TextureSim + γ·ColorDist

where:
  SSIM        = structural similarity (luminance, contrast, structure)
  TextureSim  = lightweight CNN proxy for texture quality
  ColorDist   = perceptual color distance (CIEDE2000 or simpler ΔE)

  α, β, γ     = learned weights or manually tuned
```

**Benefits:**
- SSIM: Fast, captures structure, proven baseline
- CNN Proxy: Learned texture/feature similarity, flexible
- Color Distance: Perceptual color space, important for visual quality

**Implementation:**
- Run SSIM every frame (cheap)
- Run CNN proxy every 2-4 frames (moderate cost)
- Combine scores with weighted average
- Adapt weights based on content type (photos vs graphics vs UI)

**Performance Budget (1080p, 60 FPS = 16.67ms):**
- SSIM: 2ms
- CNN Proxy (every 4 frames): 5ms / 4 = 1.25ms average
- Color Distance: 1ms
- Total: ~4.25ms (leaves 12.4ms for rendering)

---

## Connections to Existing Knowledge

### Relationship to LPIPS WebGL Shader Research

This research directly builds on the finding that **full LPIPS is too heavy for real-time WebGL deployment** (10-30 FPS). The identified alternatives—SSIM, SSIMULACRA2, MILO, and distilled CNNs—provide practical paths to real-time perceptual guidance.

**Key Insight:** The perceptual metrics landscape has evolved significantly in 2023-2026:
- SSIMULACRA2 (2022-2023) now matches or exceeds LPIPS correlation with human perception
- MILO (2025) specifically designed for real-time optimization
- CUDA acceleration for VMAF (2023) makes production neural metrics feasible

### From Training-Time to Runtime Metrics

**Training-Time Paradigm (2016-2023):**
- Use expensive perceptual losses (VGG, LPIPS) to train models
- Deploy lightweight models that run without perceptual network
- "Train with perception, run without it"

**Emerging Runtime Paradigm (2024-2026):**
- GPU-accelerated perceptual metrics enable real-time assessment
- Hybrid architectures combine offline training with runtime guidance
- Lightweight proxies (MILO, distilled CNNs) balance quality and speed

**This Research Bridges the Gap:**
- SSIMULACRA2 at 85 FPS proves modern metrics can run real-time
- Distillation techniques enable custom lightweight proxies
- WebGPU compute shaders make browser-based deployment viable

### Broader Context: The Perceptual Quality Timeline

**2004:** SSIM introduced - first widely-adopted perceptual metric beyond PSNR

**2016:** Johnson et al. "Perceptual Losses for Real-Time Style Transfer"
- VGG features as perceptual loss for training
- 3 orders of magnitude faster than optimization-based methods

**2018:** LPIPS (Zhang et al.) - "Unreasonable Effectiveness of Deep Features"
- Learned calibration weights improve perceptual correlation
- Becomes standard for training neural synthesis models

**2022-2023:** SSIMULACRA2 (Cloudinary)
- MS-SSIM + XYB color space + tuning on modern codecs
- Best subjective correlation among all metrics

**2023:** VMAF-CUDA (Netflix/NVIDIA)
- Official GPU acceleration for production video quality
- Integrated into FFmpeg v6.1

**2024:** GPU-Accelerated SSIMULACRA2 (TurboMetrics)
- 25x speedup, real-time at 1080p
- Proves complex perceptual metrics can run at 60+ FPS

**2025:** MILO
- Lightweight perceptual metric specifically for optimization
- Designed for real-time applications from ground up

**2026:** This Research
- Comprehensive survey of shader-optimized perceptual metrics
- Practical deployment strategies for WebGL/WebGPU
- Bridges gap between training-time quality and runtime performance

### Technical Parallels

**Neural Architecture Search (NAS):**
The challenge of finding minimal perceptual proxies parallels NAS: searching architecture space for optimal speed/quality tradeoff. MobileNetV3, EfficientNetV2 designed via NAS.

**Knowledge Distillation:**
Creating lightweight perceptual metrics is classic knowledge distillation:
- Teacher: Full LPIPS/VGG (slow, accurate)
- Student: 3-5 layer proxy (fast, approximates teacher)
- Training: Soft targets from teacher guide student

**Compression Techniques:**
Lightweight metrics employ same techniques as model compression:
- **Quantization:** FP32 → FP16 → INT8
- **Pruning:** Remove unnecessary channels/layers
- **Distillation:** Transfer knowledge to smaller model

**GPU Computing Evolution:**
- **CUDA (2007):** GPGPU becomes accessible
- **Compute Shaders (2012):** OpenGL 4.3 integrates compute into graphics pipeline
- **WebGL2 (2017):** Brings compute capabilities to browsers (via extensions)
- **WebGPU (2023+):** Native compute shaders for web, ML-first design

---

## Follow-Up Questions & New Research Topics

### Immediate Technical Questions

1. **What is the minimal distilled CNN architecture that maintains >0.85 correlation with LPIPS for texture synthesis tasks?**
   - Priority: 8 - Critical for enabling real-time perceptual guidance in NCAs
   - Approach: Systematic NAS over 2-7 layer architectures with varying channel counts

2. **How does MILO compare to LPIPS and SSIMULACRA2 for texture quality assessment?**
   - Priority: 7 - MILO is new (2025) and claims real-time performance
   - Need: Benchmarks on texture synthesis datasets, correlation analysis

3. **Can SSIMULACRA2's CUDA implementation be ported to WebGPU compute shaders?**
   - Priority: 6 - Would enable best-in-class perceptual metric in browsers
   - Challenges: Float precision differences, extension availability, XYB color space conversion

4. **What is the optimal combination of cheap (SSIM) and expensive (LPIPS proxy) metrics for real-time perceptual guidance?**
   - Priority: 7 - Hybrid approaches may provide best speed/quality balance
   - Variables: Update frequencies, weighting schemes, content-adaptive switching

### Broader Research Directions

5. **Perceptual metric distillation for specific domains (faces, textures, materials)**
   - Priority: 6 - Domain-specific metrics may be smaller and more accurate
   - Example: Texture-specific LPIPS trained only on procedural/natural textures

6. **WebAssembly SIMD implementations of lightweight perceptual metrics**
   - Priority: 5 - Alternative to GPU for CPU-bound perceptual assessment
   - Benefits: No GPU required, universal browser support, no WebGL complexity

7. **Real-time perceptual optimization during NCA inference**
   - Priority: 9 - Novel capability enabled by fast perceptual metrics
   - Approach: NCA accepts perceptual gradient signals, adapts in real-time to user constraints

8. **Perceptual quality prediction from compressed latent representations**
   - Priority: 6 - Useful for generative models operating in latent space (diffusion, VAE)
   - Connection: MILO already works in latent space, extend to NCAs with latent codes

### Application-Focused

9. **User study: Do real-time perceptual metrics improve perceived quality in interactive texture synthesis?**
   - Priority: 7 - Validates entire approach with human subjects
   - Question: Does SSIM/MILO-guided NCA produce textures users prefer over unguided?

10. **Benchmark suite for shader-optimized perceptual metrics**
    - Priority: 5 - Enables fair comparison across implementations
    - Include: Speed tests, correlation analysis, device compatibility matrix

---

## Sources

### Modern Perceptual Metrics (2022-2026)

1. [SSIMULACRA2 GitHub Repository](https://github.com/cloudinary/ssimulacra2) - Official implementation by Cloudinary
2. [SSIMULACRA2 - Codec Wiki](https://wiki.x266.mov/docs/metrics/SSIMULACRA2) - Technical documentation and usage
3. [Fast Computation of SSIMULACRA2 on GPUs](https://wiki.x266.mov/blog/turbo-metrics-performance) - TurboMetrics CUDA implementation benchmarks
4. [Detecting Psychovisual Impact with SSIMULACRA](https://cloudinary.com/blog/detecting_the_psychovisual_impact_of_compression_related_artifacts_using_ssimulacra) - Cloudinary blog post
5. [MILO: Lightweight Perceptual Quality Metric](https://arxiv.org/abs/2509.01411) - September 2025 paper
6. [Efficient Perceptual Image Super Resolution: AIM 2025 Benchmark](https://arxiv.org/html/2510.12765v1) - Perceptual quality constraints
7. [DISTS: Deep Image Structure and Texture Similarity](https://github.com/dingkeyan93/DISTS) - Official implementation
8. [Similarity Metrics for MR Image Translation](https://www.nature.com/articles/s41598-025-87358-0) - January 2025 DISTS evaluation

### VMAF and Video Quality

9. [Calculating Video Quality Using NVIDIA GPUs and VMAF-CUDA](https://developer.nvidia.com/blog/calculating-video-quality-using-nvidia-gpus-and-vmaf-cuda/) - Official NVIDIA blog
10. [Netflix VMAF GitHub Repository](https://github.com/Netflix/vmaf) - Perceptual video quality assessment
11. [VMAF-E: MainConcept Fast VMAF](https://www.mainconcept.com/vmaf-e) - 10x faster lightweight estimate
12. [VMAF-torch: PyTorch Implementation](https://github.com/alvitrioliks/VMAF-torch) - Unofficial PyTorch port
13. [Toward a Better Quality Metric for Video](https://netflixtechblog.com/toward-a-better-quality-metric-for-the-video-community-7ed94e752a30) - Netflix TechBlog on VMAF

### Classical Metrics and GPU Implementations

14. [GPU Based Image Quality Assessment using SSIM](https://www.researchgate.net/publication/301779704_GPU_Based_Image_Quality_Assessment_using_Structural_Similarity_SSIM_Index) - CUDA acceleration research
15. [OpenCV: Similarity Check (PSNR and SSIM) on GPU](https://docs.opencv.org/4.x/dd/d3d/tutorial_gpu_basics_similarity.html) - OpenCV GPU tutorial
16. [pytorch-msssim: Fast MS-SSIM for PyTorch](https://github.com/VainF/pytorch-msssim) - Differentiable multi-scale SSIM
17. [SSimDownscaler.glsl GitHub Gist](https://gist.github.com/igv/36508af3ffc84410fe39761d6969be10) - GLSL SSIM shader for mpv
18. [SSimSuperRes.glsl GitHub Gist](https://gist.github.com/igv/2364ffa6e81540f29cb7ab4c9bc05b6b) - SSIM-based sharpening shader

### Butteraugli and Google Metrics

19. [Butteraugli GitHub Repository](https://github.com/google/butteraugli) - Google's perceptual distance metric
20. [Butteraugli Rust Implementation](https://github.com/imazen/butteraugli) - Pure Rust port from libjxl
21. [Guetzli: Perceptually Guided JPEG Encoder](https://research.google/pubs/pub46077/) - Uses Butteraugli for optimization
22. [Butteraugli - Codec Wiki](https://wiki.x266.mov/docs/metrics/butteraugli) - Technical overview

### Lightweight Neural Networks and Distillation

23. [Lightweight Deep Learning for Resource-Constrained Environments Survey](https://dl.acm.org/doi/10.1145/3657282) - ACM Computing Surveys
24. [Comparative Analysis of Lightweight Deep Learning Models](https://arxiv.org/html/2505.03303v1) - MobileNetV3, SqueezeNet, EfficientNetV2 benchmarks
25. [Light CNN GitHub Repository](https://github.com/Xavier-Zeng/Light_CNN) - MobileNet, ShuffleNet, SqueezeNet implementations
26. [Lightweight Design and Optimization methods for DCNNs](https://arxiv.org/html/2412.16886v1) - Progress and futures
27. [Optimising TinyML with Quantization and Distillation](https://www.nature.com/articles/s41598-025-94205-9) - Scientific Reports 2025
28. [Model Optimization Revolution: Pruning, Distillation, PEFT in 2025](https://medium.com/@hs5492349/the-model-optimization-revolution-how-pruning-distillation-and-peft-are-reshaping-ai-in-2025-c9f79a9e7c2b)

### Perceptual Hashing

29. [pHash: Open Source Perceptual Hash Library](https://www.phash.org/) - Official website
30. [imagehash: Python Perceptual Image Hashing](https://github.com/JohannesBuchner/imagehash) - pHash, dHash, aHash, wHash
31. [JImageHash: Perceptual Image Hashing in Java](https://github.com/KilianB/JImageHash) - Java implementation
32. [Similarity Hashing and Perceptual Hashes](https://billatnapier.medium.com/similarity-hashing-and-perceptial-hashes-963fba36c8b5) - Technical overview
33. [Implementation and Benchmarking of Perceptual Image Hash Functions](https://www.phash.org/docs/pubs/thesis_zauner.pdf) - Thesis by Zauner

### WebGL, WebGPU, and Browser Technologies

34. [Fast, Parallel Applications with WebAssembly SIMD](https://v8.dev/features/simd) - V8 blog
35. [WebAssembly and SIMD: A Match Made in the Browser](https://robaboukhalil.medium.com/webassembly-and-simd-7a7daa4f2ecd) - Medium article
36. [WebAssembly and WebGPU Enhancements for Web AI](https://developer.chrome.com/blog/io24-webassembly-webgpu-1) - Chrome Developers
37. [WebAssembly SIMD Patterns for Data Crunching](https://medium.com/@Nexumo_/8-webassembly-simd-patterns-for-data-crunching-8333c6c088b4) - 8 practical patterns
38. [Compute Shader - OpenGL Wiki](https://www.khronos.org/opengl/wiki/Compute_Shader) - Official OpenGL documentation
39. [LearnOpenGL: Compute Shaders Introduction](https://learnopengl.com/Guest-Articles/2022/Compute-Shaders/Introduction) - Tutorial
40. [OpenGL ES SDK: Introduction to Compute Shaders](https://arm-software.github.io/opengl-es-sdk-for-android/compute_intro.html) - ARM mobile tutorial

### Perceptual Losses and Style Transfer

41. [Perceptual Losses for Real-Time Style Transfer](https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16.pdf) - Johnson et al. ECCV 2016
42. [The Unreasonable Effectiveness of Deep Features as a Perceptual Metric](https://richzhang.github.io/PerceptualSimilarity/) - LPIPS research page
43. [LPIPS GitHub Repository](https://github.com/richzhang/PerceptualSimilarity) - Official implementation
44. [VGG Perceptual Loss PyTorch Implementation](https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49) - Reference code
45. [Learning to Predict Perceptual Visibility in Game Rendering](https://www.nature.com/articles/s41598-024-78254-0) - Scientific Reports 2024
46. [GPU-Friendly Laplacian Texture Blending](https://arxiv.org/html/2502.13945v1) - February 2025 paper

### Additional Image Quality Assessment

47. [Awesome Image Quality Assessment](https://github.com/chaofengc/Awesome-Image-Quality-Assessment) - Comprehensive IQA paper collection
48. [Image Quality Metrics: PSNR vs SSIM](https://ieeexplore.ieee.org/document/5596999/) - IEEE comparison study
49. [A Hitchhiker's Guide to Structural Similarity](https://arxiv.org/pdf/2101.06354) - Comprehensive SSIM overview
50. [Making Sense of PSNR, SSIM, VMAF](https://visionular.ai/vmaf-ssim-psnr-quality-metrics/) - Practical comparison guide

---

## Conclusion

The landscape of shader-optimized perceptual metrics in 2026 is mature and production-ready. **SSIM and its variants** provide real-time baseline quality assessment with 30-80x GPU acceleration, while **SSIMULACRA2** achieves the best subjective correlation at 85+ FPS on modern hardware. **VMAF-CUDA** brings production-grade neural perceptual assessment to real-time video workflows with 4-37x speedups.

For browser-based deployment, **WebGL2 SSIM implementations** are straightforward and proven, while **WebGPU** opens the door to more complex perceptual metrics including distilled lightweight CNNs. **MILO** (2025) represents the cutting edge: a learned perceptual metric explicitly designed for real-time optimization tasks.

**Key Takeaways:**

1. **Real-time perceptual assessment is achievable today:** SSIM at 100+ FPS, SSIMULACRA2 at 85 FPS, VMAF-CUDA at 36x speedup

2. **Multiple tiers of quality/speed tradeoffs exist:**
   - Ultra-fast: PSNR, perceptual hashing (500+ FPS, limited quality)
   - Fast: SSIM, dHash (100+ FPS, good quality)
   - Balanced: MS-SSIM, SSIMULACRA2, MILO (60-85 FPS, excellent quality)
   - Neural: VMAF-CUDA, distilled CNNs (30-60 FPS, state-of-the-art quality)

3. **WebGL2 is viable for SSIM and lightweight CNNs:** Fragment shaders can implement separable Gaussian filters and 3-5 layer networks with careful optimization

4. **WebGPU is the future:** 3x speedups for neural workloads, compute-first design, native tensor operations

5. **Hybrid approaches maximize performance:** Combine cheap metrics (SSIM every frame) with expensive metrics (LPIPS proxy every N frames) for optimal quality/speed balance

**For Practitioners:**

- **Production video:** Use VMAF-CUDA or SSIMULACRA2 with TurboMetrics
- **Browser texture synthesis:** Implement SSIM in WebGL2 shaders, explore MILO when implementations mature
- **Game rendering:** MS-SSIM in compute shaders for multi-scale artifact detection
- **Mobile/Edge:** SSIM with OpenGL ES or MobileNetV3-based distilled metric
- **Research:** Experiment with MILO, distilled CNNs, and WebGPU compute shaders

**The Gap is Closing:**

The historical divide between training-time perceptual quality (LPIPS, VGG) and runtime performance constraints is rapidly narrowing. With GPU acceleration, modern metrics, and distillation techniques, **real-time perceptual guidance during synthesis** is becoming practical rather than aspirational. The next frontier is making these capabilities accessible in browsers via WebGPU and standardized libraries.

**The future is perceptually-guided, real-time, and browser-accessible.**
