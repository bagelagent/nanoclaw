# Real-Time Perceptual-Guided Texture Synthesis in WebGL

**Research Topic:** Integrating shader-based LPIPS into live NCA or procedural generation for real-time perceptual-guided texture synthesis in WebGL

**Research Date:** February 22, 2026
**Priority:** 7/10
**Source:** Research follow-up from LPIPS WebGL shader study

---

## Summary

Real-time perceptual-guided texture synthesis in WebGL represents the convergence of three key technologies: Neural Cellular Automata (NCA) for efficient procedural generation, perceptual loss functions (particularly LPIPS) for human-aligned quality assessment, and WebGL/shader-based acceleration for real-time performance. While NCAs have achieved remarkable real-time performance in browsers (60+ FPS at full HD), and perceptual losses are widely used in training, the direct integration of LPIPS-style perceptual metrics into real-time WebGL rendering pipelines remains a significant technical challenge due to the computational demands of VGG-based feature extraction.

The current state-of-the-art employs a **"train with perception, run without it"** paradigm: models are trained offline using full perceptual losses (LPIPS, VGG features), then deployed as lightweight cellular automata or implicit neural representations that run efficiently in WebGL without requiring the perceptual network during inference. Recent advances in 2026 show promising paths toward hybrid architectures and approximation techniques that could enable limited perceptual guidance during real-time synthesis.

---

## Key Findings

### 1. Neural Cellular Automata Achieve Real-Time Performance

**NCAs are production-ready for WebGL:**
- The "Growing Neural Cellular Automata" work by Distill (2020) demonstrated browser-based NCAs using WebGL/GLSL shaders
- Recent work "Neural Cellular Automata: From Cells to Pixels" (2026) achieves **full-HD (1920×1080) real-time generation** using SwissGL
- NCAs with implicit decoders (Local Pattern Producing Networks) can generate high-resolution outputs from coarse cellular grids
- DyNCA (2023) achieves **2-4 orders of magnitude** performance improvement over previous dynamic texture synthesis methods
- μNCA demonstrates that texture generation can work with **ultra-compact models** (few hundred parameters)

**Technical Implementation:**
- Cell states: typically 16-dimensional vectors (RGBA + 12 hidden channels)
- Update rule: Small neural networks (~8,000 parameters) applied per cell
- Perception: Sobel filters for local gradients create 48D perception vectors
- Stochastic updates: 50% random masking eliminates synchronization dependencies
- WebGL optimization: 8-bit quantization using arctan compression for bounded activations

**Key Architecture Insight:**
The separation of concerns in NCA+decoder systems is elegant: NCAs handle the temporal/spatial evolution on a coarse grid, while lightweight implicit decoders (LPPNs) upscale to arbitrary resolution. This architecture is inherently parallelizable and maps naturally to fragment shaders.

### 2. Perceptual Losses Are Training Tools, Not Runtime Tools

**Current Practice:**
- Perceptual losses (LPIPS, VGG features, Gram matrices) are used **during training** to guide learning
- The trained model (NCA, feed-forward network, etc.) runs **without** the perceptual network at inference time
- This achieves 3+ orders of magnitude speedup compared to optimization-based methods

**LPIPS Technical Details:**
- Computes Euclidean distance between deep network activations (AlexNet, VGG, or SqueezeNet)
- AlexNet (9.1 MB): fastest for forward scoring
- VGG (58.9 MB): recommended for backpropagation/optimization
- SqueezeNet (2.8 MB): most compact
- Learned linear calibration weights improve correlation with human perception
- Scores: <0.1 = imperceptible, 0.1-0.2 = small differences, >0.2-0.3 = significant quality loss

**Why Not Runtime Perceptual Loss?**
VGG-based perceptual losses require:
1. Multiple forward passes through convolutional layers (5 pooling layers in VGG16)
2. Feature extraction at multiple scales (typically layers relu1_2, relu2_2, relu3_3, relu4_3)
3. Significant memory for intermediate activations
4. Computational complexity that's prohibitive at 60 FPS

### 3. WebGL2 Technical Capabilities and Limitations

**Strengths:**
- Float and half-float textures always available (critical for neural network feature maps)
- 3D textures, non-power-of-two textures supported
- Integer and float textures for precise computations
- Guaranteed 16+ texture units per shader
- Efficient render-to-texture workflows

**Limitations for Neural Network Inference:**
- Float textures **not filterable by default** (only nearest interpolation) unless OES_texture_float_linear extension is available
- Float textures **not color-renderable by default** without extensions
- Must check extension support for features like linear filtering of floats
- Texture format combinations can fail even if individually supported
- Limited texture unit count (16 minimum) constrains simultaneous feature map access

**Practical Impact:**
These limitations mean that implementing VGG-style networks in WebGL2 requires careful handling of texture formats, explicit extension checking, and potentially falling back to half-float precision or using workarounds for filtering operations.

### 4. Emerging Solutions and Workarounds

**A. ShaderNN: Fragment + Compute Shader Hybrid**
- First framework to jointly exploit fragment shaders and compute shaders for CNN inference
- Layer-level shader selection: choose optimal shader type per layer
- Achieves favorable performance vs TensorFlow Lite on mobile devices
- Designed for "parametrically small" neural networks (perfect for perceptual approximations)
- OpenGL-based, could inform WebGL strategies

**B. SwissGL: Simplified GPGPU Programming**
- Minimalistic wrapper over WebGL2 (<1000 lines)
- Single `glsl()` function for shader management
- Eliminates verbose WebGL boilerplate
- Particle simulations run **hundreds of steps per second** even on mobile
- Used successfully for NCA implementations (cells2pixels project)
- Expression format for simple operations, multiline for complex shaders

**C. Lightweight Perceptual Approximations**
The Johnson et al. "Perceptual Losses for Real-Time Style Transfer" approach points the way:
- Train a feed-forward network to approximate perceptual optimization
- Network processes 512×512 images at **20 FPS** (2016 hardware)
- 3 orders of magnitude faster than optimization-based methods
- The feed-forward network "learns to quickly approximate solutions to the optimization problem"

**D. WebGPU Migration Path**
While WebGL2 is current focus, WebGPU represents the future:
- TensorFlow.js stable diffusion shows **3x performance gain** WebGL→WebGPU
- Compute shaders designed for ML workloads (vs graphics-focused WebGL)
- Better memory access patterns (tensors as buffers, not textures)
- No awkward texture packing/unpacking
- Native compute shader support without workarounds

### 5. Practical Integration Strategies

Based on current technology (Feb 2026), three viable approaches emerge:

#### **Strategy A: Offline Perceptual Training Only (Production-Ready)**
- Train NCA/procedural generator using full LPIPS/VGG losses
- Deploy lightweight model (<10KB weights) to WebGL
- Real-time performance: 60+ FPS at 1080p
- Quality: Indistinguishable from perceptually-guided generation
- **Drawback:** No runtime perceptual feedback, can't adapt to user input dynamically

#### **Strategy B: Hybrid Client-Server (MixRT/StreamSplat Approach)**
- Lightweight mesh/displacement map rendered in WebGL client
- High-fidelity perceptual refinement on server GPU
- Depth-based fusion combines client and server renders
- **MixRT achieves:** 30+ FPS at 1280×720 on MacBook M1 Pro
- **Advantage:** Full perceptual quality with acceptable latency
- **Drawback:** Requires server infrastructure, network dependency

#### **Strategy C: Approximate Perceptual Network in WebGL (Experimental)**
- Implement ultra-compact perceptual proxy in shaders
- Options:
  - Simplified VGG (2-3 layers only, heavy pruning)
  - Learn to approximate LPIPS with few convolutions
  - Use SqueezeNet (2.8 MB) as starting point
  - Distill VGG knowledge into 3-5 layer network
- Target: 1-2ms perceptual check budget at 60 FPS (16.67ms frame)
- **Challenge:** Maintaining perceptual correlation with simplified architecture
- **Use case:** Interactive refinement where 30-60 FPS acceptable

### 6. NCA Training with Perceptual Metrics

Research shows the following training approaches work well:

**Loss Functions:**
- **Appearance:** Pixel-wise L2, SSIM, LPIPS, or Gram matrices (style loss)
- **Motion** (for dynamic textures): Temporal consistency metrics
- **Stability:** Overflow loss to prevent state explosions
- **Gradient normalization:** Per-variable L2 normalization prevents training instabilities

**Training Techniques:**
- **Pool-based training:** Maintain 1024+ starting states, sample batches, replace highest-loss samples with seed
- **Damage-based training:** Random circular damage regions before iterations to encourage regeneration
- **Residual updates:** Output incremental changes rather than absolute states
- **Stochastic masking:** Zero out 50% of cell updates randomly
- **Multi-scale supervision:** Compare at multiple resolutions for better detail

**Practical Performance:**
- DyNCA training uses appearance + motion targets
- μNCA achieves good results with only few hundred parameters
- Multi-texture synthesis possible with signal-responsive updates

---

## Deep Dive: Technical Feasibility Analysis

### Question: Can We Run LPIPS in Real-Time in WebGL?

Let's break down the computational requirements:

**VGG16 Architecture (typical perceptual loss baseline):**
- Input: 224×224×3 (or larger, scaled proportionally)
- Conv layers: 13 convolutional layers
- Pooling: 5 max pooling layers
- Typical feature extraction: 5 intermediate layers (relu1_2 through relu4_3)
- Parameters: ~138M (full network), but only early layers needed for perceptual loss

**Computational Analysis:**

For 512×512 input at 60 FPS (16.67ms budget):

1. **Feature Extraction:** ~50-100 million MACs (multiply-accumulate operations) for forward pass through first 3-4 conv blocks
2. **Memory:** ~20-50 MB for intermediate feature maps
3. **Comparison:** L2 distance across 64-512 channel feature maps at multiple scales

**Modern GPU Capabilities:**
- Mobile GPUs: ~100-500 GFLOPS
- Desktop WebGL: 1-5 TFLOPS
- Required: ~3-6 GFLOPS for VGG forward pass at 60 FPS

**Verdict:** Theoretically possible on desktop, challenging on mobile.

**BUT:** The real bottleneck is WebGL2's architecture:
- Texture packing overhead (CNNs want buffers, WebGL2 gives textures)
- Extension dependencies for float operations
- Memory bandwidth for large feature maps
- Shader compilation and state changes

**More Realistic Assessment:**
- Full LPIPS: 10-30 FPS on desktop WebGL2, 2-5 FPS mobile
- Lightweight proxy (3-5 layers): 30-60 FPS desktop, 15-30 FPS mobile
- With WebGPU: 2-3× improvement possible

### Hybrid Architecture Proposal

Combining insights from cells2pixels, DyNCA, and perceptual loss research:

```
┌─────────────────────────────────────────────────────┐
│              User Input / Seed State                │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│  Coarse NCA Grid (e.g., 64×64)                      │
│  • 16-channel states                                │
│  • Sobel perception                                 │
│  • Learned update rule (8K params)                  │
│  • WebGL fragment shader                            │
│  • Runs at 100+ FPS                                 │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│  Implicit Decoder (LPPN)                            │
│  • Tiny MLP per pixel                               │
│  • Input: local cell average + coordinates          │
│  • Output: RGB(A) at target resolution              │
│  • Parallelizable across all pixels                 │
│  • Runs at 60 FPS for 1080p                         │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│  Optional: Lightweight Perceptual Check             │
│  • Simplified 3-layer CNN (SqueezeNet-inspired)     │
│  • Compute perceptual delta from target             │
│  • Feed gradient back to NCA as modulation          │
│  • Runs every N frames (N=2-4) for 30-60 FPS        │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│              Final Rendered Output                  │
└─────────────────────────────────────────────────────┘
```

**Key Design Decisions:**

1. **NCA operates on coarse grid:** Reduces computation dramatically while maintaining expressiveness
2. **Decoder handles upscaling:** Lightweight, parallelizable, no perceptual network needed
3. **Perceptual check is optional:** Can skip entirely for maximum FPS, or run at lower frequency
4. **Perceptual network is ultra-simplified:** Only 3-5 layers, possibly trained to approximate LPIPS rather than being full VGG
5. **Gradient modulation, not direct control:** Perceptual feedback modulates NCA evolution rather than dictating it

**Training Strategy:**
- Train NCA+Decoder using full LPIPS/VGG losses offline
- Optionally train lightweight perceptual proxy to approximate LPIPS scores
- Deploy both to WebGL for runtime perceptual guidance
- Proxy loss guides real-time adaptation to user inputs or constraints

### WebGL Implementation Considerations

**Texture Strategy:**
- NCA state: `RGBA32F` or `RGBA16F` texture (64×64 needs 4 textures for 16 channels)
- Feature maps: `RGBA16F` if extensions available, fallback to `RGBA8` with normalization
- Ping-pong rendering: Two framebuffers for current/next state
- Check `OES_texture_float_linear` support for smooth filtering

**Shader Organization:**
```glsl
// NCA Update Shader (Fragment)
uniform sampler2D u_state;     // Current cell states
uniform sampler2D u_weights;   // Network weights as texture
uniform float u_dt;            // Time step

void main() {
    // Sobel perception
    vec3 grad_x = sobel_x(u_state, v_texcoord);
    vec3 grad_y = sobel_y(u_state, v_texcoord);

    // Perception vector (48D, need multiple textures)
    // ...neural network forward pass...

    // Stochastic update
    if (random(v_texcoord) > 0.5) {
        FOut = current_state + delta * u_dt;
    } else {
        FOut = current_state;
    }
}
```

**Performance Targets:**
- NCA update: <2ms (120 FPS capable)
- Decoder upscale: <8ms at 1080p
- Perceptual check: <5ms (if included)
- Total: ~15ms = 66 FPS (leaves 1.67ms headroom)

---

## Connections to Existing Knowledge

### Relationship to Previous Research

**From LPIPS WebGL Shader Study:**
This research directly follows up on investigating LPIPS implementation in WebGL. The key insight is that **full LPIPS is too heavy for 60 FPS**, but lightweight approximations or hybrid architectures are viable.

**From Procedural Techniques Research:**
NCAs represent a learnable form of procedural generation. Unlike traditional noise functions (Perlin, Worley) or reaction-diffusion, NCAs can be trained to produce specific textures while maintaining the self-organizing, infinite properties of procedural systems.

**From Neural Rendering:**
Implicit neural representations (INRs) like the LPPN decoders used with NCAs are part of the broader neural rendering revolution. They convert discrete representations (NCA grids) into continuous functions (decode at any resolution).

### Broader Context

**Computer Graphics Evolution:**
This work sits at the intersection of three trends:
1. **Procedural content generation** (classic graphics): infinite, parameterizable, compact
2. **Neural/learned approaches** (modern ML): data-driven, perceptually-aligned, flexible
3. **Real-time interactivity** (web/games): browser-based, accessible, low-latency

**Why This Matters:**

Traditional procedural generation (Perlin noise, L-systems, etc.) is fast and compact but hard to control precisely. Deep learning methods (GANs, diffusion models) produce high-quality results but are slow and require large models. **NCAs with perceptual training bridge this gap:**

- Compact models (KB, not GB)
- Real-time performance (60+ FPS)
- Perceptually-guided quality (trained with LPIPS)
- Self-organizing properties (regeneration, stability)
- Browser-accessible (WebGL, no installation)

**Applications:**
- **Game development:** Procedural textures that match art direction
- **Creative tools:** Interactive texture synthesis for artists
- **Generative art:** Real-time evolving patterns with aesthetic control
- **Material design:** PBR texture generation with perceptual constraints
- **Video effects:** Real-time stylization and texture transfer

### Technical Parallels

**Neural Architecture Search (NAS):**
The process of finding compact perceptual proxies is similar to NAS: searching for minimal architectures that approximate expensive computations. SqueezeNet → LPIPS approximation is a distillation problem.

**Knowledge Distillation:**
Training a small network to mimic LPIPS scores is classic knowledge distillation: the teacher (VGG + LPIPS) trains the student (3-layer proxy) on soft targets.

**Differentiable Simulation:**
NCAs are differentiable cellular automata. This connects to broader differentiable simulation trends (differentiable physics, rendering, etc.) where simulations are made learnable.

---

## Follow-Up Questions & New Research Topics

### Immediate Technical Questions

1. **What is the minimal perceptual network that maintains >0.8 correlation with LPIPS?**
   - Could a 3-layer CNN trained on LPIPS predictions approximate it well enough?
   - Priority: 6 - Directly enables Strategy C implementation

2. **How do texture format limitations in WebGL2 affect CNN layer implementations?**
   - Quantitative benchmarks for float vs half-float precision in perceptual metrics
   - Priority: 5 - Practical implementation detail

3. **Can we train NCAs to be sensitive to perceptual gradients at inference time?**
   - Rather than fixed weights, could NCA updates accept perceptual feedback signals?
   - Priority: 7 - Novel architecture enabling true real-time perceptual guidance

### Broader Research Directions

4. **Perceptual metrics beyond LPIPS optimized for real-time GPU evaluation**
   - DISTS, SSIM+, learned metrics specifically designed for shader implementation
   - Priority: 8 - Could revolutionize approach entirely

5. **Multi-scale NCA hierarchies for adaptive quality**
   - Coarse NCA at 60 FPS, fine refinement at 15 FPS with perceptual guidance
   - Priority: 6 - Practical quality/performance tradeoff

6. **WebGPU implementation of full LPIPS + NCA pipeline**
   - With compute shaders and better memory model, is full LPIPS at 60 FPS feasible?
   - Priority: 7 - Near-future technology, worth investigating

7. **User study: perceptual quality of NCA textures vs GAN/diffusion outputs**
   - Do users notice the difference? Is LPIPS correlation sufficient?
   - Priority: 5 - Validates entire approach

### Creative Applications

8. **Interactive texture evolution with real-time style constraints**
   - User paints style regions, NCA adapts textures with perceptual guidance
   - Priority: 4 - Application-focused, depends on core tech

9. **Procedural PBR material synthesis with perceptual albedo/normal/roughness correlation**
   - Multi-channel NCA trained with perceptual losses on material maps
   - Priority: 6 - High-value application for game dev

10. **Real-time video stylization using NCA propagation with perceptual temporal consistency**
    - Frame-to-frame NCA updates guided by perceptual motion coherence
    - Priority: 7 - Exciting application, publishable if successful

---

## Sources

### Primary Research Papers & Projects

1. [Neural Cellular Automata: From Cells to Pixels](https://cells2pixels.github.io/) - 2026 work with implicit decoders and SwissGL implementation
2. [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/) - Foundational NCA work with WebGL implementation
3. [DyNCA: Real-Time Dynamic Texture Synthesis](https://dynca.github.io/) - 2023 work achieving 2-4 order of magnitude speedup
4. [LPIPS GitHub Repository](https://github.com/richzhang/PerceptualSimilarity) - Reference implementation of learned perceptual metric
5. [μNCA: Ultra-Compact Neural Cellular Automata](https://ar5iv.labs.arxiv.org/html/2111.13545) - Demonstrates minimal parameter counts
6. [Multi-texture synthesis through Neural Cellular Automata](https://www.nature.com/articles/s41598-025-23997-7) - Recent 2025 work on NCA texture synthesis

### Technical Implementations & Libraries

7. [SwissGL GitHub Repository](https://github.com/google/swissgl) - Minimalistic WebGL2 wrapper for GPGPU
8. [SwissGL Demos](https://google.github.io/swissgl/) - Live demonstrations
9. [ShaderNN](https://github.com/inferenceengine/shadernn) - Fragment + compute shader CNN inference
10. [ShaderNN Research Paper](https://www.sciencedirect.com/science/article/pii/S0925231224013997) - Academic publication on ShaderNN
11. [MixRT: Mixed Neural Representations for Real-Time NeRF](https://arxiv.org/html/2312.11841) - WebGL neural rendering at 30+ FPS
12. [StreamSplat: Hybrid Client-Server Architecture](https://dl.acm.org/doi/10.1145/3746237.3746316) - WebGL neural graphics with depth fusion

### Perceptual Loss & Style Transfer

13. [Perceptual Losses for Real-Time Style Transfer](https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16.pdf) - Johnson et al. foundational paper
14. [The Unreasonable Effectiveness of Deep Features as a Perceptual Metric](https://richzhang.github.io/PerceptualSimilarity/) - LPIPS research page
15. [VGG Perceptual Loss PyTorch Implementation](https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49) - Reference implementation
16. [VGG Loss Explained](https://paperswithcode.com/method/vgg-loss) - Technical overview

### WebGL/WebGPU Technical Resources

17. [WebGL2 Fundamentals: Shaders and GLSL](https://webglfundamentals.org/webgl/lessons/webgl-shaders-and-glsl.html) - Core WebGL concepts
18. [WebGL2 Data Textures](https://webgl2fundamentals.org/webgl/lessons/webgl-data-textures.html) - Texture handling for CNNs
19. [WebGL2 Cross-Platform Issues](https://webgl2fundamentals.org/webgl/lessons/webgl-cross-platform-issues.html) - Extension compatibility
20. [WebGPU: Next Generation Browser Graphics](https://www.linkedin.com/pulse/webgpu-next-generation-browser-graphics-compute-4d-pipeline-5chyc) - Future migration path
21. [How I Re-implemented PyTorch for WebGPU](https://praeclarum.org/2023/05/19/webgpu-torch.html) - WebGPU ML framework

### Graphics & Rendering Techniques

22. [Efecto: Real-Time ASCII and Dithering Effects with WebGL](https://tympanus.net/codrops/2026/01/04/efecto-building-real-time-ascii-and-dithering-effects-with-webgl-shaders/) - Recent 2026 WebGL effects
23. [MeshNCA: Neural Cellular Automata on Meshes](https://meshnca.github.io/) - 3D extension of NCA concepts
24. [Graphics-LPIPS](https://github.com/MEPP-team/Graphics-LPIPS) - LPIPS adapted for 3D graphics
25. [Neural Geometric Level of Detail](https://nv-tlabs.github.io/nglod/assets/nglod.pdf) - Real-time implicit 3D shape rendering

### Additional Context

26. [TensorFire](https://tenso.rs/) - WebGL deep learning library
27. [Jeeliz WebGL Deep Learning](https://github.com/jeeliz/jeelizAR) - CNN on GPU with WebGL
28. [regl-cnn: Digit Recognition with CNN in WebGL](https://github.com/Erkaman/regl-cnn) - Educational WebGL CNN example
29. [A Review of Image Quality Metrics for Generative Models](https://blog.paperspace.com/review-metrics-image-synthesis-models/) - Context on perceptual metrics
30. [DISTS: Deep Image Structure and Texture Similarity](https://pmc.ncbi.nlm.nih.gov/articles/PMC7817470/) - Alternative perceptual metric

---

## Conclusion

Real-time perceptual-guided texture synthesis in WebGL is at a fascinating inflection point. The core technologies—NCAs for efficient generation, perceptual losses for quality, WebGL for deployment—are all mature and proven. However, **direct integration remains challenging** due to computational constraints.

The most promising path forward is **hybrid architectures** that combine:
1. Offline training with full perceptual losses
2. Lightweight deployment of learned models (NCAs, implicit decoders)
3. Optional ultra-simplified perceptual proxies for runtime feedback

With WebGPU on the horizon, compute shader support will likely make full LPIPS at 60 FPS feasible within 1-2 years. Until then, the "train with perception, run without it" paradigm delivers excellent results today.

**The next breakthrough** will likely come from finding compact perceptual proxies—3-5 layer networks that approximate LPIPS with >0.8 correlation. This would enable true real-time perceptual guidance without sacrificing performance.

For practitioners today: use NCAs trained with LPIPS for production work. For researchers: explore perceptual distillation and WebGPU implementations. The future of interactive, perceptually-guided generative graphics is bright.
