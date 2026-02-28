# Real-Time Performance of Hybrid RD+Noise Systems

**Research ID**: rq-1770925716000-hybrid-performance
**Topic**: Real-time performance of hybrid RD+noise systems - how many layers can run at 60fps in WebGL?
**Researched**: 2026-02-24
**Tags**: performance, webgl, real-time, hybrid-methods, optimization

---

## Summary

This research investigates the real-time performance characteristics of hybrid procedural generation systems that combine reaction-diffusion (RD) algorithms with noise functions in WebGL. The key finding is that **60fps performance is achievable but highly dependent on resolution, layer complexity, and implementation strategy**. Modern WebGPU offers 15x performance gains over WebGL for compute-heavy workloads, making it the preferred platform for hybrid systems in 2026.

**Critical Performance Budget**: At 60fps, the entire frame budget is **16.67ms**, including both CPU and GPU work. Fragment shaders, which execute per-pixel and are the primary bottleneck, must fit within this constraint alongside all other rendering operations.

---

## Key Findings

### 1. Frame Budget and Constraints

**60fps Frame Budget**: 16.67ms total per frame (CPU + GPU combined)
- Fragment shaders execute per-pixel, making them the primary bottleneck
- Screen resolution has massive impact: 1920×1080 = 2,073,600 pixels per frame
- Each additional layer multiplies computational cost linearly

**Expensive Operations to Avoid**:
- Loops and conditionals in fragment shaders
- Complex trigonometric functions (though sin-based approaches can be hardware-accelerated)
- Multiple texture samples with cache misses
- Random access patterns that defeat texture caching

### 2. Reaction-Diffusion Performance

**Gray-Scott Model Characteristics**:
- Uses texture ping-pong technique (two framebuffers alternating read/write roles)
- Required because WebGL prohibits reading from a texture while rendering to it
- Typical simulation: 128×128 to 1024×1024 resolution
- Each iteration requires: 9 texture samples (3×3 neighborhood kernel) + convolution math
- Can run at 60fps with default settings in WebGL, though performance varies by browser

**WebGL Fragment Shader Implementation**:
- Ping-pong state changes are unavoidable (framebuffer binding, texture uniforms)
- State changes have non-trivial cost that adds up with multiple passes
- Optimization: reduce texture resolution to 1/4 canvas size with minimal visual quality loss

**WebGPU Compute Shader Implementation**:
- "At least on some devices, the compute variant is a lot faster"
- Uses storage textures instead of framebuffers
- Recommended workgroup size: 64 threads
- For 1024×1024 texture: 64×64 workgroup dispatch (16×16 pixels per workgroup)
- Optimization: threads can prefetch pixels into shared workgroup memory to eliminate redundant texture reads

**Specific Benchmark** (NVIDIA GeForce 8800, older hardware):
- Method 3 procedural terrain: ~260 blocks/second (~80% faster than Method 2)
- Modern GPUs (AMD Radeon RX 7900 XTX): can augment scenes with 79,710 instances in 3.74ms

### 3. Noise Function Performance

**Fractal Brownian Motion (FBM)**:
- Each octave doubles frequency, halves amplitude
- **Computational cost scales linearly with octave count**
- Each octave = one additional noise function evaluation
- Typical octave range: 4-10 octaves
  - 4 octaves: baseline for noticeable detail
  - 8-10 octaves: high detail but significantly more expensive

**Perlin vs Simplex Noise GPU Performance**:
- **2D**: Negligible difference, Perlin often preferred for predictability
- **3D**: Simplex becomes advantageous
  - Perlin: 8 gradient samples (2^3)
  - Simplex: 4 gradient samples (3+1)
- **4D**: Simplex significantly faster
  - Perlin: 16 gradient samples (2^4)
  - Simplex: 5 gradient samples (4+1)

**Optimized 3D Perlin**: ~50 Pixel Shader 2.0 instructions

**Sin-based FBM**: "Super performant" due to hardware-accelerated trigonometric functions, faster than polynomial/hash-based approaches

### 4. Hybrid System Constraints

**Multi-Pass Rendering Costs**:
- Ping-pong technique requires texture swaps and framebuffer rebinding per pass
- WebGL prohibits reading and writing same texture simultaneously
- State changes accumulate: hundreds of iterations = hundreds of state changes
- Batching draw calls improves performance significantly

**Texture Sampling Budget**:
- Texture reads are "usually the most expensive operations"
- Performance depends heavily on cache-friendliness
- Random access patterns cause cache misses → performance degradation
- Recommendation: "Do as much work in vertex shader, interpolate to fragments"

**Layer Composition**:
- Each layer requires full-screen quad rendering
- Additive/multiplicative blending has minimal cost
- Dominant cost: executing fragment shader logic per-pixel per-layer

### 5. WebGL vs WebGPU in 2026

**WebGPU Advantages**:
- **15x performance gains** over WebGL for compute-heavy workloads
- 70% browser support in 2026
- Can render 1 million data points at 60fps (would bring WebGL "to a crawl")
- Compute shaders more efficient than fragment shader ping-ponging
- Better for physics simulations using GPGPU

**Migration Strategy**:
- Progressive enhancement: WebGPU primary, WebGL fallback
- Captures performance gains for 70% of users, maintains compatibility for 30%
- WebGPU recommended for new hybrid systems in 2026

### 6. Mobile vs Desktop Performance

**Desktop GPU Capabilities**:
- Modern GPUs: millisecond-scale procedural generation
- High-resolution textures (1024×1024+) at real-time rates
- 10+ FBM octaves feasible if other operations are minimized

**Mobile GPU Constraints**:
- Thermal throttling and power limits
- 3GB RAM typical for mid-range devices
- Solution: chunk-based generation (16×16 tiles)
- Example: 256×256 tile map generated in <1.5s on Snapdragon 665
- Reduction strategy: evaluate generation functions on-demand instead of storing in textures

---

## Practical Estimates: How Many Layers at 60fps?

Based on research findings, here are realistic estimates for **1920×1080 full-screen rendering**:

### Simple Noise Layers (2D Perlin/Simplex, 4-6 octaves):
- **WebGL**: 5-10 layers at 60fps on desktop GPU
- **WebGPU**: 15-25 layers at 60fps on desktop GPU
- **Mobile**: 3-5 layers at 60fps (mid-range device)

### Reaction-Diffusion Layer (128×128 simulation upscaled):
- **WebGL**: 1-2 RD layers + 3-5 noise layers at 60fps
- **WebGPU**: 1-3 RD layers + 10-15 noise layers at 60fps
- **Mobile**: 1 RD layer (64×64) + 2-3 noise layers at 60fps

### Hybrid System (RD simulation composited with multiple noise layers):
- **Optimal WebGL**: 1 RD pass (128×128) + 4-6 FBM layers (4-6 octaves each)
- **Optimal WebGPU**: 2 RD passes + 10-12 FBM layers (6-8 octaves)
- **Mobile WebGL**: 1 RD pass (64×64) + 2-3 FBM layers (4 octaves)

### High-Detail Scenario:
- Single RD layer with 10+ noise compositing layers: **Possible at 60fps** if:
  - RD simulation runs at reduced resolution (64×64 or 128×128)
  - Noise layers use efficient simplex noise (3D/4D) or sin-based FBM
  - Fragment shader operations are minimized (no conditionals/loops)
  - Vertex shader handles as much interpolation as possible

### Performance Degradation Factors:
- 4K resolution (3840×2160): **4x pixels** = 4x fragment shader cost → expect 1/4 the layer count
- Per-frame RD iterations: Each iteration adds full ping-pong cost
- Texture sampling patterns: Random access can halve effective performance
- Browser variations: Chrome typically fastest, Firefox/Safari 10-30% slower

---

## Optimization Strategies

### 1. **Reduce Simulation Resolution**
- Run RD at 1/4 to 1/16 canvas size
- Upscale with bilinear filtering (minimal quality loss)
- Frees budget for more noise layers

### 2. **Minimize State Changes**
- Batch draw calls wherever possible
- Reuse framebuffer objects
- Pre-bind textures when feasible

### 3. **Leverage Vertex Shader**
- Move calculations from fragment to vertex shader
- Interpolate results across fragments
- "Any calculation done on vertices and interpolated = performance boon"

### 4. **Use Efficient Noise Implementations**
- 2D: Perlin or simplex (similar performance)
- 3D/4D: Simplex (fewer gradient samples)
- Consider sin-based FBM for maximum speed

### 5. **Optimize Octave Counts**
- Start with 4 octaves, only add more if visually necessary
- Diminishing returns beyond pixel-resolution detail
- "No need for infinite sums"—natural performance ceiling exists

### 6. **Compute Shader Optimization (WebGPU)**
- Workgroup size 64 is recommended
- Prefetch neighborhood pixels into shared memory
- Process multiple pixels per thread (e.g., 2×2 tiles)
- Reduces redundant texture reads during convolution

### 7. **Precision Control**
- Use lowest precision fragment shader when sufficient (lowp)
- Balance between quality and ALU instruction count

### 8. **Migrate to WebGPU**
- 15x performance improvement for compute-heavy workloads
- 70% browser support in 2026
- Keep WebGL fallback for remaining 30%

---

## Connections to Existing Knowledge

### Neural Cellular Automata (NCA):
- NCAs face similar real-time constraints as RD systems
- Both use neighborhood convolution operations
- NCA update rules may be more complex than Gray-Scott equations
- Hybrid NCA+noise systems likely have similar performance profiles

### Hierarchical Generation:
- Multi-scale approaches can optimize performance
- Coarse layers (low-res RD) + fine layers (high-frequency noise)
- Mirrors image pyramid techniques in traditional graphics

### CLIP/VGG Conditioning:
- Adding semantic guidance (CLIP embeddings) to RD/noise systems
- Performance impact: additional texture lookups + embedding interpolation
- May require reducing base layer count to maintain 60fps

### Differentiable Rendering:
- Backpropagation through hybrid RD+noise systems possible but expensive
- Real-time forward pass != real-time backward pass
- Training hybrid systems likely requires offline computation

---

## Follow-Up Questions

1. **Learned Hybrid Systems**: Can neural networks learn to predict RD+noise combinations without full simulation, enabling more layers at 60fps?

2. **Asymmetric Resolution**: What's the optimal resolution ratio between RD simulation and noise layers for perceptual quality vs performance?

3. **Temporal Coherence**: Can frame-to-frame coherence be exploited to amortize RD computation across multiple frames?

4. **Sparse Evaluation**: For interactive applications, can RD+noise be evaluated only in visible regions or at variable rates across the screen?

5. **Neural Texture Compression**: Can learned codecs compress RD+noise outputs to reduce bandwidth and enable more layers?

6. **WebGPU Work Graphs**: How do 2026 GPU work graphs (dynamic workload generation) impact hybrid procedural systems performance?

---

## Sources

- [GitHub - piellardj/reaction-diffusion-webgl](https://github.com/piellardj/reaction-diffusion-webgl)
- [Reaction-Diffusion Compute Shader in WebGPU | Codrops](https://tympanus.net/codrops/2024/05/01/reaction-diffusion-compute-shader-in-webgpu/)
- [WebGPU 2026: 70% Browser Support, 15x Performance Gains | byteiota](https://byteiota.com/webgpu-2026-70-browser-support-15x-performance-gains/)
- [WebGPU Fluid Simulations: High Performance & Real-Time Rendering | Codrops](https://tympanus.net/codrops/2025/02/26/webgpu-fluid-simulations-high-performance-real-time-rendering/)
- [Reaction diffusion on shader – ciphrd](https://ciphrd.com/2019/08/24/reaction-diffusion-on-shader/)
- [4.2 Reaction Diffusion - WebGPU Unleashed](https://shi-yan.github.io/webgpuunleashed/Compute/reaction_diffusion.html)
- [The Book of Shaders: Fractal Brownian Motion](https://thebookofshaders.com/13/)
- [Inigo Quilez's fBM](https://iquilezles.org/articles/fbm/)
- [WebGL Fire Shader Based on Fractal Brownian Motion | Mark's Project Blog](https://blog.fixermark.com/posts/2025/webgl-fire-shader-based-on-fbm/)
- [WebGL best practices - Web APIs | MDN](https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API/WebGL_best_practices)
- [Real-Time Procedural Generation with GPU Work Graphs | ACM](https://dl.acm.org/doi/10.1145/3675376)
- [Chapter 1. Generating Complex Procedural Terrains Using the GPU | NVIDIA Developer](https://developer.nvidia.com/gpugems/gpugems3/part-i-geometry/chapter-1-generating-complex-procedural-terrains-using-gpu)
- [Simplex Noise vs Perlin Noise: When and Why - PulseGeek](https://pulsegeek.com/articles/simplex-noise-vs-perlin-noise-when-and-why/)
- [Efficient computational noise in GLSL (arXiv)](https://ar5iv.labs.arxiv.org/html/1204.1461)
- [Analyze shader performance | Android Developers](https://developer.android.com/agi/frame-trace/shader-performance)
- [Gray-Scott Model of a Reaction-Diffusion System](https://itp.uni-frankfurt.de/~gros/StudentProjects/Projects_2020/projekt_schulz_kaefer/)
- [Reaction-Diffusion by the Gray-Scott Model: Pearson's Parameterization](http://www.mrob.com/pub/comp/xmorphia/index.html)
- [Reaction-Diffusion Tutorial - Karl Sims](https://www.karlsims.com/rd.html)
- [Stateful Rendering with Ping-Pong Technique | Medium](https://olha-stefanishyna.medium.com/stateful-rendering-with-ping-pong-technique-6c6ef3f5091a)
- [WebGPU — From Ping Pong WebGL To Compute Shader | Medium](https://medium.com/phishchiang/webgpu-from-ping-pong-webgl-to-compute-shader-%EF%B8%8F-1ab3d8a461e2)
- [WebGL Image Processing Continued](https://webglfundamentals.org/webgl/lessons/webgl-image-processing-continued.html)
