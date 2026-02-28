# LPIPS Distillation for WebGL: Converting Distilled Perceptual Model to GLSL Shader Code

**Research ID:** rq-1771629235297-lpips-webgl-shader
**Date:** February 21, 2026
**Priority:** 6
**Tags:** webgl, shader, perceptual-metrics, real-time, browser

---

## Summary

Converting a distilled LPIPS (Learned Perceptual Image Patch Similarity) model to GLSL shader code for real-time browser-based perceptual loss computation is technically feasible but requires careful architectural choices. The key challenge is fitting the distilled model's neural network inference into WebGL fragment shaders while maintaining real-time performance (60+ fps). Three main approaches exist: (1) ShaderNN-style fragment shader inference using texture-based computation, (2) NVIDIA RTX-style cooperative vectors for tensor operations (not available in WebGL), and (3) explicit GLSL implementation of small MLPs. For a <10K parameter distilled LPIPS model, the most practical WebGL approach combines texture-based weight storage with ping-pong rendering for multi-layer inference, leveraging fp16 half-float textures and depthwise separable convolutions for efficiency.

---

## Key Findings

### 1. LPIPS Architecture & Distillation Target

**Original LPIPS Structure:**
- Uses pre-trained AlexNet backbone for feature extraction
- Extracts features from 5 convolutional layers: Conv1 (64 channels), Conv2 (192 channels), Conv3 (384 channels), Conv4 (256 channels), Conv5 (256 channels)
- Applies learned linear calibration weights on top of features
- Total parameters: ~60M for AlexNet + calibration weights
- AlexNet is the fastest backbone and performs best as a forward metric

**Distillation Goal:**
- Target: <10K parameters while maintaining 65-80% of BAPPS (Berkeley Adobe Perceptual Patch Similarity) performance
- Approach: Knowledge distillation from AlexNet teacher to tiny student network
- Architecture candidates: Depthwise separable convolutions, MobileNet-style blocks, tiny MLPs

**Key Insight:** The distilled model doesn't need to replicate AlexNet's architecture—it only needs to approximate the final perceptual distance metric. This enables radical simplification.

### 2. Neural Network Inference in WebGL Shaders

**Three Implementation Paradigms:**

#### A. ShaderNN (Mobile-Optimized Fragment Shader Inference)
- **Innovation:** First framework to leverage fragment shaders for CNN inference
- **Architecture:** Hybrid compute + fragment shader approach with layer-level optimization
- **Key Features:**
  - Texture-based I/O for zero-copy GPU integration
  - Supports basic CNN operators (conv, pooling, activation)
  - Optimized for parametrically small models (<10K parameters ideal)
  - Pre-built static computation graph with operator fusion
- **Performance:** Outperforms TensorFlow Lite on Qualcomm/MediaTek mobile GPUs
- **Availability:** Open source at github.com/inferenceengine/shadernn
- **WebGL Compatibility:** Uses OpenGL ES, directly translatable to WebGL 2.0

**Sources:**
- [ShaderNN on ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0925231224013997)
- [ShaderNN GitHub](https://github.com/inferenceengine/shadernn)

#### B. NVIDIA RTX Neural Shaders (High-End Desktop)
- **Innovation:** Cooperative vectors expose Tensor Core hardware acceleration to shaders
- **Architecture:** MLPs inlined directly in shader code with `optixCoopVecMatMul` intrinsics
- **Key Features:**
  - Hardware-accelerated matrix operations in shaders
  - Training support (forward + backward propagation) in shaders
  - 20x texture compression for neural assets
- **Performance:** Film-quality materials at real-time framerates
- **Limitations:** Requires NVIDIA RTX hardware, OptiX/DirectX Shader Model 6.9, NOT available in WebGL
- **Relevance:** Proves neural inference in shaders is viable for real-time rendering

**Sources:**
- [Neural Rendering with Cooperative Vectors (NVIDIA)](https://developer.nvidia.com/blog/neural-rendering-in-nvidia-optix-using-cooperative-vectors/)
- [NVIDIA RTX Neural Shading SDK](https://github.com/NVIDIA-RTX/RTXNS)

#### C. SimpNet (VRChat Fragment Shader CNN)
- **Innovation:** Trainable CNN entirely in HLSL fragment shaders
- **Architecture:** Render textures store weights, outputs, and intermediate values
- **Key Features:**
  - Supports conv layers, activations, classification
  - Training via backpropagation in shaders
  - Deployed in real-time virtual environments (VRChat)
- **Limitations:** Primarily HLSL (DirectX), requires porting to GLSL
- **Relevance:** Demonstrates CNNs are feasible in pure shader code without specialized APIs

**Source:** [SimpNet GitHub](https://github.com/SCRN-VRC/SimpNet-Deep-Learning-in-a-Shader)

### 3. WebGL-Specific Implementation Strategy

**For LPIPS distilled to <10K parameters, the optimal WebGL approach:**

#### Weight Storage: FP16 Half-Float Textures
- **Format:** `gl.HALF_FLOAT` (WebGL 2.0 native, WebGL 1.0 via extension)
- **Benefits:**
  - 50% memory savings vs FP32
  - Sufficient precision for neural network weights
  - Hardware-supported in modern GPUs
- **Limitations:**
  - May not be color-renderable by default (check extensions)
  - Often requires `gl.NEAREST` filtering (no bilinear interpolation)
- **Alternative:** Standard compressed texture formats (BC/DXT, ETC) are unsuitable—designed for visual imagery, introduce unacceptable artifacts in numerical data

**Sources:**
- [OES_texture_half_float (MDN)](https://developer.mozilla.org/en-US/docs/Web/API/OES_texture_half_float)
- [WebGL Compressed Texture Formats (MDN)](https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API/Compressed_texture_formats)

#### Multi-Layer Inference: Ping-Pong Rendering
- **Problem:** WebGL prohibits reading from a texture while rendering to it
- **Solution:** Two framebuffers (FBOs) that alternate roles each layer
  - FBO A: Input texture → Fragment shader computes layer → Output to FBO B
  - FBO B: Input texture → Fragment shader computes next layer → Output to FBO A
  - Repeat for each layer, swapping read/write textures
- **Implementation:**
  ```glsl
  // Pseudocode structure
  for (int layer = 0; layer < numLayers; layer++) {
      // Bind input texture (previous layer output or initial input)
      // Bind output framebuffer (next layer input)
      // Run fragment shader for current layer
      // Swap input/output textures
  }
  ```
- **Benefits:** Enables multi-pass shader pipelines, essential for deep networks
- **Performance:** Each layer adds one draw call; for 5-10 layer network, 5-10 passes at 60fps is achievable

**Sources:**
- [Stateful Rendering with Ping-Pong Technique (Medium)](https://olha-stefanishyna.medium.com/stateful-rendering-with-ping-pong-technique-6c6ef3f5091a)
- [WebGL Ping-Pong Shading (grencez.dev)](https://grencez.dev/2020/webgl-texture-ping-pong-20200607/)

#### Efficient Convolutions: Separable Kernels in GLSL
- **Standard Convolution:** N×N kernel requires N² texture lookups per pixel
  - Example: 5×5 kernel = 25 lookups
- **Separable Convolution:** Split into horizontal (1×N) + vertical (N×1) passes
  - Example: 5×5 separable = 5 + 5 = 10 lookups (2.5× faster)
  - Requires two render passes with ping-pong technique
- **Depthwise Separable (MobileNet-style):**
  - Depthwise: Convolve each channel independently (3×3 kernel per channel)
  - Pointwise: 1×1 convolution to mix channels
  - Drastically reduces parameters and computation
- **GLSL Optimization:**
  - Use `texture2D()` with hardware linear filtering to combine neighboring pixels (2× speedup)
  - Fragment shader runs per-pixel; minimize operations per invocation
  - Move invariant computations to vertex shader when possible

**Sources:**
- [Convolution Part Four: Separable Kernels](https://taylorpetrick.com/blog/post/convolution-part4)
- [Optimizing Convolutions (john-chapman.github.io)](https://john-chapman.github.io/2019/03/29/convolution.html)

### 4. Proposed Architecture for Distilled LPIPS in WebGL

**Network Design (Targeting ~5K-10K Parameters):**

1. **Input:** 256×256 RGB image pair (reference and distorted)
2. **Early Feature Extraction (Tiny Conv):**
   - 3×3 depthwise separable conv, 3 → 16 channels
   - ReLU activation
   - Parameters: ~27 (3×3×3 depthwise) + 48 (1×1×3×16 pointwise) = ~75
3. **Perceptual Encoding (3-4 Tiny Blocks):**
   - Each block: 3×3 depthwise separable conv, 16 → 16 channels + ReLU
   - Parameters per block: ~144 (3×3×16) + 256 (1×1×16×16) = ~400
   - Total for 4 blocks: ~1,600 parameters
4. **Spatial Pooling:**
   - Average pooling to reduce spatial dimensions (256×256 → 64×64 or 32×32)
   - Implemented via mipmapping or explicit averaging in shader
5. **Distance Computation (Small MLP):**
   - Concatenate reference and distorted features: 16×2 = 32 channels
   - MLP: 32 → 64 → 32 → 1 (perceptual distance scalar)
   - Parameters: 32×64 + 64×32 + 32×1 = 2,048 + 2,048 + 32 = ~4,128
6. **Total Parameters:** ~75 + 1,600 + 4,128 = ~5,803 parameters

**Shader Implementation:**
- **Weights Encoding:** Pack all weights into a single 2D texture (e.g., 128×128 fp16 texture can store 16K floats)
- **Layer Pipeline:**
  - Vertex shader: Pass-through (fullscreen quad)
  - Fragment shader: Lookup weights from texture, perform convolution or MLP layer
  - 8-10 render passes total (ping-ponging between FBOs)
- **Final Output:** Single-channel texture containing perceptual distance value per pixel (or single global value via reduction)

**Performance Estimates:**
- **Fragment Shader Complexity:** 10-50 instructions per layer (simple conv/MLP)
- **Texture Lookups:** 9-25 per layer (3×3 to 5×5 kernels)
- **Resolution:** 256×256 = 65,536 pixels × 10 layers = 655,360 fragment shader invocations
- **Modern GPUs:** 100M+ fragment shader invocations/sec → ~150-600 fps for this workload
- **Realistic Target:** 60fps easily achievable, likely 120+ fps on desktop GPUs

---

## Deep Dive: Technical Challenges & Solutions

### Challenge 1: Insufficient Tensor Operations in WebGL

**Problem:** WebGL lacks native tensor/matrix operations (unlike CUDA, OptiX cooperative vectors, or Metal Performance Shaders).

**Solution:**
- **Explicit Implementation:** Manually code matrix multiplies and convolutions in GLSL
  - For 1×1 convolutions (pointwise): Simple weighted sum of input channels
  - For 3×3 convolutions: Loop over 9 neighbors, accumulate weighted sum
  - For MLP fully-connected layers: Dot product between input vector and weight row
- **Example GLSL Snippet (3×3 Depthwise Conv):**
  ```glsl
  vec3 depthwiseConv3x3(sampler2D inputTex, vec2 uv, vec2 texelSize, sampler2D weightTex) {
      vec3 result = vec3(0.0);
      for (int dy = -1; dy <= 1; dy++) {
          for (int dx = -1; dx <= 1; dx++) {
              vec2 offset = vec2(float(dx), float(dy)) * texelSize;
              vec3 pixel = texture2D(inputTex, uv + offset).rgb;
              vec3 weight = texelDecode(weightTex, dy+1, dx+1); // Lookup from weight texture
              result += pixel * weight;
          }
      }
      return result;
  }
  ```

### Challenge 2: Knowledge Distillation for Extreme Compression

**Problem:** Compressing 60M parameter AlexNet-based LPIPS to <10K parameters while preserving perceptual correlation.

**Solution (From Prior Research):**
- **Teacher-Student Training:**
  - Teacher: Original LPIPS (AlexNet + calibration)
  - Student: Tiny network (5K-10K params)
  - Loss: Cosine similarity between teacher and student distance predictions
- **Domain Specialization:** Train on texture synthesis dataset (not general ImageNet)
- **Teacher-Assistant Chain:** Intermediate 100K param model bridges the gap
- **Expected Performance:** 65-80% of teacher performance on BAPPS benchmark

**Key Finding:** This distillation must happen *before* shader conversion. The shader implementation assumes weights are already trained.

**Source:** Prior research rq-1771607469255-distilled-lpips-empirical (completed)

### Challenge 3: Real-Time Performance Budget

**Problem:** 60fps = 16.67ms per frame. Perceptual loss computation must fit within 1-2ms to leave budget for rendering.

**Solution:**
- **Resolution Trade-off:** Compute LPIPS at lower resolution (128×128 or 64×64 instead of 256×256)
  - Perceptual metrics are relatively resolution-invariant
  - 4× resolution reduction = 16× fewer fragment shader invocations
- **Asynchronous Computation:**
  - Render main scene in one frame
  - Compute LPIPS in next frame using previous frame's output
  - Pipeline latency of 1 frame (~16ms) is acceptable for many applications
- **Selective Computation:** Only compute LPIPS for regions of interest or on user interaction (not every frame)

### Challenge 4: Debugging & Validation

**Problem:** Shader debugging is notoriously difficult; validating numerical correctness of NN inference is critical.

**Solution:**
- **Reference Implementation:** First implement in PyTorch/TensorFlow, export weights
- **Weight Export:** Convert trained weights to binary format or base64-encoded strings embedded in GLSL
- **Unit Testing:**
  - Test each layer in isolation (CPU Python vs GPU GLSL)
  - Use known input images, compare outputs
  - Tools: WebGL error checking, fragment shader output visualization
- **Gradient Checking (Optional):** If training in shader (advanced), validate gradients against automatic differentiation

---

## Connections to Existing Knowledge

### Related Research Topics

1. **Distilled LPIPS Training** (rq-1771607469255): Provides the <10K parameter model that serves as input to this shader conversion
2. **NAS for Perceptual Metrics** (rq-1771607469256): Could optimize architecture specifically for shader deployment constraints
3. **Hybrid Loss Functions for NCA** (rq-1771629235298): Real-time perceptual loss in shaders enables novel NCA training paradigms
4. **Real-Time Hybrid RD+Noise Systems** (rq-1770925716000): Perceptual metrics in shaders could guide procedural generation at runtime

### Broader Context: Neural Rendering Pipeline

Shader-based LPIPS fits into larger trend of **neural-rendering convergence**:
- **Training:** Offline (PyTorch, TensorFlow) with full distillation from large teacher
- **Deployment:** Online (WebGL shaders) with tiny distilled student
- **Applications:**
  - Real-time texture synthesis quality feedback
  - Interactive artistic style transfer
  - Perceptual-aware image compression in browser
  - NCA training acceleration (replace costly CPU LPIPS calls)

---

## Follow-Up Questions & New Research Topics

### Immediate Follow-Ups
1. **Empirical Validation:** Implement the proposed 5K-10K param architecture in WebGL, benchmark on real devices (mobile, desktop, integrated GPUs)
2. **Automated Shader Code Generation:** Build tool to auto-convert PyTorch/ONNX tiny CNNs to GLSL fragment shaders
3. **Perceptual Loss for WebGL Texture Synthesis:** Integrate LPIPS shader into live NCA or procedural generation demo

### Longer-Term Explorations
1. **WebGPU Compute Shaders:** With WebGPU adoption, can we leverage compute shaders for more efficient tensor operations?
   - Answer: Yes, compute shaders enable shared memory, barriers, and more efficient data access patterns
   - See: [WebGPU Compute Shader Example (Medium)](https://medium.com/phishchiang/webgpu-from-ping-pong-webgl-to-compute-shader-%EF%B8%8F-1ab3d8a461e2)
2. **Learned Shader Compression:** Can we distill the *shader code itself* (meta-optimization)?
3. **Hybrid CPU-GPU Pipeline:** Offload some layers to CPU SIMD (WASM) for complex operations?

---

## Sources

### Neural Inference in Shaders
- [ShaderNN: Mobile GPU Inference (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S0925231224013997)
- [ShaderNN GitHub Repository](https://github.com/inferenceengine/shadernn)
- [NVIDIA Neural Rendering with Cooperative Vectors](https://developer.nvidia.com/blog/neural-rendering-in-nvidia-optix-using-cooperative-vectors/)
- [NVIDIA RTX Neural Shading SDK](https://github.com/NVIDIA-RTX/RTXNS)
- [How to Get Started with Neural Shading (NVIDIA)](https://developer.nvidia.com/blog/how-to-get-started-with-neural-shading-for-your-game-or-application)
- [SimpNet: CNN in Fragment Shaders](https://github.com/SCRN-VRC/SimpNet-Deep-Learning-in-a-Shader)
- [Neural Shading Transforming Graphics (Medium)](https://medium.com/coding-nexus/neural-shading-how-trainable-shaders-and-gpu-ai-are-transforming-real-time-graphics-d1e97ab2cc9f)

### LPIPS Architecture
- [LPIPS GitHub Repository](https://github.com/richzhang/PerceptualSimilarity)
- [LPIPS PyTorch Metrics Documentation](https://lightning.ai/docs/torchmetrics/stable/image/learned_perceptual_image_patch_similarity.html)
- [Experimenting with LPIPS (Medium)](https://medium.com/dive-into-ml-ai/experimenting-with-lpips-metric-as-a-loss-function-6948c615a60c)
- [LPIPS Paper (CVPR 2018)](https://arxiv.org/abs/1801.03924)

### WebGL Shader Techniques
- [Stateful Rendering with Ping-Pong (Medium)](https://olha-stefanishyna.medium.com/stateful-rendering-with-ping-pong-technique-6c6ef3f5091a)
- [WebGL Ping-Pong Shading (grencez.dev)](https://grencez.dev/2020/webgl-texture-ping-pong-20200607/)
- [WebGL Image Processing Continued](https://webglfundamentals.org/webgl/lessons/webgl-image-processing-continued.html)
- [Convolution Part Four: Separable Kernels](https://taylorpetrick.com/blog/post/convolution-part4)
- [Optimizing Convolutions (John Chapman)](https://john-chapman.github.io/2019/03/29/convolution.html)

### WebGL Texture Formats & Optimization
- [OES_texture_half_float (MDN)](https://developer.mozilla.org/en-US/docs/Web/API/OES_texture_half_float)
- [WebGL Compressed Texture Formats (MDN)](https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API/Compressed_texture_formats)
- [Choosing Texture Formats for WebGL/WebGPU](https://www.donmccurdy.com/2024/02/11/web-texture-formats/)
- [WebGL Best Practices (MDN)](https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API/WebGL_best_practices)

### AlexNet Architecture
- [AlexNet Wikipedia](https://en.wikipedia.org/wiki/AlexNet)
- [Understanding AlexNet (LearnOpenCV)](https://learnopencv.com/understanding-alexnet/)
- [AlexNet Architecture Explained (Medium)](https://medium.com/@siddheshb008/alexnet-architecture-explained-b6240c528bd5)

### Knowledge Distillation
- [Knowledge Distillation (Neptune.ai)](https://neptune.ai/blog/knowledge-distillation)
- [PyTorch Knowledge Distillation Tutorial](https://docs.pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html)
- [Knowledge Distillation (Wikipedia)](https://en.wikipedia.org/wiki/Knowledge_distillation)
- [Demystifying Knowledge Distillation (Medium)](https://medium.com/@weidagang/demystifying-knowledge-distillation-in-neural-networks-0f4c82c070ed)

### WebGPU Future Directions
- [WebGPU Compute Shaders (Medium)](https://medium.com/phishchiang/webgpu-from-ping-pong-webgl-to-compute-shader-%EF%B8%8F-1ab3d8a461e2)

---

## Conclusion

Converting a distilled <10K parameter LPIPS model to WebGL GLSL shaders for real-time perceptual loss computation is **technically feasible and practically viable** for browser-based applications. The optimal approach combines:

1. **Distilled Architecture:** Depthwise separable convolutions + small MLP (~5K-10K params)
2. **Weight Storage:** FP16 half-float textures (50% memory savings)
3. **Multi-Layer Inference:** Ping-pong rendering with 8-10 passes
4. **Performance:** 60fps+ achievable at 128×128 resolution, 120fps+ at 64×64

**Key Enablers:**
- ShaderNN demonstrates fragment shader CNN inference is production-ready
- WebGL 2.0 provides fp16 textures and render-to-texture capabilities
- Modern GPUs (mobile and desktop) have sufficient fragment shader throughput

**Critical Dependencies:**
- Successful distillation of LPIPS to <10K params (prior research: rq-1771607469255)
- Careful architecture design for shader-friendly operations (depthwise separable convs, small MLPs)
- Empirical validation on target hardware (mobile browsers, desktop WebGL)

**Next Steps:**
1. Implement proof-of-concept: Export distilled weights → Generate GLSL code → Benchmark performance
2. Develop automated tooling: PyTorch/ONNX → GLSL shader generator
3. Integrate into applications: Real-time NCA, texture synthesis, perceptual compression

This research bridges **knowledge distillation** (ML) and **real-time graphics** (WebGL shaders), enabling perceptual-aware rendering systems entirely in the browser.
