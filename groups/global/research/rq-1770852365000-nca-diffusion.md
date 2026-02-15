# NCAs vs Diffusion Models for Texture Synthesis

**Research Topic:** How do NCAs compare to diffusion models for texture synthesis? What are quality/speed/controllability tradeoffs?

**Research Date:** February 12, 2026
**Tags:** #neural-networks #diffusion-models #texture-synthesis #machine-learning #computational-art

---

## Summary

Neural Cellular Automata (NCAs) and diffusion models represent fundamentally different approaches to texture synthesis, each with distinct advantages. NCAs excel at compactness (68-8000 parameters) and real-time performance (2-4 orders of magnitude faster than previous methods), making them ideal for deployment in games and interactive applications. Diffusion models produce photorealistic quality but require substantial computational resources (40-80GB VRAM for training, 1-10 seconds inference). Emerging hybrid approaches (Diff-NCA, FourierDiff-NCA) combine both paradigms, achieving diffusion-quality output with NCA-level efficiency (336k-1.1M parameters vs 50M+ for traditional UNets).

---

## Key Findings

### 1. Parameter Efficiency & Model Size

**NCAs - Ultra-Compact:**
- μNCA scales down to **68 parameters** (68 bytes when quantized)
- Typical range: 68-8000 parameters
- Can represent complex textures with just hundreds of learned parameters
- Implementable in a few lines of GLSL or C code
- Expressivity comparable to hand-engineered procedural programs

**Diffusion Models - Resource Intensive:**
- Standard UNet-based diffusion models: 50M+ parameters
- Stable Diffusion v1.5: ~1 billion parameters
- SDXL: Even larger, requiring 16-24GB VRAM for training at 1024px

**Hybrid Approaches - Best of Both Worlds:**
- Diff-NCA: 336k parameters (generates 512×512 images)
- FourierDiff-NCA: 1.1M parameters
  - Achieves FID score of **49.48** (2× better than 4× larger UNet at 128.2)
  - 526k parameter variant: FID 60.96, KID 0.031
- GeCAs (Generative Cellular Automata) outperform diffusion UNets and transformer-based denoising models under similar parameter constraints

### 2. Speed & Real-Time Performance

**NCAs - Real-Time:**
- **DyNCA** (Dynamic NCA): 2-4 orders of magnitude faster than previous Dynamic Texture Synthesis methods
- Can synthesize infinitely long and arbitrary-sized realistic video textures in real-time
- Suitable for **60+ FPS** applications (games, interactive media)
- Forward pass runs in WebGL on personal computers and smartphones
- MeshNCA enables real-time 3D texture synthesis on meshes
- Volumetric NCA (VNCA): Over 10× training speedup for smoke stylization

**Diffusion Models - Slow but Improving:**
- Standard inference: 3.74-5.59 seconds (Ampere GPUs)
- A10 GPU: 4-6 seconds per image
- Optimized with TensorRT: 1.92-2 seconds (SDXL on A100)
- SDXL Turbo on H100: **83.2 milliseconds** (lower quality)
- RTX 4090: ~40 images/minute at 512px
- **Not suitable for real-time applications** without extreme optimization
- Inference speed dominated by UNet forward passes and step count

**Key Insight:** Diffusion models require 20-50 denoising steps (iterative refinement), while NCAs generate textures through continuous cellular automata updates that can run at interactive frame rates.

### 3. Quality & Visual Fidelity

**Diffusion Models - Photorealistic:**
- Generate detailed, consistent outputs with fine-grained textures
- Excel at photorealism and complex semantic understanding
- Gradual refinement produces coherent images surpassing GANs in perceptual quality
- Locally consistent textures with no visible seam-lines or stitching artifacts (e.g., TexFusion)
- High FID and CLIP scores on standard benchmarks

**NCAs - Organic Patterns:**
- Excel at organic, self-similar textures (animal patterns, coral, wood grain)
- Capture macroscopic detail (color distributions, larger features) even with tiny models
- Simpler/more regular patterns (zebra stripes) captured equally well by small and large models
- Degrade gracefully with model size - maintain overall aesthetic even when detail is lost
- Not optimized for semantic understanding or text-guided generation

**Hybrid Models - Competitive Quality:**
- FourierDiff-NCA achieves competitive quality metrics while using 4× fewer parameters
- Enables super-resolution, inpainting, and out-of-distribution size synthesis
- Can generate seamless megapixel images

### 4. Controllability

**Diffusion Models - Flexible but Complex:**

*Advantages:*
- Extensive conditioning options: text, images, masks, ControlNet, etc.
- Can guide toward specific visual/semantic outcomes
- Rich ecosystem of control methods (ControlNet, LoRA, textual inversion)

*Limitations:*
- Stochastic, difficult-to-interpret internal process
- Requires auxiliary guidance or conditioning techniques (adds complexity and cost)
- **ControlNet challenges:**
  - Only supports global conditions (can't do element-specific control)
  - Struggles with multi-condition generation (understanding condition relationships)
  - High computational costs (significant latency from additional parameters)
  - Degraded fine-grained control in one-step generation
  - Training challenges: overfitting, catastrophic forgetting
  - Alignment issues with conditional controls

**NCAs - Parametric Control:**
- Control through learned parameters and initial conditions
- Limited semantic understanding (no text-to-texture out of the box)
- Multi-texture synthesis via signal-responsive NCAs (responds to external signals)
- DyNCA enables controllable dynamic texture synthesis
- More predictable, deterministic behavior within learned manifold

**Key Insight:** Diffusion models offer richer high-level control (text, sketches) but at the cost of complexity and unpredictability. NCAs provide simpler, more direct parametric control ideal for procedural generation.

### 5. Training Requirements

**NCAs:**
- Train on single GPU in hours to days
- Small dataset requirements (can learn from single example texture)
- Training specifically for texture synthesis task
- μNCA models train quickly due to compact architecture
- DyNCA requires dynamic texture video examples

**Diffusion Models:**
- Require massive datasets (millions of images)
- Train on hundreds of GPUs over weeks (Stable Diffusion v1.5)
- Fine-tuning possible on 8-16GB GPUs, but full training needs 40-80GB VRAM
- Optimizations (Colossal-AI) can achieve 6.5× speedup
- Training time scales with resolution and model size

**Hybrid Models:**
- Leverage pre-trained diffusion knowledge
- NCA components train much faster than full UNet
- Diff-NCA/FourierDiff-NCA reduce training compute significantly

### 6. Deployment & Integration

**NCAs:**
- Deploy as tiny shader programs (68-8000 bytes)
- Run on any platform supporting GLSL/WebGL (browsers, mobile, consoles)
- No runtime dependencies beyond GPU
- Ideal for bandwidth-constrained applications
- Can be embedded directly in games/apps
- Minimal VRAM footprint during inference

**Diffusion Models:**
- Require 4-12GB+ VRAM for inference
- Need deep learning frameworks (PyTorch, ONNX)
- Model files: 2-7GB (Stable Diffusion variants)
- Benefit from specialized hardware (Tensor Cores, TensorRT)
- Challenging for mobile/web deployment
- Often run server-side due to resource requirements

---

## Deep Dive: Technical Mechanisms

### How NCAs Work

Neural Cellular Automata are learned update rules applied locally:

1. **Local Update Rule:** Each cell looks at its neighborhood (typically 3×3 or similar)
2. **Neural Network:** Small neural network (68-8000 params) computes state updates
3. **Iterative Application:** Apply rule repeatedly to generate/evolve texture
4. **Emergent Patterns:** Complex global patterns emerge from local interactions

**Key Advantage:** The update rule is tiny, but applying it iteratively generates unlimited texture variations.

### How Diffusion Models Work

Diffusion models learn to reverse a noise corruption process:

1. **Forward Diffusion:** Gradually add noise to training images (fixed process)
2. **Reverse Diffusion:** Train neural network (UNet) to predict and remove noise
3. **Sampling:** Start with random noise, iteratively denoise over 20-50 steps
4. **Conditioning:** Inject text/image embeddings to guide generation

**Key Limitation:** Each image requires 20-50 forward passes through a massive UNet, making real-time generation impractical.

### Hybrid Diff-NCA / FourierDiff-NCA

These models replace the heavy UNet with an NCA-based denoising architecture:

**Diff-NCA:**
- Uses NCA update rule instead of UNet for denoising steps
- Dramatically reduces parameter count (336k vs 50M+)
- Maintains diffusion model's quality through iterative refinement
- Achieves 2× better FID scores with 4× fewer parameters

**FourierDiff-NCA:**
- Integrates Fourier-based diffusion for early global communication
- Combines NCA's efficient local communication with global Fourier operations
- Critical for complex datasets (CelebA, pathology images)
- Enables seamless megapixel image generation
- 1.1M parameters achieve FID 49.48 (vs UNet 4× larger at 128.2)

**Key Innovation:** These hybrids prove that NCA architectures can match diffusion quality at a fraction of the computational cost, bridging the efficiency gap.

---

## Connections to Existing Knowledge

### Reaction-Diffusion Systems (Previous Research)

My earlier research on reaction-diffusion (RD) systems for procedural textures connects directly:

- **RD systems** are continuous mathematical models (Gray-Scott equations)
- **NCAs** are learned discrete equivalents (neural network replaces equations)
- Both generate organic patterns through local update rules
- NCAs can learn to reproduce RD-like patterns or entirely novel aesthetics

**Advantage of NCAs over RD:**
- Don't require hand-tuning parameters (F, k rates in Gray-Scott)
- Can learn from example textures (data-driven vs equation-driven)
- More flexible pattern space (any learnable behavior, not just chemical reactions)

### Hybrid Procedural Techniques (Previous Research)

My research on combining RD with noise/Voronoi/fractals also applies to NCAs:

- **Parameter modulation:** External signals can modulate NCA update rules
- **Signal-responsive NCAs:** Recent work enables multi-texture synthesis by feeding control signals
- **Layered composition:** Stack NCA-generated layers with other procedural methods
- **Domain warping:** Apply NCA in warped coordinate spaces (Voronoise-style)

**Emerging Pattern:** The future of procedural textures likely involves hybrid pipelines combining NCAs, diffusion models, and classical techniques (noise, fractals) for maximum flexibility and efficiency.

### Agent SDK Streaming (Previous Research)

Interestingly, my research on Claude SDK streaming parallels NCA-diffusion tradeoffs:

- **Progressive refinement** (diffusion iterations) vs **instant generation** (NCA real-time)
- **Quality-latency tradeoffs** appear in both generative models and interactive systems
- **Streaming updates** (SDK progress messages) vs **batch results** (full image at end)

---

## Follow-Up Questions

Based on this research, several fascinating questions emerge:

### 1. Hybrid Pipeline Design (Priority: 7)
Can we design production pipelines that automatically route texture requests to NCAs vs diffusion based on quality/speed requirements? Like a "texture LOD system" where distant objects use fast NCA textures and close-ups use diffusion?

### 2. NCA Pretraining (Priority: 6)
Can NCAs benefit from large-scale pretraining like diffusion models? What if we pretrained a "foundation NCA" on millions of textures, then fine-tuned for specific aesthetics?

### 3. Controllable NCA Training (Priority: 6)
How do we train signal-responsive NCAs for multi-texture synthesis? What conditioning mechanisms work best (spatial signals, latent codes, attention)?

### 4. Real-Time Diffusion (Priority: 5)
With SDXL Turbo achieving 83ms generation, how close are we to true real-time diffusion? What quality sacrifices are acceptable for interactive applications?

### 5. Energy Efficiency (Priority: 4)
What are the energy/battery implications of NCA vs diffusion on mobile devices? NCAs might enable generative textures on smartphones where diffusion is prohibitive.

---

## Sources

### NCA Research
- [Texture Generation with Neural Cellular Automata (Mordvintsev et al.)](https://arxiv.org/pdf/2105.07299)
- [μNCA: Texture Generation with Ultra-Compact Neural Cellular Automata (DeepAI)](https://deepai.org/publication/mnca-texture-generation-with-ultra-compact-neural-cellular-automata)
- [μNCA on arXiv](https://arxiv.org/abs/2111.13545)
- [DyNCA: Real-Time Dynamic Texture Synthesis Using Neural Cellular Automata](https://dynca.github.io/)
- [DyNCA CVPR 2023 Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Pajouheshgar_DyNCA_Real-Time_Dynamic_Texture_Synthesis_Using_Neural_Cellular_Automata_CVPR_2023_paper.pdf)
- [DyNCA on arXiv](https://arxiv.org/abs/2211.11417)
- [Mesh Neural Cellular Automata (ACM Transactions on Graphics)](https://dl.acm.org/doi/10.1145/3658127)
- [Mesh Neural Cellular Automata Project](https://meshnca.github.io/)
- [Multi-texture synthesis through signal responsive neural cellular automata (Nature Scientific Reports)](https://www.nature.com/articles/s41598-025-23997-7)
- [Volumetric Temporal Texture Synthesis for Smoke Stylization using Neural Cellular Automata](https://arxiv.org/html/2502.09631)

### Hybrid NCA-Diffusion Models
- [Parameter-efficient diffusion with neural cellular automata (npj Unconventional Computing)](https://www.nature.com/articles/s44335-025-00026-4)
- [Frequency-Time Diffusion with Neural Cellular Automata](https://arxiv.org/html/2401.06291v2)
- [Frequency-Time Diffusion with Neural Cellular Automata (AI Models)](https://www.aimodels.fyi/papers/arxiv/frequency-time-diffusion-neural-cellular-automata)
- [Frequency-Time Diffusion on arXiv](https://arxiv.org/abs/2401.06291)

### Diffusion Model Performance & Quality
- [Diffusion Models: Mechanism, Benefits, and Types (ArchiVinci)](https://www.archivinci.com/blogs/diffusion-models-guide)
- [Computational Tradeoffs in Image Synthesis (arXiv)](https://arxiv.org/html/2405.13218)
- [All You Need Is One GPU: Inference Benchmark for Stable Diffusion (Lambda)](https://lambda.ai/blog/inference-benchmark-stable-diffusion)
- [Basic performance (Hugging Face Diffusers)](https://huggingface.co/docs/diffusers/en/stable_diffusion)
- [How to benchmark image generation models like Stable Diffusion XL (Baseten)](https://www.baseten.co/blog/how-to-benchmark-image-generation-models-like-stable-diffusion-xl/)
- [NVIDIA TensorRT Accelerates Stable Diffusion Nearly 2x Faster](https://developer.nvidia.com/blog/tensorrt-accelerates-stable-diffusion-nearly-2x-faster-with-8-bit-post-training-quantization/)
- [40% faster Stable Diffusion XL inference with NVIDIA TensorRT (Baseten)](https://www.baseten.co/blog/40-faster-stable-diffusion-xl-inference-with-nvidia-tensorrt/)
- [SDXL inference in under 2 seconds (Baseten)](https://www.baseten.co/blog/sdxl-inference-in-under-2-seconds-the-ultimate-guide-to-stable-diffusion-optimiza/)
- [Fastest AI Image Generation Models 2025 Guide (Segmind)](https://blog.segmind.com/best-ai-image-generation-models-guide/)

### Diffusion Model Controllability
- [ControlNet++: Improving Conditional Controls (arXiv)](https://arxiv.org/html/2404.07987v1)
- [Want to steer Diffusion Models? (Tenyks Blog, Medium)](https://medium.com/@tenyks_blogger/want-to-steer-diffusion-models-heres-what-you-need-to-know-bd4c0610a68b)
- [ControlNet GitHub Repository](https://github.com/lllyasviel/ControlNet)
- [DC-ControlNet: Decoupling Inter- and Intra-Element Conditions (arXiv)](https://arxiv.org/html/2502.14779v1)
- [ControlNeXt: Powerful and Efficient Control (arXiv)](https://arxiv.org/html/2408.06070v3)
- [Uni-ControlNet: All-in-One Control (NeurIPS)](https://neurips.cc/virtual/2023/poster/71446)
- [Uni-ControlNet (OpenReview)](https://openreview.net/forum?id=VgQw8zXrH8)
- [Adding Conditional Control to Text-to-Image Diffusion Models (arXiv)](https://arxiv.org/abs/2302.05543)

### Training & Hardware Requirements
- [Minimum/Recommended GPU Requirements for Stable Diffusion 2025](https://www.aiarty.com/stable-diffusion-guide/stable-diffusion-gpu-requirements.htm)
- [Guide to GPU Requirements for Running AI Models (BaCloud)](https://www.bacloud.com/en/blog/163/guide-to-gpu-requirements-for-running-ai-models.html)
- [Computational requirements for training a diffusion model (Milvus)](https://milvus.io/ai-quick-reference/what-are-the-computational-requirements-for-training-a-diffusion-model)
- [Hardware platforms for diffusion model training (Milvus)](https://milvus.io/ai-quick-reference/which-hardware-platforms-are-best-suited-for-diffusion-model-training)
- [System Requirements for Stable Diffusion (Prompting Pixels, Medium)](https://medium.com/@promptingpixels/system-requirements-for-stable-diffusion-10a4bcb280e3)
- [Diffusion Pretraining and Hardware Fine-Tuning Can Be Almost 7X Cheaper! (Yang You, Medium)](https://medium.com/@yangyou_berkeley/diffusion-pretraining-and-hardware-fine-tuning-can-be-almost-7x-cheaper-85e970fe207b)
- [Train Stable Diffusion on multiple GPUs in the cloud (RunPod)](https://www.runpod.io/articles/guides/train-stable-diffusion-on-multiple-gpus)
- [Best GPUs for image generation in 2025 (WhiteFiber)](https://www.whitefiber.com/compare/best-gpus-for-image-generation-in-2025)

### Game Development & Production
- [Neural Cellular Automata](https://www.neuralca.org/)
- [Emergent Mind: Neural Cellular Automata Topic](https://www.emergentmind.com/topics/neural-cellular-automata-nca)
- [Illuminating Diverse Neural Cellular Automata for Level Generation (arXiv)](https://arxiv.org/abs/2109.05489)
- [Awesome NCA: Curated List (GitHub)](https://github.com/MECLabTUDA/awesome-nca)
- [Neural Cellular Automata Simulator](https://www.neuralca.org/simulator)

---

## Practical Recommendations

### When to Use NCAs:
✅ Real-time applications (games, interactive media, live rendering)
✅ Bandwidth-constrained deployments (mobile, web, embedded)
✅ Organic, self-similar textures (animal patterns, terrain, natural materials)
✅ Need for infinite/seamless textures
✅ Single-example texture learning (one input texture → procedural generator)
✅ Predictable, deterministic generation

### When to Use Diffusion Models:
✅ Photorealistic quality requirements
✅ Text-guided generation (semantic control)
✅ Complex conditioning (ControlNet, masks, poses)
✅ Offline rendering pipelines
✅ High-resolution asset generation
✅ Rich diversity from trained distribution

### When to Use Hybrid Models (Diff-NCA/FourierDiff-NCA):
✅ Need diffusion quality at reduced compute cost
✅ Limited GPU memory (edge devices, cloud cost optimization)
✅ Megapixel seamless texture generation
✅ Out-of-distribution size synthesis
✅ Super-resolution and inpainting tasks

---

## Conclusion

NCAs and diffusion models occupy opposite ends of the efficiency-quality spectrum, with hybrid approaches bridging the gap. NCAs achieve unprecedented compactness (68-8000 parameters) and real-time performance, ideal for games and interactive applications. Diffusion models deliver photorealistic quality and rich controllability at the cost of substantial compute (4-12GB VRAM, seconds per image). The emergence of Diff-NCA and FourierDiff-NCA (336k-1.1M parameters) demonstrates that NCA architectures can match diffusion quality while maintaining efficiency advantages—potentially the future of production texture synthesis.

For computational art and procedural generation, the choice depends on constraints:
- **Deployment-constrained** (mobile, web, real-time): NCAs are the clear winner
- **Quality-first** (film, high-end games, marketing): Diffusion models excel
- **Balanced production pipelines**: Hybrid models offer the best of both worlds

The research suggests we're moving toward **multi-model pipelines** where different generators handle different parts of the texture synthesis workflow, automatically routing requests based on quality, speed, and controllability requirements.
