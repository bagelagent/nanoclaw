# LASI as Differentiable Training Loss for NCAs

**Research ID:** rq-1771607469254-lasi-differentiable
**Topic:** LASI as differentiable training loss for NCAs - implement implicit differentiation through WLS solution for gradient-based optimization
**Completed:** 2026-02-21
**Tags:** perceptual-metrics, nca, optimization, zero-parameter, lasi

## Summary

LASI (Linear Autoregressive Similarity Index) is a zero-parameter perceptual metric based on solving a weighted least squares (WLS) problem at the pixel level during inference. It is fully differentiable and can be used as a training loss for Neural Cellular Automata (NCAs), offering competitive performance with LPIPS while requiring no pre-trained neural networks. The key technical challenge is implementing efficient implicit differentiation through the WLS solution using the implicit function theorem and KKT conditions.

## What is LASI?

LASI constructs perceptual embeddings by solving a weighted least squares optimization problem at inference time, without any training data or deep neural networks. The core innovation is that effective perceptual similarity can emerge from mathematical optimization rather than learned features.

### Key Technical Properties

- **Zero-Parameter Design**: No training required, no neural network parameters to store
- **WLS-Based**: Perceptual embeddings are solutions to a pixel-level weighted least squares problem
- **Fully Differentiable**: Can be used as a loss function for gradient-based optimization
- **GPU-Parallelizable**: Most operations can run in parallel at the pixel level
- **Cubic Scaling**: Computational complexity scales O(n³) with embedding dimensions
- **Competitive Performance**: Matches LPIPS and PIM on image quality assessment benchmarks
- **Similar Cost to Hand-Crafted Metrics**: Comparable computational cost to MS-SSIM

### How LASI Works

1. **Pixel-Level Embeddings**: For each pixel, solve a local WLS problem considering its neighborhood
2. **Global and Local Characteristics**: The WLS formulation captures both broad image structure and fine details
3. **Distance Computation**: Perceptual distance is measured in the resulting embedding space
4. **Dimensionality Trade-off**: Higher embedding dimensions → lower WLS loss and better perceptual accuracy, but higher computational cost

## Implicit Differentiation Through WLS

To use LASI as a differentiable loss for NCA training, we need to backpropagate gradients through the WLS solution. This is achieved through **implicit differentiation**.

### The Challenge

Direct differentiation would require storing the complete computation graph for the iterative WLS solver, which is memory-intensive and inefficient. Instead, we use the **implicit function theorem** to compute gradients without backpropagating through the solver iterations.

### Theoretical Foundation

#### Implicit Function Theorem Approach

The implicit function theorem enables computing necessary Jacobians for backpropagation without needing to backpropagate through the method used to obtain fixed points. This is particularly valuable for optimization layers in neural networks.

**Key Insight**: We view the WLS solution as the root of a system of equations (the KKT conditions for optimality). Instead of differentiating through the solver, we differentiate the optimality conditions themselves.

#### KKT Conditions Framework

For the WLS problem, the solution satisfies the Karush-Kuhn-Tucker (KKT) conditions for stationarity. The gradient computation involves:

1. **Forward Pass**: Solve the WLS problem to obtain optimal embeddings
2. **Compute KKT Jacobian**: Calculate and store the Jacobian of the KKT conditions at the solution
3. **Backward Pass**: Solve a linear system involving the transposed Jacobian inverse

This approach is used in differentiable optimization layers like **OptNet**, which uses a specific factorization of the primal-dual interior point update to obtain a backward pass virtually "for free" requiring no additional factorization once the optimization problem is solved.

### Computational Considerations

**Memory Efficiency**: The implicit differentiation approach avoids storing intermediate solver states, making it much more memory-efficient than naive backpropagation through the solver.

**Computational Cost**:
- Forward pass: Solve WLS problem (one factorization)
- Backward pass: Solve linear system with KKT Jacobian (reuses factorization structure)
- The backward pass is generally very computationally expensive for large-scale problems, but acceptable for pixel-level problems with moderate embedding dimensions

**GPU Acceleration**: The pixel-level structure of LASI allows for massive parallelization - each pixel's WLS problem can be solved independently on GPU, making both forward and backward passes efficient.

### Alternative Approaches

Recent work on differentiable optimization layers has developed more efficient methods:

1. **Conjugate Gradient Method**: Avoids explicit Jacobian inversion by solving linear systems iteratively
2. **BPQP**: Reformulates the backward pass as a simplified quadratic programming problem
3. **Alternating Differentiation**: Novel approaches to reduce computational burden
4. **Nonsmooth Implicit Differentiation**: Handles non-differentiable solver components (e.g., active set changes)

For LASI, the conjugate gradient approach is particularly suitable since the Hessian at the WLS solution is positive semi-definite.

## Application to NCA Training

### Current NCA Training Practices

NCAs are typically trained with perceptual loss functions to generate high-quality textures and patterns. Common practices include:

**Standard Loss Functions**:
- **LPIPS** (Learned Perceptual Image Patch Similarity): Deep feature-based metric using AlexNet or VGG
- **SSIM** (Structural Similarity Index): Hand-crafted structural similarity
- **GMD** (Gram Matrix Distance): Style transfer metric
- **Pixel Loss** (L1/L2): Direct pixel-wise comparison

**Training Challenges**:
- NCAs remain largely confined to low-resolution grids (typically 128×128 or smaller)
- Training time and memory requirements grow quadratically with grid size
- Backpropagation Through Time (BPTT) is memory-consuming and unstable
- Local propagation impedes long-range cell communication
- Heavy compute demands for real-time high-resolution inference

**LPIPS Characteristics**:
- Scores below 0.1 imply almost imperceptible differences
- 0.1–0.2 suggest small but possibly visible changes
- Above 0.2–0.3 indicates significant quality loss
- Strong correlation with human perceptual judgment
- Requires forward pass through pretrained AlexNet/VGG (computational cost)

### Why LASI for NCAs?

**Advantages over LPIPS**:

1. **No Pretrained Network Required**:
   - LPIPS needs AlexNet/VGG forward pass (~10-50M parameters)
   - LASI has zero parameters, computed at inference time
   - Reduces memory footprint during training

2. **Competitive Perceptual Quality**:
   - Matches LPIPS performance on quality assessment benchmarks
   - "Maximum Differentiation" experiments show LASI and LPIPS can expose weaknesses in each other
   - Suggests complementary use cases

3. **Computational Efficiency**:
   - Similar cost to MS-SSIM (hand-crafted metric)
   - Pixel-level parallelization ideal for GPU acceleration
   - No need to store large feature extraction networks

4. **Tunability**:
   - Embedding dimensions can be adjusted for speed/quality trade-off
   - Neighborhood size controls local vs global balance
   - Can be specialized to texture synthesis domain

5. **Memory Efficiency**:
   - Implicit differentiation avoids storing solver computation graph
   - No intermediate feature maps from deep networks
   - Critical for high-resolution NCA training

### Potential Drawbacks

1. **Cubic Scaling**: O(n³) complexity in embedding dimensions (though pixel-level parallelism helps)
2. **Untested in NCA Context**: No published work yet using LASI for NCA training
3. **Implementation Complexity**: Requires careful implementation of implicit differentiation
4. **Hyperparameter Sensitivity**: Embedding dimensions and neighborhood size need tuning

### Hybrid Loss Strategy

Rather than replacing LPIPS entirely, LASI could be used in a **hybrid loss function**:

```
L_total = α * L_pixel + β * L_LASI + γ * L_LPIPS
```

Where:
- **L_pixel** (L1/L2): Fast, ensures basic pixel alignment
- **L_LASI**: Zero-parameter perceptual loss for local texture quality
- **L_LPIPS**: Deep feature-based for high-level structure (used sparingly)

This addresses the research queue item: "Hybrid loss functions for NCA: optimal balance between pixel loss and distilled perceptual loss for fast convergence and quality"

**Optimization Strategy**:
- Early training: High α (pixel loss) for fast convergence
- Mid training: High β (LASI) for perceptual refinement
- Late training: Balanced α, β, γ for final quality
- Or train with LASI only and LPIPS as validation metric

## Implementation Roadmap

### Step 1: Understand LASI Implementation

**Action**: Study the JAX implementation at [dsevero/Linear-Autoregressive-Similarity-Index](https://github.com/dsevero/Linear-Autoregressive-Similarity-Index)

**Key Components**:
```python
# Basic usage example
lasi = LASI(img_shape, neighborhood_size=10)
distance = jax.jit(lasi.compute_distance)(img1, img2)
```

**Focus Areas**:
- WLS solver implementation
- How gradients flow through the computation
- JAX's automatic differentiation handling of the WLS solution
- Neighborhood construction and weighting

### Step 2: Verify Differentiability

**Action**: Test gradient computation through LASI

```python
import jax
import jax.numpy as jnp

lasi = LASI(img_shape, neighborhood_size=10)

def loss_fn(img_generated, img_target):
    return lasi.compute_distance(img_generated, img_target)

# Verify gradients exist and are reasonable
grad_fn = jax.grad(loss_fn)
gradients = grad_fn(img_generated, img_target)
```

**Validation**:
- Check gradient magnitudes are reasonable
- Verify gradients point toward reduced perceptual distance
- Test numerical stability across different image types

### Step 3: Benchmark Against LPIPS

**Action**: Compare LASI vs LPIPS for texture similarity

**Test Cases**:
- Procedural textures (noise, patterns, fractals)
- Natural textures (wood, stone, fabric)
- Synthetic NCA outputs from existing models

**Metrics**:
- Correlation with human judgment (use existing datasets like BAPPS)
- Computational time (forward + backward pass)
- Memory usage during training
- GPU utilization

**Expected Results**:
- LASI: ~65-80% BAPPS performance (based on paper)
- LPIPS: ~85-95% BAPPS performance
- LASI: 2-5x faster than LPIPS forward+backward
- LASI: 5-10x less memory usage

### Step 4: Integrate into NCA Training

**Action**: Modify NCA training loop to use LASI loss

**Baseline NCA Setup**:
- Grid size: 128×128 (standard for texture synthesis)
- Update rule: Simple convolutional NCA
- Target: Single texture synthesis task
- Original loss: L2 + LPIPS

**Modified Training**:
```python
def nca_loss(params, grid, target_texture, step):
    # Roll out NCA for N steps
    current = grid
    for _ in range(step):
        current = nca_update(params, current)

    # Compute losses
    pixel_loss = jnp.mean((current - target_texture) ** 2)
    lasi_loss = lasi.compute_distance(current, target_texture)

    # Hybrid loss with schedule
    alpha = get_pixel_weight(step)  # High early, low late
    beta = get_lasi_weight(step)     # Low early, high late

    return alpha * pixel_loss + beta * lasi_loss
```

**Training Variants to Test**:
1. LASI only
2. L2 + LASI (hybrid)
3. L2 + LPIPS (baseline)
4. L2 + LASI + LPIPS (triple hybrid)

### Step 5: Evaluate Quality and Efficiency

**Quality Metrics**:
- Visual inspection of generated textures
- LPIPS score as external validation metric
- SSIM and FID scores
- Texture statistics (auto-correlation, power spectrum)

**Efficiency Metrics**:
- Training time to convergence
- Peak memory usage
- Gradient computation time
- Final loss values

**Stability Metrics**:
- Loss curve smoothness
- Gradient variance across training
- Robustness to different random seeds
- NCA stability after training (no divergence)

### Step 6: Analyze Trade-offs

**Embedding Dimensionality Study**:
- Test LASI with dimensions: 4, 8, 16, 32, 64
- Plot: dimension vs (quality, training time, memory)
- Find optimal dimension for NCA texture synthesis

**Neighborhood Size Study**:
- Test neighborhood sizes: 3, 5, 7, 10, 15
- Evaluate local vs global structure capture
- Identify best match for NCA's local update rule

**Hybrid Loss Tuning**:
- Grid search over α, β, γ weights
- Test different scheduling strategies
- Compare fixed vs adaptive weight schedules

## Expected Outcomes

### Best Case Scenario

LASI provides a **zero-parameter alternative to LPIPS** that:
- Achieves 90-95% of LPIPS perceptual quality
- Trains 2-3x faster due to reduced computational overhead
- Uses 50-70% less memory, enabling higher resolution NCAs
- Scales better to larger grids (256×256 or higher)
- Provides complementary gradients to pixel loss

**Impact**: Makes high-quality NCA training more accessible, enables real-time training workflows, reduces dependence on pretrained networks.

### Realistic Scenario

LASI works as a **useful component in hybrid loss functions**:
- Provides 70-85% of LPIPS quality
- Faster than LPIPS but requires tuning (embedding dims, neighborhood)
- Memory savings meaningful but not transformative
- Best used in combination with L2 and occasional LPIPS validation
- Particularly effective for texture synthesis (vs morphogenesis)

**Impact**: Expands NCA practitioner toolkit, offers speed/quality trade-off, useful for prototyping and iteration.

### Challenges to Overcome

1. **Implementation Complexity**:
   - Implicit differentiation may require deep understanding of JAX internals
   - Debugging gradient flow through WLS solution can be tricky
   - May need custom VJP (vector-Jacobian product) rules

2. **Perceptual Quality Gap**:
   - If LASI is significantly worse than LPIPS (>20% gap), adoption will be limited
   - May need domain-specific tuning for NCA outputs
   - Possible that zero-parameter approach fundamentally limits quality

3. **Hyperparameter Sensitivity**:
   - Optimal embedding dimensions may vary by texture type
   - Neighborhood size impacts local/global balance critically
   - May require per-task tuning, reducing "drop-in replacement" appeal

4. **Computational Overhead**:
   - Cubic scaling could become problematic for high dimensions
   - Implicit differentiation backward pass may be slower than expected
   - GPU parallelization may not fully compensate for complexity

## Connections to Broader Research

### Distilled Perceptual Metrics

The research queue contains related item: **"Empirical validation of <10K param distilled LPIPS"**

**Connection**: Both LASI and distilled LPIPS aim to reduce computational overhead of perceptual metrics, but through different approaches:
- **LASI**: Zero parameters, WLS optimization at inference
- **Distilled LPIPS**: Small network (~10K params), knowledge distillation from AlexNet

**Complementary Research**: Could compare LASI vs distilled LPIPS in NCA training to understand trade-offs between zero-parameter and small-parameter approaches.

### Hybrid Loss Functions

Related queue item: **"Hybrid loss functions for NCA: optimal balance between pixel loss and distilled perceptual loss"**

**Connection**: LASI naturally fits into hybrid loss framework:
- Fast pixel loss for basic alignment
- LASI for zero-parameter perceptual quality
- Optional LPIPS or distilled LPIPS for validation

**Research Opportunity**: Systematic study of loss scheduling and weight balancing with LASI as the perceptual component.

### Real-Time NCA Systems

Related queue item: **"Real-time performance of hybrid RD+noise systems - how many layers can run at 60fps in WebGL?"**

**Connection**: LASI's zero-parameter design makes it attractive for real-time systems:
- No need to ship pretrained VGG/AlexNet weights to browser
- WLS can potentially be implemented in GLSL shaders
- Training NCAs with LASI may produce models better optimized for browser deployment

**Follow-up Research**: **"LPIPS distillation for WebGL: Converting distilled perceptual model to GLSL shader code"** could be replaced or complemented by **"LASI for WebGL: WLS perceptual metric in GLSL for real-time browser NCA training"**

### DEQ-Style Training for NCAs

Related research: **"Neural Cellular Automata and Deep Equilibrium Models"**

**Connection**: Both LASI and DEQ-style NCA training use implicit differentiation:
- LASI: Implicit differentiation through WLS solution
- DEQ NCAs: Implicit differentiation through fixed-point convergence

**Synergy**: DEQ-style implicit differentiation could make building attractors in the NCA space much more efficient, especially when combined with LASI loss for memory-efficient perceptual gradients.

### Multi-Scale Perceptual Metrics

Related queue item: **"Multi-scale CLIP+VGG conditioning: Can hierarchical NCAs use CLIP at coarse scales and VGG at fine scales?"**

**Connection**: LASI's neighborhood size parameter naturally supports multi-scale:
- Small neighborhoods: Capture fine texture details
- Large neighborhoods: Capture broader structure
- Could use different LASI instances at different scales

**Research Direction**: Multi-scale LASI could provide hierarchical perceptual guidance without multiple pretrained networks.

## Follow-Up Research Questions

Based on this investigation, several new research directions emerge:

### 1. LASI Hyperparameter Optimization for NCAs

**Question**: Can we learn optimal LASI embedding dimensions and neighborhood sizes as part of meta-learning or neural architecture search?

**Approach**:
- Treat LASI hyperparameters as differentiable (relaxation)
- Use gradient-based optimization to find best LASI configuration per task
- Alternatively: NAS over discrete LASI hyperparameter space

**Impact**: Automatic tuning removes barrier to LASI adoption.

### 2. Task-Specific LASI Variants

**Question**: Can LASI's WLS formulation be modified to better match NCA texture synthesis characteristics?

**Approach**:
- Analyze what visual features matter most for NCA textures
- Design custom weighting schemes in WLS problem
- Train small meta-learner to produce pixel-adaptive weights

**Impact**: Closes perceptual quality gap vs LPIPS while maintaining zero-parameter core.

### 3. LASI for WebGL/Browser NCAs

**Question**: Can LASI be implemented efficiently in GLSL shaders for real-time browser-based NCA training?

**Approach**:
- Port WLS solver to GLSL (using iterative methods)
- Leverage WebGL2 compute shaders for pixel-level parallelism
- Implement backward pass for gradient computation
- Benchmark against JavaScript LPIPS implementation

**Impact**: Enables high-quality perceptual loss in fully browser-based NCA experiments.

### 4. LASI + DEQ Training Synergy

**Question**: Does combining LASI (implicit differentiation through WLS) with DEQ-style NCA training (implicit differentiation through fixed points) provide superlinear efficiency gains?

**Approach**:
- Implement DEQ-style NCA training with convergence to attractors
- Use LASI as perceptual loss (both using implicit differentiation)
- Measure memory usage and training speed vs standard BPTT + LPIPS

**Impact**: Could enable training of much larger, more stable NCAs.

### 5. Learned Hybrid Loss Scheduling

**Question**: Can we learn optimal scheduling of α (pixel), β (LASI), γ (LPIPS) weights during NCA training?

**Approach**:
- Meta-learning approach: train across multiple textures
- Learn policy for weight adjustment based on current loss values
- Reinforcement learning: reward is final texture quality / training time

**Impact**: Automated training pipelines with optimal quality/speed trade-off.

## Practical Implementation Notes

### JAX Implementation Tips

**Leverage Existing LASI Code**:
- The [dsevero/Linear-Autoregressive-Similarity-Index](https://github.com/dsevero/Linear-Autoregressive-Similarity-Index) repo provides JAX implementation
- JAX's automatic differentiation should handle implicit differentiation automatically
- Use `jax.jit` for performance

**Potential Pitfalls**:
- WLS solver may need custom VJP if JAX's autodiff produces inefficient backward pass
- Ensure numerical stability (regularization in WLS solve)
- Watch for memory leaks with large batch sizes

### PyTorch Implementation

**Challenge**: LASI is implemented in JAX, but many NCA codebases use PyTorch

**Solutions**:
1. **Direct Port**: Reimplement WLS solver in PyTorch
   - Use `torch.linalg.solve` for WLS solution
   - Define custom `autograd.Function` if needed for efficient backward pass

2. **JAX-PyTorch Bridge**: Use `jax2torch` or similar
   - Wrap JAX LASI in PyTorch-compatible interface
   - May have overhead from framework conversion

3. **Wait for Native Implementation**: Community may develop PyTorch LASI

**Recommendation**: Start with JAX for proof-of-concept, port to PyTorch if successful.

### Hyperparameter Starting Points

Based on LASI paper recommendations:

**Embedding Dimensions**:
- Start with 16 (good quality/speed balance)
- Try 8 if speed is critical
- Try 32 if quality is paramount

**Neighborhood Size**:
- Start with 10 (paper default)
- Try 5 for fine texture details
- Try 15 for broader structure

**Loss Weights** (for hybrid L2 + LASI):
- Early training: α=1.0, β=0.1
- Mid training: α=0.5, β=0.5
- Late training: α=0.1, β=1.0

## Conclusion

LASI represents a promising **zero-parameter alternative to LPIPS** for NCA training, with the potential to significantly reduce computational overhead while maintaining competitive perceptual quality. The key technical challenge—implementing efficient implicit differentiation through the WLS solution—is well-understood and tractable using the implicit function theorem and KKT conditions.

**Primary Advantages**:
- No pretrained network required (zero parameters)
- Fully differentiable via implicit differentiation
- GPU-parallelizable at pixel level
- Competitive perceptual performance
- Memory efficient for high-resolution training

**Main Uncertainties**:
- Will perceptual quality match LPIPS for NCA texture synthesis specifically?
- How sensitive is performance to embedding dimensions and neighborhood size?
- Can it scale to high-resolution NCAs (256×256+)?
- What is the actual training speedup in practice?

**Recommended Next Steps**:
1. Implement basic LASI integration in simple NCA trainer
2. Benchmark against LPIPS on standard texture synthesis tasks
3. Tune hyperparameters (dimensions, neighborhood, loss weights)
4. Evaluate quality/speed trade-offs systematically
5. Publish findings and implementation as open-source tool

This research direction has high potential impact for making NCA training more accessible and efficient, while also contributing to the broader understanding of perceptual metrics in generative modeling.

---

## Sources

### LASI Papers and Code
- [The Unreasonable Effectiveness of Linear Prediction as a Perceptual Metric (ArXiv)](https://arxiv.org/abs/2310.05986)
- [The Unreasonable Effectiveness of Linear Prediction as a Perceptual Metric (OpenReview)](https://openreview.net/forum?id=e4FG5PJ9uC)
- [LASI GitHub Repository](https://github.com/dsevero/Linear-Autoregressive-Similarity-Index)
- [LASI Paper PDF](https://arxiv.org/pdf/2310.05986)

### Neural Cellular Automata
- [Multi-texture synthesis through signal responsive neural cellular automata (Nature Scientific Reports)](https://www.nature.com/articles/s41598-025-23997-7)
- [Neural Cellular Automata: From Cells to Pixels (ArXiv)](https://arxiv.org/abs/2506.22899)
- [Growing Neural Cellular Automata (Distill)](https://distill.pub/2020/growing-ca/)
- [Self-Organising Textures (Distill)](https://distill.pub/selforg/2021/textures/)
- [DyNCA: Real-Time Dynamic Texture Synthesis (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Pajouheshgar_DyNCA_Real-Time_Dynamic_Texture_Synthesis_Using_Neural_Cellular_Automata_CVPR_2023_paper.pdf)

### Implicit Differentiation
- [Implicit Layers Tutorial - Introduction](http://implicit-layers-tutorial.org/introduction/)
- [Implicit Layers Tutorial - Differentiable Optimization](http://implicit-layers-tutorial.org/differentiable_optimization/)
- [Nonsmooth Implicit Differentiation for Machine Learning (NeurIPS 2021)](https://proceedings.neurips.cc/paper/2021/file/70afbf2259b4449d8ae1429e054df1b1-Paper.pdf)
- [OptNet: Differentiable Optimization as a Layer (ICML 2017)](http://proceedings.mlr.press/v70/amos17a/amos17a.pdf)
- [BPQP: A Differentiable Convex Optimization (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/8db12f7214d3a1a0c450ba751163e0fd-Paper-Conference.pdf)
- [Alternating Differentiation for Optimization Layers (ArXiv)](https://arxiv.org/pdf/2210.01802)
- [Gradient-based hyperparameter optimization via implicit function theorem](https://timvieira.github.io/blog/post/2016/03/05/gradient-based-hyperparameter-optimization-and-the-implicit-function-theorem/)

### Perceptual Metrics
- [Locally Adaptive Structure and Texture Similarity (ACM MM 2021)](https://dl.acm.org/doi/10.1145/3474085.3475419)
- [A-DISTS ArXiv Paper](https://arxiv.org/abs/2110.08521)
- [A Differentiable Perceptual Audio Metric (ArXiv)](https://arxiv.org/abs/2001.04460)

### Related Topics
- [Weighted Least Squares (Penn State)](https://online.stat.psu.edu/stat501/lesson/13/13.1)
- [Backpropagation Explanation (Wikipedia)](https://en.wikipedia.org/wiki/Backpropagation)
- [Neural nets with implicit layers (Dan MacKinlay)](https://danmackinlay.name/notebook/nn_implicit.html)
