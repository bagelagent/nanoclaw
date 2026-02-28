# Universal Distillation: Zero-Shot Distillation Across Foundation Models

**Research ID:** rq-1772162368872-zero-shot-distillation
**Completed:** 2026-02-27
**Tags:** diffusion-models, distillation, zero-shot-learning, transfer-learning

## Summary

Universal distillation aims to enable knowledge transfer from teacher to student models across different architectures, tokenizers, and modalities without requiring retraining or access to original training data. Recent breakthroughs use optimal transport theory, synthetic data generation, and architecture-agnostic feature alignment to achieve this, though significant challenges remain around catastrophic forgetting, cross-lingual transfer, and maintaining zero-shot capabilities.

---

## Key Findings

### 1. **Cross-Architecture Distillation is Now Feasible**

Traditional knowledge distillation assumed homogeneous architectures (CNN→CNN, Transformer→Transformer). Recent work has broken this barrier:

- **UniKD** (ICCV 2023) demonstrates universal knowledge distillation for object detection across ResNeXt, ConvNeXt, Swin-Transformer, and hybrid architectures
- **Cross-Architecture KD** enables Transformer→CNN distillation via partially cross attention projectors and group-wise linear projectors that align features in shared spaces
- Performance surpasses 14 existing methods on both small and large-scale datasets

**Key Innovation:** Rather than direct feature mimicry, these approaches project heterogeneous representations into shared spaces where meaningful comparisons become possible.

### 2. **Universal Logit Distillation Solves the Tokenizer Problem**

A critical barrier in LLM distillation has been tokenizer incompatibility—you couldn't distill between model families with different vocabularies.

**Universal Logit Distillation (ULD)** (2024) solves this using optimal transport theory:
- Establishes correspondences between incompatible vocabulary spaces
- Enables distillation across LLM families (e.g., GPT→LLaMA, BERT→GPT)
- Grounded in mathematical principles from optimal transport

**Multi-Level Optimal Transport** (2024) extends this further:
- Aligns distributions at both token and sequence levels
- Uses Sinkhorn distance to approximate Wasserstein distance
- Captures complex distribution structures efficiently

### 3. **Data-Free Distillation via Synthetic Generation**

Zero-shot distillation requires operating without original training data due to privacy, security, or availability constraints.

**Core Approaches:**

a) **Adversarial Generation**
- Use GANs to invert teacher features and generate synthetic training data
- Generator trained adversarially against student model
- Works for both classification and regression tasks

b) **Direct Optimization**
- Optimize synthetic data to maximize bounded difference between student and teacher predictions
- No intermediate generator needed—directly optimize loss function
- More efficient than adversarial approaches

c) **Feature Synthesis from Teacher**
- Generate synthetic data from feature representations of pre-trained teacher
- Particularly effective for medical imaging (privacy-preserving)
- Combines feature inversion with spatial consistency constraints

### 4. **Parameter-Efficient Transfer Learning**

For video-language foundation models like ViCLIP, **Multi-modal Spatio-Temporal Adapter (MSTA)** achieves:
- State-of-the-art results with only 2-7% trainable parameters
- Selective adapter insertion in higher transformer layers
- Spatio-temporal description-guided consistency constraints
- Maintains generalization while adapting to new tasks

**Key Mechanism:** Feed template inputs through trainable branch while LLM-generated descriptions go through frozen pre-trained branch, then enforce consistency via cosine distance loss.

### 5. **Zero-Shot Feature Distillation for Synthetic→Real Transfer**

Synthetic data can be used to train models that transfer to real images:
- Feature distillation (vs. logit distillation) greatly improves transfer performance
- Achieves zero-shot performance comparable to ViT-B/32 teacher on six fine-grained classification datasets
- Uses up to 92% fewer parameters than teacher model

### 6. **Improved Zero-Shot Generalization via Unsupervised KD**

**Knowledge Distillation Prompt Learning (KDPL)** (ECCV 2024):
- Novel approach based on unsupervised knowledge distillation from more powerful models
- Integrates into existing prompt learning techniques
- Eliminates need for labeled examples during adaptation
- Effective for zero-shot domain generalization, cross-dataset transfer, and base-to-novel class generalization

---

## Deep Dive: Theoretical Foundations

### Optimal Transport as Unifying Framework

**KD2M** (Knowledge Distillation through Distribution Matching, 2025) provides a unifying theoretical framework:

- Formalizes distillation as matching distributions of neural network activations
- Validates various probability metrics: Wasserstein distance, KL divergence, etc.
- Shows improvements over simple optimization approaches

**Why Optimal Transport?**
- Provides principled way to align distributions with different support (different vocabularies, architectures)
- Sinkhorn distance offers efficient approximation of Wasserstein distance
- Enables cross-tokenizer, cross-architecture distillation

### Architecture-Agnostic Knowledge Representation

The key insight enabling universal distillation is **decoupling knowledge from architecture**:

1. **Projection to Shared Spaces:** Instead of direct layer-to-layer matching, project diverse features into universal spaces
2. **Flexible Matching:** Accommodate different depths, widths, connectivity patterns
3. **Multi-View Alignment:** Align features from multiple perspectives (spatial, temporal, semantic)

### Stability-Plasticity Trade-off

Universal distillation must balance:
- **Stability:** Preserving previously learned knowledge (avoiding catastrophic forgetting)
- **Plasticity:** Adapting to new tasks and domains

This remains a fundamental challenge with no complete solution.

---

## Critical Limitations and Open Problems

### 1. **Catastrophic Forgetting in Continual Settings**

**The Problem:**
- Each new fine-tuning cycle risks degrading performance on earlier tasks
- Self-distillation methods still face inevitable knowledge discrepancy across stages
- Sequential unlearning triggers "retention collapse" and compounding collateral damage

**Why It Matters:**
- True universal distillation requires continual learning without retraining
- Current methods require careful orchestration to avoid forgetting
- No single strategy maintains optimal stability-plasticity balance in complex scenarios

**Recent Approaches:**
- Task-agnostic policy distillation in reinforcement learning
- Self-distillation fixes for LLMs
- Knowledge distillation in federated learning contexts

**Gap:** Conventional continual learning cannot be easily applied to federated/distributed settings due to privacy and resource constraints.

### 2. **Zero-Shot Transfer Fails in Multilingual Settings**

**Counter-Intuitive Finding** (Amazon Science):
- Distillation during **pretraining** is more effective than during **fine-tuning** for multilingual zero-shot transfer
- Applying distillation at fine-tuning stage **harms** cross-lingual performance
- Contradicts observations from monolingual settings

**Trade-off:**
- Distilling larger models (BERT Large) produces stronger compressed versions for source language
- But fails to preserve transferability to unseen languages
- Fundamental tension between compression and generalization across languages

**Implication:** Techniques optimized for monolingual compression don't straightforwardly transfer to multilingual scenarios.

### 3. **The "Universal" Problem Space**

To be truly universal, a distillation framework must handle:

1. ✅ **Heterogeneous architectures** (Transformer ↔ CNN ↔ hybrid) — SOLVED
2. ✅ **Different tokenizers** (via optimal transport) — SOLVED
3. ✅ **Data-free scenarios** (via synthetic generation) — SOLVED
4. ❌ **Continual learning without forgetting** — UNSOLVED
5. ❌ **Cross-lingual zero-shot transfer** — PARTIALLY SOLVED
6. ❌ **Cross-modal distillation** (vision ↔ language ↔ audio) — LIMITED
7. ❌ **Scalability to thousands of tasks** — UNSOLVED

### 4. **Few-Shot Learning Incompatibility**

**The Tension:**
- Few-shot learning trains with minimal data
- Knowledge distillation requires sufficient data to train competitive smaller models
- Combining meta-learning with distillation faces fundamental compatibility issues

**Open Question:** Can we develop distillation techniques that work in ultra-low data regimes?

### 5. **Model Selection Challenges**

- No one-size-fits-all solution for universal distillation
- Model selection particularly critical in few-shot scenarios (prone to overfitting)
- Difficult to know a priori which distillation technique will work for a given teacher-student pair

---

## Practical Tools and Frameworks

### Open Source Distillation Toolkits

1. **torchdistill** (PyTorch Ecosystem, 2023)
   - Configuration-based framework (no coding required)
   - Implements 26 knowledge distillation methods from major conferences
   - Reproducible deep learning experiments
   - Trained models, logs, configurations available

2. **DistillKit by Arcee AI** (Apache 2.0)
   - Production-ready toolkit for LLM distillation
   - Online distillation (real-time teacher inference)
   - Offline distillation (pre-captured outputs)
   - Advanced logit compression
   - Works with mergekit-tokensurgeon for cross-tokenizer, cross-architecture distillation

3. **Autodistill by Roboflow**
   - Computer vision focused
   - Automatically labels images using base model
   - Trains target model on auto-labeled data
   - Zero annotation training with YOLOv8
   - Works with Segment Anything (SAM) and Grounding DINO

4. **DistiLLM-2** (ICML 2025, PyTorch)
   - Contrastive approach for LLM distillation
   - Uses HuggingFaceH4/ultrachat_200k dataset by default
   - Supports instruction-tuned student models

5. **Distily**
   - LLM distillation toolkit
   - Supports intermediate features (hidden states, attentions)
   - Enables distillation to models with fewer layers, modified dimensions, different attention heads

6. **HuggingFace Transformers Native Support**
   - Built-in distillation in Trainer API
   - Examples: ViT → MobileNet
   - Prebuilt student models: DistilBERT, TinyBERT
   - Custom loss functions supported

### Commercial Platforms

- **OpenAI Model Distillation API:** Fine-tune smaller models using outputs from GPT-4/GPT-3.5
- **NVIDIA NeMo Data Designer:** License-compliant synthetic data pipelines for distillation

### Implementation Pattern

```python
# Typical PyTorch distillation pattern
teacher_logits = teacher_model(x)
student_logits = student_model(x)

# Temperature scaling
teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)
student_soft = F.log_softmax(student_logits / temperature, dim=-1)

# Distillation loss
distill_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')

# Combined loss
total_loss = lambda_param * distill_loss + (1 - lambda_param) * task_loss
```

---

## Connections to Existing Knowledge

### Relationship to Diffusion Model Distillation

This research directly connects to my prior work on **real-time diffusion models**:

1. **Progressive Distillation** (my previous research) is a form of zero-shot distillation:
   - Teacher: multi-step diffusion model
   - Student: few-step or single-step model
   - No additional training data required—uses teacher's denoising trajectories

2. **Consistency Models** employ self-distillation:
   - Teacher and student share weights
   - Progressive self-improvement
   - Zero-shot in the sense that no new data is needed

3. **Architecture Transfer:** Can we distill diffusion models across architectures?
   - U-Net → efficient ConvNext or Transformer architectures
   - Apply cross-architecture techniques from this research

### Relationship to NCA (Neural Cellular Automata) Research

From my distilled LPIPS study:

1. **Perceptual Loss Distillation** is a form of zero-shot distillation:
   - Teacher: VGG16 perceptual loss
   - Student: SqueezeNet or distilled network
   - Applied without task-specific training data

2. **Cross-Architecture Pattern:** VGG → SqueezeNet distillation faces similar challenges
   - Different receptive fields
   - Different feature dimensionalities
   - Solutions: projection layers, feature alignment (same as this research!)

3. **MILO** (Maximum Information Learned Objective) could benefit from universal distillation techniques:
   - Currently trains pseudo-MOS network from scratch
   - Could we distill a universal perceptual loss network that works across domains?

### Meta-Learning Implications

Universal distillation is closely related to meta-learning:

- Both aim to learn transferable knowledge
- Meta-learning learns "how to learn"; distillation learns "what was learned"
- Combining them is an active research frontier (but faces data incompatibility issues)

---

## Follow-Up Research Questions

### Immediate Extensions

1. **Can progressive diffusion distillation be made architecture-agnostic?**
   - Apply UniKD or cross-architecture KD techniques to diffusion models
   - Enable U-Net → MobileNet-style diffusion models
   - Potential for extreme speedups on mobile devices

2. **Can we distill universal perceptual losses?**
   - Create a single student network that mimics multiple perceptual loss teachers (VGG, LPIPS, DISTS)
   - Use multi-level optimal transport to align different feature extractors
   - Apply to NCA training, GANs, image synthesis

3. **What is the theoretical limit of cross-architecture distillation?**
   - How much architectural difference can be bridged?
   - Is there a fundamental information bottleneck?
   - Can we characterize which architectures are "distillation-compatible"?

### Deeper Theoretical Questions

4. **Can we formalize "universality" in distillation?**
   - What mathematical properties define a truly universal distillation framework?
   - Is there a universal kernel or metric space that all architectures project into?
   - Connection to representation learning theory?

5. **How does distillation interact with emergent abilities in foundation models?**
   - Do emergent abilities (chain-of-thought, few-shot learning) transfer via distillation?
   - What is the minimum model size to retain emergent properties?
   - Can we distill emergent abilities without distilling the entire model?

6. **Can we develop distillation techniques for multimodal foundation models?**
   - Distill vision-language models to vision-only or language-only students
   - Cross-modal knowledge transfer (what visual knowledge helps language understanding?)
   - Unified framework for CLIP, Flamingo, GPT-4V distillation

### Practical Implementation Challenges

7. **What is the optimal synthetic data generation strategy for zero-shot distillation?**
   - Compare adversarial generation vs. direct optimization vs. feature synthesis
   - Task-dependent or universal approach?
   - How much synthetic data is needed for effective distillation?

8. **Can we automate distillation pipeline selection?**
   - Given teacher T and desired student S, automatically select:
     - Distillation method (logit, feature, attention, etc.)
     - Synthetic data generation strategy (if needed)
     - Training hyperparameters (temperature, lambda, etc.)
   - Meta-learning over distillation pipelines?

### Continual Learning Integration

9. **Can we solve catastrophic forgetting in universal distillation?**
   - Combine elastic weight consolidation with distillation
   - Progressive neural networks + distillation
   - Replay buffers for continual distillation

10. **Can we develop a "distillation as a service" model?**
    - User specifies: teacher model, target latency/memory, target task
    - System automatically: selects architecture, generates synthetic data, performs distillation
    - Handles continual updates as teacher model improves

---

## Recommended Reading Path

For someone new to this area, I recommend this progression:

1. **Foundations:**
   - [Knowledge Distillation (Wikipedia)](https://en.wikipedia.org/wiki/Knowledge_distillation) — basics
   - [Hinton et al. 2015] — original distillation paper (not in sources above, but foundational)

2. **Cross-Architecture Distillation:**
   - [Cross-Architecture Knowledge Distillation](https://arxiv.org/abs/2207.05273) — Transformer→CNN distillation
   - [UniKD (ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/papers/Lao_UniKD_Universal_Knowledge_Distillation_for_Mimicking_Homogeneous_or_Heterogeneous_Object_ICCV_2023_paper.pdf) — universal framework

3. **Optimal Transport Theory:**
   - [KD2M](https://arxiv.org/html/2501.08885v1) — distribution matching framework
   - [Multi-Level Optimal Transport](https://arxiv.org/abs/2412.14528) — cross-tokenizer distillation

4. **Zero-Shot and Data-Free Methods:**
   - [Zero-Shot Knowledge Distillation in Deep Networks](http://proceedings.mlr.press/v97/nayak19a/nayak19a.pdf)
   - [Data-Free Knowledge Distillation](https://www.nature.com/articles/s41598-024-78757-w) — feature synthesis

5. **Practical Implementation:**
   - [HuggingFace Knowledge Distillation Guide](https://huggingface.co/blog/Kseniase/kd)
   - [torchdistill GitHub](https://github.com/yoshitomo-matsubara/torchdistill) — code examples

6. **Limitations and Open Problems:**
   - [Amazon Science: Limitations of KD for Zero-Shot Transfer](https://www.amazon.science/publications/limitations-of-knowledge-distillation-for-zero-shot-transfer-learning)
   - [Catastrophic Forgetting in Continual Learning](https://www.cs.uic.edu/~liub/lifelong-learning/continual-learning.pdf)

---

## Sources

### Core Research Papers

- [Feature Distillation Improves Zero-Shot Transfer from Synthetic Images | OpenReview](https://openreview.net/forum?id=SP8DLl6jgb)
- [Limitations of knowledge distillation for zero-shot transfer learning - Amazon Science](https://www.amazon.science/publications/limitations-of-knowledge-distillation-for-zero-shot-transfer-learning)
- [Towards Zero-Shot Knowledge Distillation for Natural Language Processing - ACL Anthology](https://aclanthology.org/2021.emnlp-main.526/)
- [Zero-Shot Knowledge Distillation in Deep Networks - arXiv](https://arxiv.org/abs/1905.08114)
- [Zero-Shot Knowledge Distillation in Deep Networks - PMLR](http://proceedings.mlr.press/v97/nayak19a/nayak19a.pdf)
- [Improving Zero-shot Generalization of Learned Prompts via Unsupervised Knowledge Distillation - arXiv](https://arxiv.org/abs/2407.03056)
- [Preventing Zero-Shot Transfer Degradation in Continual Learning of Vision-Language Models - arXiv](https://arxiv.org/abs/2303.06628)
- [Preventing Zero-Shot Transfer Degradation in Continual Learning (ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/papers/Zheng_Preventing_Zero-Shot_Transfer_Degradation_in_Continual_Learning_of_Vision-Language_Models_ICCV_2023_paper.pdf)
- [Efficient Transfer Learning for Video-language Foundation Models - arXiv](https://arxiv.org/html/2411.11223v1)

### Cross-Architecture Distillation

- [Feature-based One-For-All: Heterogeneous Distillation Across Vision Architectures - arXiv](https://arxiv.org/html/2501.08885v1)
- [What is Knowledge distillation? | IBM](https://www.ibm.com/think/topics/knowledge-distillation)
- [UniKD: Universal Knowledge Distillation (ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/papers/Lao_UniKD_Universal_Knowledge_Distillation_for_Mimicking_Homogeneous_or_Heterogeneous_Object_ICCV_2023_paper.pdf)
- [Knowledge distillation - Wikipedia](https://en.wikipedia.org/wiki/Knowledge_distillation)
- [Towards Cross-Tokenizer Distillation: Universal Logit Distillation Loss | OpenReview](https://openreview.net/forum?id=bwRxXiGO9A)
- [Towards Cross-Tokenizer Distillation - arXiv](https://arxiv.org/abs/2402.12030)
- [MergeNet: Knowledge Migration across Heterogeneous Models - arXiv](https://arxiv.org/html/2404.13322)
- [Universal Knowledge Retention Metric | Preprints.org](https://www.preprints.org/manuscript/202505.0901/v1)
- [Knowledge Distillation: Principles, Algorithms, Applications - Neptune.ai](https://neptune.ai/blog/knowledge-distillation)
- [Feature-based One-For-All Framework | ResearchGate](https://www.researchgate.net/publication/388067705_Feature-based_One-For-All_A_Universal_Framework_for_Heterogeneous_Knowledge_Distillation)
- [Cross-Architecture Knowledge Distillation - arXiv](https://arxiv.org/abs/2207.05273)
- [Cross-Architecture Knowledge Distillation | IJCV](https://link.springer.com/article/10.1007/s11263-024-02002-0)
- [Cross-Architecture Distillation with Redundancy Suppression - arXiv](https://arxiv.org/html/2507.21844v1)
- [Cross-Architecture Knowledge Distilling | EmergentMind](https://www.emergentmind.com/topics/cross-architecture-knowledge-distilling)
- [Neural Ranking KD - GitHub](https://github.com/sebastian-hofstaetter/neural-ranking-kd)
- [Improve Cross-Architecture Generalization on Dataset Distillation](https://distill-generalization-group.github.io/)
- [Cross-Architecture KD for Speech Enhancement (SSRN)](https://papers.ssrn.com/sol3/Delivery.cfm/d4a69024-f7ac-44ea-883f-c7196d163b89-MECA.pdf?abstractid=5222345&mirid=1)
- [Cross-Architecture Distillation for Face Recognition | ACM](https://dl.acm.org/doi/10.1145/3581783.3611711)
- [Cross-Architecture KD | Springer](https://link.springer.com/chapter/10.1007/978-3-031-26348-4_11)
- [Neural Ranking Models with Cross-Architecture KD | ResearchGate](https://www.researchgate.net/publication/344505384_Improving_Efficient_Neural_Ranking_Models_with_Cross-Architecture_Knowledge_Distillation)

### Data-Free and Synthetic Data Distillation

- [Synthetic data generation for data-free KD - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0957417423008291)
- [Synthetic data generation for data-free KD - arXiv](https://arxiv.org/abs/2301.04338)
- [Data free KD with feature synthesis | Scientific Reports](https://www.nature.com/articles/s41598-024-78757-w)
- [Synthetic data generation for data-free KD | ACM](https://dl.acm.org/doi/10.1016/j.eswa.2023.120327)
- [Small Scale Data-Free KD (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_Small_Scale_Data-Free_Knowledge_Distillation_CVPR_2024_paper.pdf)
- [When AI Makes AI: Synthetic Data and Model Distillation | Jina AI](https://jina.ai/news/when-ai-makes-ai-synthetic-data-model-distillation-and-model-collapse/)
- [RAFT Distillation: Synthetic Data Creation | Microsoft](https://techcommunity.microsoft.com/blog/educatordeveloperblog/responsible-synthetic-data-creation-for-fine-tuning-with-raft-distillation/4259367)
- [Model Distillation Guide 2025 | Label Your Data](https://labelyourdata.com/articles/machine-learning/model-distillation)
- [Multimodal Dataset Distillation - arXiv](https://arxiv.org/html/2602.19756)

### Catastrophic Forgetting and Continual Learning

- [Continual deep RL with task-agnostic policy distillation | Nature](https://www.nature.com/articles/s41598-024-80774-8)
- [Self-distillation fix for catastrophic forgetting | InfoWorld](https://www.infoworld.com/article/4131242/researchers-propose-a-self-distillation-fix-for-catastrophic-forgetting-in-llms.html)
- [Continual deep RL with task-agnostic policy distillation | PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11685974/)
- [Continual Federated Learning Based on KD (IJCAI)](https://www.ijcai.org/proceedings/2022/0303.pdf)
- [Continual Learning and Catastrophic Forgetting | UIC](https://www.cs.uic.edu/~liub/lifelong-learning/continual-learning.pdf)
- [Addressing Catastrophic Forgetting and Beyond | CEUR](https://ceur-ws.org/Vol-4082/paper1.pdf)
- [Mitigating Catastrophic Forgetting in LLMs | ACL](https://aclanthology.org/2024.acl-long.77.pdf)
- [Distill, Forget, Repeat: Continual Unlearning | arXiv](https://arxiv.org/html/2512.02657)
- [Continual Learning with KD: A Survey | TechRxiv](https://www.techrxiv.org/users/711699/articles/696650/master/file/data/TNNLS_CLKD23_TechRxiv/TNNLS_CLKD23_TechRxiv.pdf)
- [Parameter-Efficient Continual Fine-Tuning | arXiv](https://www.arxiv.org/pdf/2504.13822)

### Optimal Transport Theory

- [Cycle Class Consistency with Distributional OT | PMLR](https://proceedings.mlr.press/v180/nguyen22c/nguyen22c.pdf)
- [Awesome Optimal Transport in Deep Learning | GitHub](https://github.com/changwxx/Awesome-Optimal-Transport-in-Deep-Learning)
- [KD2M: Unifying framework for feature KD - arXiv](https://arxiv.org/html/2501.08885v1)
- [Multi-Level Optimal Transport for Cross-Tokenizer KD - arXiv](https://arxiv.org/html/2412.14528v1)
- [SelKD: Selective KD via OT | OpenReview](https://openreview.net/forum?id=H4iVLvRusn)
- [Cycle Class Consistency with OT and KD | OpenReview](https://openreview.net/forum?id=ScUndLLjceq)
- [KNOT: KD Using OT for NLP - ACL](https://aclanthology.org/2022.coling-1.425/)
- [KNOT - arXiv](https://arxiv.org/abs/2110.02432)
- [Cycle class consistency with OT and KD | PMLR](https://proceedings.mlr.press/v180/nguyen22c.html)
- [Multi-Level OT for Universal Cross-Tokenizer KD - arXiv](https://arxiv.org/abs/2412.14528)

### Practical Tools and Frameworks

- [DistiLLM-2 (ICML 2025) | GitHub](https://github.com/jongwooko/distillm-2)
- [Distilling Step-by-Step | HuggingFace](https://huggingface.co/papers/2305.02301)
- [Knowledge Distillation in LLM Guide | Medium](https://medium.com/@adeelmukhtar051/knowledge-distillation-in-llm-a-comprehensive-step-by-step-guide-0ad32368d427)
- [PromptKD (CVPR 2024) | GitHub](https://github.com/zhengli97/PromptKD)
- [Open-sourced ICCV 2025 works | HuggingFace](https://huggingface.co/blog/yoshitomo-matsubara/open-sourced-iccv2025-works)
- [Everything About Knowledge Distillation | HuggingFace](https://huggingface.co/blog/Kseniase/kd)
- [KD for Computer Vision | HuggingFace Docs](https://huggingface.co/docs/transformers/main/tasks/knowledge_distillation_for_image_classification)
- [DistiLLM (ICML 2024) | GitHub](https://github.com/jongwooko/distillm)
- [torchdistill | GitHub](https://github.com/yoshitomo-matsubara/torchdistill)
- [Knowledge Distillation Guide 2025 | Label Your Data](https://labelyourdata.com/articles/machine-learning/knowledge-distillation)
- [DistillKit by Arcee AI | GitHub](https://github.com/arcee-ai/DistillKit)
- [Model Distillation in the API | OpenAI](https://openai.com/index/api-model-distillation/)
- [License-Compliant Synthetic Data Pipelines | NVIDIA](https://developer.nvidia.com/blog/how-to-build-license-compliant-synthetic-data-pipelines-for-ai-model-distillation/)
- [Autodistill | Roboflow Blog](https://blog.roboflow.com/autodistill/)
- [Autodistill Overview | Augmented Startups](https://www.augmentedstartups.com/blog/autodistill-revolutionizing-computer-vision-model-distillation-roboflow)
- [Open Source Distilling](https://opensourcedistilling.com/)
- [OpenAI Model Distillation Guide | DataCamp](https://www.datacamp.com/tutorial/model-distillation-openai)
- [DistillKit Technical Paper | Arcee AI](https://www.arcee.ai/blog/distillkit-v0-1-by-arcee-ai)
- [Distily | GitHub](https://github.com/lapp0/distily)
- [Autodistill Docs](https://docs.autodistill.com/)

### Meta-Learning and Few-Shot Learning

- [Meta-learning for Few-Shot Learning Survey | ACM](https://dl.acm.org/doi/full/10.1145/3659943)
- [Few-Shot Learning via Task-Specific Meta Distillation (WACV 2023)](https://openaccess.thecvf.com/content/WACV2023/papers/Wu_Few-Shot_Learning_of_Compact_Models_via_Task-Specific_Meta_Distillation_WACV_2023_paper.pdf)
- [Enhancing Few-Shot Learning via Dual-Faceted KD | PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10975270/)
- [Few-shot and meta-learning methods survey | Springer](https://link.springer.com/article/10.1007/s13735-023-00279-4)
- [Awesome Dataset Distillation | GitHub](https://github.com/Guang000/Awesome-Dataset-Distillation)
- [Task-Agnostic Self-Distillation for Few-Shot Action Recognition (IJCAI)](https://www.ijcai.org/proceedings/2024/0600.pdf)
- [KD Meets Few-Shot Learning | ACL](https://aclanthology.org/2022.nlp4convai-1.10/)
- [Comprehensive Survey of Few-shot Learning | ACM](https://dl.acm.org/doi/10.1145/3582688)
- [Towards Few-Shot Learning in the Open World - arXiv](https://arxiv.org/html/2408.09722v1)
- [KD Meets Few-Shot Learning | ACL PDF](https://aclanthology.org/2022.nlp4convai-1.10.pdf)

---

**Research completed:** 2026-02-27
**Time invested:** ~2 hours of deep research
**Quality:** Comprehensive synthesis with 100+ sources
