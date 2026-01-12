---
title: "Crafty Patchwork: Parameter-Efficient Fine-Tuning"
date: 2026-01-12T10:52:56+05:30
draft: false
tags: ["artificial-intelligence", "large-language-models", "fine-tuning", "peft", "lora", "qlora"]
categories: ["artificial-intelligence", "large-language-models"]
math: true
summary: "An introduction to Parameter-Efficient Fine-Tuning (PEFT) techniques like LoRA, QLoRA, and more."
---

The typical modern trajectory of creating intelligent models involves adding scores of GPUs and terabytes of data. This high barrier to entry is a significant limitation, creating a "wall" between common engineers and the democratization of AI. 

Models that require billions of parameters to function for even a single training pass often cannot fit on consumer hardware. Although full training may be out of reach, we can still nudge these models in desired directions—a process known as **Fine-Tuning**. 

In the traditional sense, **Full Fine-Tuning (FFT)** was performed on earlier models where every parameter was updated during backpropagation. However, for a model like Llama 3 with 70B parameters, FFT is computationally prohibitive. Beyond the parameters themselves, the model must store gradients, optimizer states, and activation caches, which can dwarf the memory footprint of the original weights. This is where **Parameter-Efficient Fine-Tuning (PEFT)** techniques come in.

### The Geometry of Adaptation

After navigating the loss landscapes of model parameters, researchers discovered that LLMs are often over-parameterized. The solutions to specific tasks frequently lie in a much lower-dimensional manifold or "subspace" of the parameter space. This is referred to as the **Intrinsic Dimension** of the objective function—the minimum number of degrees of freedom required to reach a satisfactory solution.

During initial training, a model learns vast, general representations of language, requiring high dimensionality to capture the nuances of world knowledge. However, when fine-tuning for a specific task (such as adapting a model to write SQL code), the "delta" or required change in weights is not random. The model only needs to adjust a few dimensions of its subspace to master the new task.

This realization is the key behind the success of PEFT. In fact, the larger the model, the less "steering" effort is needed during fine-tuning. Mathematically, traditional updates look like this:

$$h = W_0x$$
$$W_{\text{tuned}} = W_0 + \Delta W$$
$$h = W_0x + \Delta Wx$$

Here, $W_0$ and $\Delta W$ share the same dimensions. However, according to the intrinsic dimension hypothesis, $\Delta W$ is **rank-deficient**, meaning it contains far fewer unique dimensions than its total size would suggest. This concept—that $\Delta W$ can be represented more compactly—is the cornerstone of LoRA.

### Low-Rank Adaptation (LoRA)

LoRA converts the theoretical concept of intrinsic dimension into a practical architecture. Instead of training $\Delta W$ directly, we decompose it into two smaller, trainable matrices, $A$ and $B$:

$$\Delta W = BA$$

- **Matrix B**: A trainable matrix with dimensions $d \times r$.
- **Matrix A**: A trainable matrix with dimensions $r \times k$.

The **rank $r$** is a hyperparameter that determines the dimensionality of the update. For minor refinements (like improving linguistic nuance), a rank of $r=4$ is often sufficient. For more complex tasks (like learning a new topic), $r=8$ or higher might be used.

Because $r \ll \min(d, k)$, the number of parameters in $A$ and $B$ is vastly smaller than in $\Delta W$.

#### Example: Memory Savings
Consider a weight matrix $W$ with dimensions $4096 \times 4096$:
- **Full $\Delta W$**: $4096 \times 4096 = 16,777,216$ parameters.
- **LoRA ($r = 8$)**:
    - Matrix $A$: $8 \times 4096 = 32,768$
    - Matrix $B$: $4096 \times 8 = 32,768$
    - **Total Parameters**: $65,536$

**Reduction Percentage**:
$$\frac{65,536}{16,777,216} \times 100 \approx 0.39\%$$

The number of trainable parameters is reduced by over 99%.

#### The Alpha Scaling Factor
The LoRA update is typically scaled:
$$h = W_0 x + \frac{\alpha}{r} (B A x)$$

$\alpha$ (Alpha) is a scaling constant, similar to a learning rate, that amplifies the signal of the adapter. Dividing by $r$ ensures that if you change the rank during experimentation, you don't need to drastically re-tune $\alpha$. It keeps the update magnitude roughly constant—much like steering a giant ship using a small but precisely controlled boat.

#### Initialization Strategy
If the added parameters ($A$ and $B$) are initialized with random values, the initial gradients would be chaotic, harming performance. Instead:
- **Matrix A** is initialized with a random Gaussian distribution (small random numbers).
- **Matrix B** is initialized to zero.

At the start of training, $\Delta W$ (which is $B \times A$) is exactly zero. This ensures the model starts with its original performance and gradually adapts.

#### VRAM and Latency
$$VRAM \approx \text{Base Model Size} + \text{Optimizer States for LoRA Parameters}$$

For a 70B model, FFT might require hundreds of gigabytes of VRAM. With LoRA, the memory overhead is significantly reduced. 

Furthermore, LoRA offers **Zero Latency Inference**. Unlike "Adapter Layers" which insert new physical layers into the network, LoRA weights can be mathematically merged into the base weights ($W_{\text{new}} = W_0 + BA$) after training. This means no extra operations are required during inference.

---

### QLoRA: Breaking the Memory Wall with Quantization

Quantized LoRA (QLoRA) addresses the model weight bottleneck. It reduces the floating-point precision of the original model weights via quantization while keeping the LoRA parameters in full precision.

Quantization is the process of mapping high-precision floating-point numbers to lower-precision integers. While standard 4-bit quantization (Int4) evenly spaces values, neural network weights typically follow a Normal distribution, clustering near zero. QLoRA solves this with **NF4 (NormalFloat 4)**: it uses quantile quantization where each bin holds an equal probability mass of the distribution. 

This innovation allows a 70B model to fit into roughly 35–40 GB of VRAM, making it possible to fine-tune on a single 48GB workstation card (like an RTX 6000 Ada) or two consumer 24GB cards.

#### Double Quantization
In quantization, we use "quantization constants" (scaling factors) to map small integers back to their approximate real values. To maintain accuracy, weights are quantized in blocks (e.g., 64 weights per block), each with its own 32-bit constant. QLoRA introduces **Double Quantization**, which quantizes the constants themselves.

1. **First Pass**: Quantize weights to 4-bit, producing 32-bit constants.
2. **Second Pass**: Quantize those 32-bit constants into 8-bit floats.

This "inception-style" approach reduces the memory overhead of the constants from ~4GB to ~0.1GB. In the tight confines of GPU memory, every gigabyte counts.

#### The Compute-Memory Trade-off
QLoRA is a marvel of efficiency, but it introduces a trade-off in compute latency. Standard hardware cannot perform computation directly on 4-bit tensors, so QLoRA uses **On-The-Fly (OTF) Dequantization**. Weights are dequantized to 16-bit for the calculation and then discarded. This makes QLoRA roughly 30% to 50% slower than standard LoRA.

---

### DoRA: Weight-Decomposed Low-Rank Adaptation

**DoRA** is a recent advancement that addresses mathematical caveats in standard LoRA. Researchers found that LoRA doesn't always match the "learning style" of full fine-tuning because magnitude and direction updates are coupled.

In vanilla LoRA, changing the direction of the weights often accidentally changes their magnitude as well. It’s like driving a car where turning the steering wheel also forces you to accelerate. DoRA decouples these two components.

#### The Mathematics of DoRA
DoRA re-parametrizes weight updates by separating the magnitude ($m$) and the direction ($V$):

$$W = m \frac{V + \Delta V}{||V + \Delta V||}$$

#### Example: The DoRA Advantage
Imagine a weight vector with values $[3, 4]$.

1.  **The LoRA Way (Coupled)**: You add an update $\Delta W = [1, 1]$.
    $$W_{\text{new}} = [3, 4] + [1, 1] = [4, 5]$$
    The ratio changed (from 3:4 to 4:5), but the length (magnitude) jumped from 5 to $\approx 6.4$. If the model only wanted to change the "concept" (direction) without becoming "louder" (magnitude), it couldn't.

2.  **The DoRA Way (Decomposed)**: DoRA treats magnitude ($m = 5$) and direction ($V = [3, 4]$) separately.
    -   **Update Direction**: Add the LoRA update to $V$, getting $V' = [4, 5]$.
    -   **Normalize**: "Re-normalize" $V'$ so its length is 1 again ($\approx [0.62, 0.78]$).
    -   **Apply Magnitude**: Multiply that pure direction by the magnitude $m$ (which might stay 5).
    $$W_{\text{final}} = 5 \times [0.62, 0.78] = [3.1, 3.9]$$

By giving magnitude and direction their own degrees of freedom, DoRA mimics Full Fine-Tuning more closely, often achieving superior results with the same number of parameters.
