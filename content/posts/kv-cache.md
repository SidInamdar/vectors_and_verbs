---
title: "The Need For Speed: KV Cache and memory optimization at Inference"
date: 2026-01-13T10:52:41+05:30
draft: false
tags: ["artificial-intelligence", "large-language-models", "inference", "kv-cache", "transformer", "vLLM", "SGLang", "RadixAttention"]
categories: ["artificial-intelligence", "large-language-models"]
math: true
summary: "An introduction to KV Caching and its role in optimizing Transformer inference."
---
Transformer architecture is recursive; it has a self-referential structure. Unlike 'recurrent' networks which have a feedback loop at each stage, transformers process tokens entirely through all the layers and then feed the generated text back into the model. The model repeats this computation until generation stops—this is the **autoregressive** nature of modern LLMs. The entire network waits for a token to be processed in a particular layer, wasting time and resources. Furthermore, the model must remember the redundant computations of the past. For models intended to handle context windows of 128,000 tokens or more—such as GPT-4 or Llama-3—this redundancy would render real-time generation physically and economically impossible. Hence, caching is necessary to optimize the inference process.

KV Caching of attention layers is not a mere software optimization; it is a fundamental architectural component of the inference runtime. It allows the model to store the **Key** and **Value** pair states of past tokens in GPU memory.

For a sequence of tokens in autoregressive generation, the model's first token effectively "waits" until the final token output is generated. This creates $O(N^2)$ computational complexity if not optimized.

**Analogy:** Imagine a child memorizing chapters from a book.
*   **Normal Caching:** He memorizes the first chapter. To learn the second, he memorizes the first chapter *again* and then the second.
*   **KV Caching:** The child writes down a summary of the first chapter that is relevant to the second before moving on. However, this creates a **"Memory Wall"**: if enough chapters are introduced, the desk becomes cluttered with sticky notes, leaving no space to work. Similarly, KV Caching will consume all available GPU space if not managed properly.

---

### Hardware Reality: Bandwidth, Latency, and Memory

LLM Inference consists of two phases with radically different hardware characteristics:

**Phase 1: Prefill**
When a user sends a prompt (e.g., "Summarize this 5,000-word essay!") to the model.
*   **Operation:** The GPU performs massive matrix-matrix multiplications (GEMMs) by projecting embeddings against weight matrices.
*   **Hardware State:** CUDA cores are completely saturated; the bottleneck is how fast the chip can perform math.
*   **Arithmetic Intensity:** Very High. Operations are completed on all tokens simultaneously.
*   **KV Cache:** The model computes Key and Value states for all tokens and writes them to memory.

**Phase 2: Decode**
When the model generates tokens one by one.
*   **Operation:** The model has one token of input and must produce one token of output.
*   **Hardware State:** The GPU loads the entire KV cache context and the model weights just to generate a single token.
*   **Arithmetic Intensity:** Low. Only a few operations are needed compared to the massive amount of memory moved.
*   **The Memory Wall:** Compute cores sit idle while waiting for data to be loaded from memory (HBM).

$$
\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Bytes}}
$$

Assume: Weights = $M$ Bytes, KV Cache = $C$ Bytes. For $MN$ elements, Computation ($FLOPs$) $\approx 2MN$. Data Moved = $M$ (weights) + $C$ (KV Cache). Since $C$ is usually very small compared to $M$ during early generation:

$$
AI_{decode} \approx \frac{2MN}{M + C} \approx 2 \text{ OPS/Byte}
$$

H100 GPUs have a peak compute of ~1,000 TFLOPS but a bandwidth of $\approx 3.35 \text{ TB/s}$, resulting in an $AI \approx 300 \text{ OPS/Byte}$. This means the GPU runs at **less than 1% of its peak compute** during token generation. The speed of generation (tokens per second) is strictly limited by how fast we can stream the KV cache and weights from memory.

---
### Tensor Mechanics and Capacity Planning

KV Cache memory grows linearly with context length. In standard Multi-Head Attention (MHA), the KV Cache consists of two tensors per layer (Key and Value). For Batch Size $B$, Sequence Length $L$, Number of Heads $H$, and Head Dimension $D$, the total size is:

$$Memory_{KV} = 2 \times Layers \times N_{Heads} \times L \times D \times \text{Precision}$$

For example, a model like Llama-3 70B with a context of 100,000 tokens:

$$Memory_{token} = 2 \times 80 \times 8 \times 128 \times 2 = 327,680 \text{ bytes} \approx 0.32 \text{ MB per token}$$
$$\text{Total Cache} = 0.32 \text{ MB} \times 100,000 = 32,000 \text{ MB} = 32 \text{ GB}$$

This highlights the severity of the problem: a single large-context prompt can burn through 32GB of GPU memory. Architectural revolutions like **Grouped-Query Attention (GQA)** help by reducing the number of KV heads, which you can read about in [this article]({{< ref "attention-mechanisms.md" >}}).

However, even with GQA, using continuous memory blocks leads to a **fragmentation crisis**:
1.  **Internal Fragmentation (Over-provisioning):** Systems often reserve space for the *maximum* possible context length, locking away memory even if the request is short.
2.  **External Fragmentation:** Small, non-contiguous blocks of free memory cannot be used to store a single larger context.
3.  **Reservation Inefficiency:** Memory is often reserved for tokens that haven't been generated yet.

To solve this, modern runtimes implement techniques like **PagedAttention** and **RadixAttention**.

---
### vLLM and PagedAttention

vLLM is a fundamental re-architecture of LLM serving. It shifts from physical memory allocation to **virtual memory**, using a **PagedAttention** algorithm borrowed from Operating Systems.

**The OS Analogy:**
In classical memory management, a program "sees" large continuous blocks of RAM. In reality, the OS breaks data into small, fixed-size "pages" and scatters them wherever physical RAM is available. A **Page Table** maps logical requests to these physical addresses.

vLLM applies this to the KV Cache:
*   **Logical KV Blocks:** The sequence of tokens as the model "sees" them (continuous).
*   **Physical KV Blocks:** The actual storage locations in GPU HBM (scattered).
*   **Block Table:** The dictionary that maps the logical view to physical reality.

$$Address(K_t) = \text{BasePtr} + (P_{block} \times \text{BlockStride}) + (O_{idx} \times \text{TokenStride})$$

This allows tensors to be located anywhere in memory. A sequence logcially spanning tokens $[0...31]$ might be stored in Physical Blocks #4 and #99, which are gigabytes apart.

**Efficiency Gains:**
Pre-vLLM systems often wasted up to 80% of memory. In vLLM, waste is capped at $\frac{B-1}{\text{Tokens}}$ (where $B$ is block size). This abstraction also enables **Copy-on-Write** for parallel sampling: multiple outputs for one prompt can share the same physical blocks for the prompt's KV cache, only creating new blocks when they start to diverge.

---
### SGLang and RadixAttention

While vLLM solved the *spatial* problem of memory allocation, **SGLang** (Structured Generation Language) addresses the *structural* and *temporal* problems. LLM use cases often share massive amounts of prefix memory (e.g., a long system prompt used across many chats).

**RadixAttention: The Memory Tree**
SGLang views memory as a **Radix Tree**. It manages the KV cache at the level of token sequences rather than individual requests.

If a user explores a maze and hits a dead end, they reset. But if they have a "GPS" (Radix Tree), the system persists a map of every path ever walked. If a new user starts a path that has been explored before (e.g., "You are a helpful assistant..."), the system "teleports" them to the existing state in the cache.

*   **Nodes:** Represent a state of the KV Cache.
*   **Edges:** Sequences of tokens transitioning between states.
*   **Values:** Pointers to physical blocks of memory.

SGLang uses **LRU (Least Recently Used) eviction** policies to manage the finite GPU memory, pruning old branches or leaf nodes when space is needed for new prompts.

---
### Structured Generation: Output Taming

When requests demand specific formats (JSON, multiple choice, etc.), SGLang introduces **Compressed Finite State Machines (FSM)**. This restricts the model's vocabulary to ensure valid outputs.

1.  **Regex Compilation:** The user's constraint is compiled into an FSM graph.
2.  **State Tracking:** The system tracks the current state as tokens are generated.
3.  **Vocabulary Masking:** Invalid tokens are masked by manipulating logits ($z$):
<div>
$$z'_i = \begin{cases} z_i & \text{if } i \in S_{valid} \\ -\infty & \text{if } i \notin S_{valid} \end{cases}$$
</div>

Standard masking is expensive, so SGLang uses **Jump-Forward Decoding**: it determines deterministic characters (like `{}` in JSON) automatically and "jumps" over them without querying the model. This can make structured generation up to **3x faster**.