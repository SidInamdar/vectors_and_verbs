---
title: "The Geometry of Meaning: Sine, ALiBi, RoPE, and HoPE"
date: 2026-01-09T21:45:00+05:30
draft: false
tags: ["nlp", "transformers", "deep-learning", "math"]
categories: ["tech"]
math: true
---

![Positional Embeddings Map](/images/positional-embeddings-header.png)

Positional Embeddings are the "voice" that tell Transformers *where* words are in a sentence. Without them, there are no mathematical principles by which LLMs can differentiate sentences with alternate arrangements of words. 

"The dog chased the cat" and "cat chased the dog" are mathematically indistinguishable to a Transformer without positional embeddings, as self-attention treats them as a chaotic bag of marbles. Here is an in-depth breakdown of the major positional encoders that solved this problem. The key ingredient in these algorithms is the delicate balance of adding information to word embeddings without polluting the semantic vectors.

### 🔥 Sinusoidal Embeddings: The Odometer of Language

Introduced in the seminal "Attention Is All You Need" paper, this was the industry standard for many years. This approach adds a vector to each word embedding based on its position. The vector is computed out of wavelike patterns that are periodic and smooth. Mathematically, for a position $pos$ and embedding dimension $i$:

$$
PE_{(pos, i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

The model uses sine and cosine waves of different frequencies to capture both short and long-range dependencies. The frequency of the waves decreases as the embedding dimension increases, which helps capture long-range dependencies in the input sequence.

The creators wanted the model to calculate distance easily, which is why a linear function was used. The model simply rotates the current position vector using a rotation matrix to derive the next position.

<div>
$$
\begin{pmatrix}
\cos(k) & \sin(k) \\
-\sin(k) & \cos(k)
\end{pmatrix} \cdot PE(pos) \approx PE(pos + k)
$$
</div>

**Low dimensions** act like a fast-ticking clock (high frequency, changing rapidly) and **high dimensions** act like a slow clock (low frequency, changing slowly).

The number **10,000** is an arbitrarily chosen high number to reduce **Aliasing**. By choosing such a high number, the creators ensured that the wavelength of the slowest dimension is incredibly long. Consequently, the pattern practically never repeats, and each position remains unique.

This combination works like a **mechanical odometer dial**. Each subsequent dimension acts as a slower-moving gear. The model learns to read these "dials" to understand exactly where a word sits in the sequence.

This mitigated the challenge of exploding gradients that previous array-based positional embeddings suffered from (e.g., adding integers 1, 2, 3...). This model keeps values between -1 and +1, adding stability while ensuring every position gets a distinct code.

> **The Problem:** This approach pollutes the word embedding matrix (semantic information). While high-dimensional vectors are almost always orthogonal—meaning the noise doesn't fully destroy the word's identity—it does not strictly *guarantee* purity.

This led to a more mature approach.

### 🔥 ALiBi: The Fading Streetlight

**Attention with Linear Biases (ALiBi)** is a method where, instead of adding a vector to the embeddings at the start, the model injects positional information linearly, directly into the attention scores. If Sinusoidal Encoding asks, "Where am I on the map?", ALiBi asks, "How much harder should I squint to see you?"

The biggest problem ALiBi solved is the **"Invisible Wall"**. If you train a Transformer on sequences of length 2048 and then feed it a book with 3000 words, it will crash or output garbage at word 2049. This is a failure of **Extrapolation**.

ALiBi throws away the idea of adding positional vectors entirely. Instead, it devises a "penalty" system for attention scores:

$$
\text{Score} = (q \cdot k^\top) - (m \cdot \text{distance})
$$

**Distance:** If words A and B are neighbors, the distance is 0, so the penalty is 0. Although the distance is linear, it does not cause exploding gradients because the raw attention scores are normalized by a **softmax** function. This distance prioritizes nearby words (local context) over distant ones.

**Slope ($m$):** If we always penalize distance equally, the model becomes myopic. ALiBi mitigates this by assigning a different slope value to each attention head:

*   **Head 1 (The Historian):** Very low slope. The penalty grows slowly, allowing it to see far back.
*   **Head 2 (The Reader):** Medium slope.
*   **Head 3 (The Myopic):** Very high slope (e.g., 0.5); it effectively cannot see beyond 2-3 words.

This mimics holding a **lantern**. Words close to you are bright, while words further away naturally fade into darkness.

**Extrapolation:** If you walk 5 miles down the road (a longer sequence), your lantern works exactly the same way. The physics of light decay are constant, allowing the model to handle sequences longer than it was trained on.

However, ALiBi forces a "sliding window" view where distant words eventually become invisible. This brings us to the modern standard.

### 🔥 RoPE: The Rotary Revolution

**Rotary Position Embeddings (RoPE)** is the current industry standard (used in Llama 3, Mistral, PaLM) because it mathematically unifies "absolute position" with "relative position." This method **rotates** the vectors in the Query and Key matrices based on their absolute position.

If a vector has 512 dimensions, RoPE breaks it into 256 pairs. Each pair is rotated by a different frequency:

$$
\theta_i = 10000^{-2i/d}
$$

The **First pair ($i = 0$)** is rotated by the highest frequency. The **Last pair ($i = 255$)** is rotated by the lowest frequency. This spread ensures the model has "high precision" (fast rotation) to distinguish immediate neighbors, and "low precision" (slow rotation) to track global position without the pattern repeating.

**Doesn't rotation change meaning?** Yes and No. The vector for "Apple" at position 1 and "Apple" at position 256 will look completely different in terms of coordinates. However, rotation does **not** change the *semantic strength* (magnitude/norm) of the vector.

**The Relativity Trick:** Transformers compare words using dot products (angles). RoPE relies on the fact that the attention mechanism ignores the absolute rotation and only detects the **difference** in rotation (relative position). Thus, the "relative meaning" is encoded purely into the angle between words, leaving the semantic "magnitude" untouched.

### 🔥 LongRoPE: The Bifocal Lens

While RoPE is powerful, it struggles when stretching context windows from 4k to 2 million tokens. The "fast" dimensions spin so rapidly over long distances that they become random noise. **LongRoPE** solves this using an evolutionary search algorithm to find a unique scaling factor ($\lambda$) for each dimension.

### The Equations

Instead of rotating by the standard $\theta_i$, LongRoPE rotates by a rescaled frequency:

$$
\theta'_i = \lambda_i \theta_i
$$

To efficiently find these $\lambda$ values without searching an infinite space, the algorithm enforces a **monotonicity constraint** based on NTK theory:

$$
\lambda_i \le \lambda_{i+1}
$$

This ensures that low-frequency dimensions (global context) are stretched more than high-frequency dimensions (local context), creating a "bifocal" effect:
1.  **High Frequencies (Local):** Kept sharp ($\lambda \approx 1$) to maintain grammar.
2.  **Low Frequencies (Global):** Stretched ($\lambda > 1$) to track massive distances without repeating.
Not worth perusing more since the next one is much better at long context positional encoding. 

### 🔥 HoPE: The Hyperbolic Slide

**Hyperbolic Rotary Positional Encoding (HoPE)** moves from the geometry of a circle to the geometry of a hyperbola (inspired by Lorentz transformations). It was designed to fix a subtle flaw in RoPE: the "wobbly" or oscillatory nature of attention scores.

#### The Core Problem: RoPE's "Wobble"

In standard RoPE, we rotate vectors around a circle.
*   **The Issue:** Circles repeat. A dial at $361^\circ$ looks identical to $1^\circ$. Even with different frequencies, the dot product fluctuates up and down as distance increases.
*   **The Consequence:** The model gets confused at long distances, potentially mistaking a word 1,000 tokens away for a neighbor just because the rotation cycles aligned, creating "noise."

#### The "Hyperbolic" Shift

HoPE replaces trigonometric functions ($\sin, \cos$) with hyperbolic functions ($\sinh, \cosh$).
*   **Circular Rotation:** Keeps distance from the center constant.
*   **Hyperbolic Rotation:** Moves points along a hyperbola. Crucially, **hyperbolic functions do not repeat**; they grow or decay exponentially.

By using hyperbolic rotation, HoPE guarantees that as words get further apart, the attention score **decays monotonically** (smoothly drops).

#### The Equations

HoPE uses a hyperbolic matrix $B(\theta, m)$ defined using the Lorentz Boost structure:
<div>
$$
B(\theta, m) = \begin{pmatrix} \cosh(m\theta) & \sinh(m\theta) \\ \sinh(m\theta) & \cosh(m\theta) \end{pmatrix}
$$
</div>
Since hyperbolic functions grow to infinity, HoPE introduces a decay penalty ($e^{-m\theta'}$). The final attention score simplifies beautifully to a function of relative distance:

$$
\text{Score} \propto e^{-|m-n|(\theta' - \theta)}
$$

This ensures the attention score **exponentially decays** as distance increases, eliminating the wobbles of standard RoPE. This the latest model that is taking up the space in the industry for extremely long context models. 

