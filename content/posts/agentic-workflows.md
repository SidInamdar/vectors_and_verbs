---
title: "Electronic Executives: RAG, ReAct and MCP"
date: 2026-01-15T17:59:29+05:30
draft: false
tags: ["rag", "react-pattern", "mcp", "anthropic", "agentic-workflows", "llm-orchestration", "artificial-intelligence", "large-language-models"]
categories: ["artificial-intelligence", "large-language-models"]
math: true
summary: "A deep dive into the cognitive architectures of modern AI agents, exploring Retrieval-Augmented Generation (RAG), the ReAct reasoning pattern, and the Model Context Protocol (MCP)."
---

Researchers spent just under a decade perfecting the art of scaling Large Language Models (LLMs) to generate responses in real time. However, these tools were largely built and improved in isolation, simply iterating batches of tokens through the model. While some models developed exceptional reasoning capabilities, they were inherently limited by their architectural solitude. 

This isolation created significant hurdles, particularly in accessing real-time or proprietary data and the inability to perform tasks requiring external tools. This has led to the emergence of advanced toolsets designed to augment reasoning models within **agentic workflows**. These tools envelop the model in sophisticated cognitive and operational architectures: **Retrieval-Augmented Generation (RAG)** solves memory constraints, **ReAct** addresses execution constraints, and the **Model Context Protocol (MCP)** bridges the connectivity gap.

---

### Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation (RAG) is often misunderstood as a simple pipeline: "Search Database $\rightarrow$ Paste into Prompt $\rightarrow$ Generate." In reality, a more accurate description of RAG is **probabilistic marginalization**. RAG treats external information not as a mere preprocessing step, but as a latent variable in the generation process.

**The Retriever and the Generator**

The RAG architecture consists of two primary components:

1.  **The Retriever ($P_{\eta}$):** A dense bi-encoder that calculates the probability of a document $z$ being relevant to a query $x$.
    <div>
    $$P_\eta(z|x) = \frac{\exp(\text{sim}(q(x), d(z)))}{\sum_{z' \in \mathcal{Z}} \exp(\text{sim}(q(x), d(z')))}$$
    </div>
    Here, $q(x)$ and $d(z)$ are the dense vector representations of the query and document, respectively, typically produced by BERT-based encoders. The similarity function ($\text{sim}$) is usually the dot product.

2.  **The Generator ($P_{\theta}$):** An LLM that generates the target sequence $y$ based on the retrieved context.

#### RAG Sequence-Level Marginalization

This model assumes that each retrieved sequence $z$ contains sufficient information to generate the entire target sequence. The system retrieves the top-$K$ documents, generates complete sequences for each, and then aggregates the probabilities:
<div>
$$P_{\text{RAG-Sequence}}(y|x) \approx \sum_{z \in \text{top-k}(P(\cdot|x))} P_{\eta}(z|x) \prod_{i=1}^{N} P_{\theta}(y_i | x, z, y_{1:i-1})$$
</div>

In this approach, each document $z$ acts as a separate context for the duration of generation. This maintains consistency but limits the model's ability to synthesize disparate facts from multiple sources.

#### RAG Token-Level Marginalization

This model is more flexible but computationally intensive. It allows the generator to consult different documents while generating each subsequent token, enabling the model to shift its focus "mid-thought":
<div>
$$P_{\text{RAG-Token}}(y|x) \approx \prod_{i=1}^{N} \sum_{z \in \text{top-k}(P(\cdot|x))} P_{\eta}(z|x) P_{\theta}(y_i | x, z, y_{1:i-1})$$
</div>

Token-level RAG offers superior performance on multi-hop reasoning tasks but introduces a significant latency penalty. Calculating attention scores for $K$ documents for every token scales linearly with $K$. Most enterprise "Chat with Data" solutions currently favor **RAG Sequence-Level Marginalization**.

---

#### Dense Passage Retrieval (DPR)

The effectiveness of the retrieval term $P_{\eta}(z|x)$ is bounded by the quality of the vector space—specifically, the breadth and depth of the training corpus. **Dense Passage Retrieval (DPR)** utilizes a bi-encoder architecture where a model (like BERT) encodes an entire document into a single high-dimensional vector. Bi-encoders process documents independently, allowing for pre-computed document embeddings. (This technique is conceptually similar to Skip-Grams, replacing the feed-forward network with a BERT-like encoder and using entire sequences as input.)

**In-Batch Negatives and Loss Function**

Training a retriever Requires distinguishing between a correct document ($d^+$) and irrelevant ones ($d^-$). To avoid expensive negative mining for every training step, DPR uses **in-batch negatives**. 

In a training batch of $B$ samples $\{(q_i, d_i^+)\}_{i=1}^B$, the document $d_i^+$ paired with query $q_i$ is the positive sample. Documents paired with *other* queries in the same batch ($d_j^+$ where $j \neq i$) serve as the negative samples. This allows the model to train on $B(B-1)$ negative pairs efficiently. The loss function is the negative log-likelihood of the positive passage:
<div>
$$\mathcal{L}(q_i, d_i^+) = - \log \frac{e^{\text{sim}(q_i, d_i^+) / \tau}}{e^{\text{sim}(q_i, d_i^+) / \tau} + \sum_{j \neq i} e^{\text{sim}(q_i, d_j^+) / \tau}}$$
</div>

The temperature parameter $\tau$ scales the dot product. As $\tau \rightarrow 0$, the model's selection becomes sharper. Contrastive learning typically uses $\tau$ values between 0.05 and 0.1.

---

### Advanced RAG Strategies

**Self-RAG (Self-Reflective RAG)**
Self-RAG trains an LLM to generate special **Reflection Tokens** alongside standard text. These tokens act as internal control signals:
*   **Retrieve Token:** Determines if retrieval is necessary.
*   **IsRel Token:** Evaluates if the retrieved document is relevant.
*   **IsSup Token:** Checks if the generated answer is supported by the document.
*   **IsUse Token:** Assesses if the answer is useful for the query.

**Corrective RAG (CRAG)**
CRAG employs a lightweight, independent "Evaluator" component that assigns a confidence score to retrieved documents:
*   **High Confidence:** Passed directly to the generator.
*   **Ambiguous Confidence:** Triggers "Knowledge Refinement," decomposing the document to isolate relevant sequences.
*   **Low Confidence:** Triggers an external web search (in "Open World" agents) or falls back to standard generation.

**Adaptive RAG**
This approach adjusts its strategy based on query complexity. Simple queries require no retrieval; moderately complex queries use single-shot retrieval; highly complex queries trigger multi-hop or iterative retrieval processes.

---

### ReAct Pattern: Reasoning and Acting

The **ReAct** (Reason + Act) pattern provides a unified paradigm for LLMs to interact with external tools. While early models were often either "thinkers" (reasoning in isolation) or "doers" (following simple commands), ReAct allows for an interweaving of both.

**Addressing Semantic Drift**
A significant drawback of **Chain of Thought (CoT)** is "semantic drift"—if a model hallucinates at Step 1, every subsequent step treats that error as ground truth. ReAct mitigates this by forcing the model to interact with the external environment to ground its reasoning.

We define a trajectory $\tau$:
$$\tau = (s_1, a_1, o_1, s_2, a_2, o_2, \ldots, s_n, a_n, o_n)$$

Where:
*   $s_t \in \mathcal{L}$ is the **Thought** (a reasoning trace in language space).
*   $a_t \in \mathcal{A}$ is the **Action** (a specific tool invocation).
*   $o_t \in \mathcal{O}$ is the **Observation** (feedback from the environment, e.g., an API result).

The objective is to maximize the probability of a successful trajectory. The probability of an action $a_t$ is conditioned on the entire history:
$$P(a_t | h_t) = P(a_t | s_1, a_1, o_1, \dots, s_t)$$

**The Chain Rule of Reasoning**
Unlike standard Reinforcement Learning, where a policy $\pi$ maps states directly to actions ($\pi: S \rightarrow A$), ReAct decomposes the process:
1.  **Reasoning Step:** $P(s_t | h_t)$ — The model analyzes the state and plans.
2.  **Action Step:** $P(a_t | h_t, s_t)$ — The model executes an action based on its reasoning.
3.  **Observation Step:** $P(o_t | h_t, a_t)$ — The environment returns feedback.

---

### Model Context Protocol (MCP)

While we will cover this in depth in a future article, here is a brief summary. The **Model Context Protocol (MCP)** is an open-standard protocol designed to maintain consistent context across different generations. It solves the $N \times M$ integration problem—where $N$ models need to work with $M$ tools—by standardizing the interface.

Introduced as a standardization effort by Anthropic, MCP is an application-layer protocol built on **JSON-RPC 2.0**. It is transport-agnostic and defines three distinct roles: **Host**, **Client**, and **Server**.

**Request Object (Client $\to$ Server):**
```json
{
  "jsonrpc": "2.0",
  "id": 42,
  "method": "tools/call",
  "params": {
    "name": "get_customer_data",
    "arguments": { "customer_id": "C-12345" }
  }
}
```

**Response Object (Server $\to$ Client):**
```json
{
  "jsonrpc": "2.0",
  "id": 42,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Customer data retrieved: ..."
      }
    ]
  }
}
```
