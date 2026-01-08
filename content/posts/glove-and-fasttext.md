---
title: "The Global Accountant and the Subword Surgeon: Decoding GloVe and FastText"
date: 2026-01-08T00:10:00+05:30
draft: false
tags: ["artificial-intelligence", "word-embeddings", "GloVe", "FastText"]
categories: ["artificial-intelligence", "embeddings"]
math: true
---

 *Imagine trying to teach a computer the difference between "Apple" the fruit and "Apple" the company. To us, the distinction is intuitive. To a machine, it’s just a string of characters. How do we turn these strings into meaningful math? While early attempts like Word2Vec gave us a great start, they missed the forest for the trees—or in some cases, the twigs for the branches. Enter GloVe and FastText: two algorithms that revolutionized how machines understand the nuances of human language.*

---

Previous static embedding models like Word2Vec successfully captured the local semantics of words, but they failed in addressing contexts between words that might not always appear in the same context window. Furthermore, they treated words as indivisible atomic units in continuous vector spaces, missing out on the internal structure of language.

The two simple but effective ideas behind GloVe and FastText are **subword units** and **global co-occurrence**. Let's dive into them.

### GloVe: A Global Mapping Strategy

**Global Vectors for Word Representation (GloVe)** was created at Stanford. The core idea was to train a model where word vectors are defined by how often they appear around other words across the *entire* corpus, rather than just locally.



#### Training Steps:

1.  **Co-occurrence Matrix `X`:** GloVe scans the entire corpus and creates a giant matrix (spreadsheet) recording the counts of how often words appear near each other.
2.  **Factorization:** It compresses this giant matrix into smaller, dense vectors.
3.  **Objective:** The function tries to equate the dot product of two word vectors with the logarithm of their probability of appearing together. The error between these values is then backpropagated to update the vectors.

$$w_i \cdot w_j + b_i + b_j = \log(X_{ij})$$

* $w$: The word vector (what we want).
* $X_{ij}$: The count from our matrix.

#### The Loss Function:

We can't satisfy the equation above perfectly for every word pair. So, we use a weighting function to minimize the squared error:

$$J = \sum_{i,j=1}^V f(X_{ij}) ({w_i}^T w_j + b_i + b_j - \log(X_{ij}))^2$$

If we do not use the weighting function, the loss would be infinite for rare words (since $\log(0) = \infty$) and frequent words (like 'the') would overpower the model.

$$
f(x) = 
\begin{cases} 
(x/x_{max})^\alpha & \text{if } x < x_{max} \\
1 & \text{otherwise} 
\end{cases}
$$

*(Typically $x_{max}=100$ and $\alpha=0.75$)*

---

### FastText: The Lego Bricks Strategy

**FastText** was created by Facebook AI Research (FAIR). The idea was to treat words not as solid rocks, but as structures made of "Lego bricks" (subword units). 

Word2Vec struggled to create embeddings for words it hadn't seen before (Out-Of-Vocabulary or OOV words). FastText solves this by breaking words down.

* GloVe treats a word like "Apple" as a single, atomic unit.
* FastText breaks "Apple" into character n-grams: `<ap, app, ppl, ple, le>`



For a complex or new word like "**Applesauce**", the model might not know the whole word, but it recognizes the Lego brick for "Apple" and the remaining bits from other subwords.

FastText takes its inspiration from the SkipGram architecture. The vector for a word is simply the sum of the vectors of its n-gram subwords:

$$v_w = \sum_{g \in G_w} z_g$$

#### Objective:

The objective is to predict context words given a target word.

$$P(w_c | w_t) = \frac{e^{s(v_{w_t} \cdot v_{w_c})}}{\sum_{w \in V} e^{s(v_{w_t} \cdot v_w)}}$$

And the scoring function is defined as:

$$s(w_t, w_c) = u_{w_t}^T (\sum_{g \in G_w} z_g)$$

* $u_{w_c}$: The vector for the context word (output vector).
* $z_g$: The vector for the n-gram $g$ (input vector).

#### Modifications to Softmax:

Calculating the full softmax for every word in the vocabulary is computationally expensive. FastText uses two methods to approximate this efficiently.

**Option A: Negative Sampling**

We pick a few random "wrong" words every time to serve as "negative samples" and calculate the loss only for the correct word and these noise words.

$$J = - \log \sigma(s(w_t, w_c)) - \sum\limits_{i=1}^{N} \mathbb{E}_{n_i \sim P_n(w)} [\log \sigma(-s(w_t, n_i))]$$

* The sigmoid function $\sigma$ squashes the score between 0 and 1.
* The first term pushes the probability of the **correct** word towards 1.
* The second term pushes the probability of the **noise** words towards 0.

**Option B: Hierarchical Softmax (Faster for infrequent words)**

Instead of a flat list of words, imagine the vocabulary arranged as a binary tree (Huffman Tree).



* **Root:** Top of the tree.
* **Leaves:** The actual words.
* To calculate the probability of a word, we trace a path from the root to that leaf. At every branching node, we calculate the probability (sigmoid) of going left or right.
* The probability of a word $w$ is the product of the probabilities of the turns taken to reach it.

### Code Example

```python
from gensim.models import FastText
from gensim.test.utils import common_texts

# Initialize the FastText model with specific hyperparameters
model = FastText(
    vector_size=100,    # Dimensionality of the word vectors
    window=5,           # Maximum distance between the current and predicted word within a sentence
    min_count=1,        # Ignores all words with total frequency lower than this
    sg=1,               # Training algorithm: 1 for skip-gram; 0 for CBOW
    hs=0,               # If 0, and negative is non-zero, negative sampling will be used
    negative=5,         # Number of "noise words" to be drawn for negative sampling
    workers=4,          # Number of worker threads to train the model
    epochs=10,          # Number of iterations over the corpus
    min_n=3,            # Minimum length of character n-grams
    max_n=6             # Maximum length of character n-grams
)

# Build the vocabulary from the provided text corpus
model.build_vocab(common_texts)

# Train the model on the corpus
model.train(
    common_texts, 
    total_examples=model.corpus_count, 
    epochs=model.epochs
)

# Calculate and print the cosine similarity between two words in the vocabulary
print(f"Similarity between 'computer' and 'human': {model.wv.similarity('computer', 'human')}")

# Demonstrate FastText's ability to handle Out-Of-Vocabulary (OOV) words
# Even if 'computation' wasn't in the training data, FastText constructs a vector using character n-grams
oov_vector = model.wv['computation']

# Check if a specific word exists in the model's fixed vocabulary index
word = "computer"
is_in_vocab = word in model.wv.key_to_index
print(f"\nIs the word '{word}' in the model's vocabulary? {is_in_vocab}")