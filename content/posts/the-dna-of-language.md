---
title: "The DNA of Language: A Deep Dive into LLM Tokenization concepts"
date: 2026-01-05T00:23:31+05:30
draft: true
tags: ["nlp", "llm", "tokenization"]
categories: ["tech", "deep-learning"]
---
Imagine you have to build a house. You cannot build a stable house only with massive boulders as walls (too big) or one cannot build a house only with tiny pebbles (too small). We need exactly the right sized bricks. Same analogy applies for linguistics. We need to figure out strategies to break down petabytes of data into usable atomic chunks.  

The the context of Large Language Models (LLMs), these bricks are called tokens. Tokens enable us to view a sizable amount of fluid language data into discrete mathematical language which machines can process. It is the invisible filter in the heart of LLMs where every prompt is passed and every response is born. Let's dive into the most popular tokenization strategies used in LLMs.

### 1. Byte Pair Encoding (BPE)

It is the most popular tokenization strategy used in LLMs mostly because it made the engineers worry less about confronting the problem of missing tokens. It recieved massive popularity after the release of GPT-2, GPT-3 and Llama and most important, it is a simple bottoms up frequency based strategy.

### How does it work?
BPE starts with a vocabulary of single characters and then iteratively merges the most frequent pairs of characters until it reaches the desired vocabulary size. This creates a tree like subword joining structure. 
By favouring frequent sequences, BPE ensures that the most common words and phrases are represented by a single token, while rarer words and phrases are represented by a combination of tokens. This makes BPE a powerful tool for representing natural language data in a compact and efficient way. 

The Byte level BPE goes even further to start the process with the UTF-8 byte as a character. This ensures that the tokenization process is language agnostic and can handle any language, even those languages that use non-latin scripts or cannot be split by spaces. 

![BPE Merge Process Diagram](/images/bpe-merge-process.png)

### 2. WordPiece
*To be continued...*