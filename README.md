# Project 1 Design Spec: The Unofficial Guide

## Domain
**Off-Campus Housing Experiences**
This domain captures student-generated knowledge, unwritten logistics, and unfiltered reviews of apartment complexes near the campus (such as the Enclave). This information is highly valuable because official housing handbooks and complex websites hide negative realities like maintenance delays, utility bill spikes, and heavy commute traffic.

## Documents
* **Source Dataset:** 10 collected unstructured student review logs, local community forum threads, and neighborhood testimony files detailing lease terms, maintenance histories, and local travel times.

## Chunking Strategy
* **Strategy:** Fixed-character sliding window.
* **Chunk Size:** 500 characters.
* **Overlap:** 100 characters.
* **Reasoning:** Student reviews are typically dense but short. A 500-character window ensures individual complaints or data points stay fully self-contained in a single chunk, while the 100-character overlap prevents critical boundaries (like an apartment name or rent figure) from being split awkwardly between adjacent vectors.

## Retrieval Approach
* **Embedding Model:** `all-MiniLM-L6-v2` via `sentence-transformers` (Local deployment).
* **Top-K Value:** `k=4`
* **Production Tradeoff Reflection:** For a live production deployment, we would balance local latency vs. cloud API costs. While a hosted OpenAI or Cohere embedding model offers larger token context windows and stronger multilingual support, a local model like `all-MiniLM` eliminates external API costs entirely and ensures sub-millisecond local retrieval speeds.

## Evaluation Plan
1. What is the average price of rent in the area for student layouts?
2. What do students say about the commute and traffic timings from the Dania Beach campus to the Enclave?
3. What do reviews say about how quickly management fixes AC units during early summer?
4. How much do utility and electric bills typically spike during the summer months?
5. What are the late-night food launch/lunch delivery options near the dorms?

## Anticipated Challenges
1. **Noisy Text Data:** Raw student reviews often include erratic punctuation, emojis, and slang that can distort embedding vectors.
2. **Context Fragmentation:** If a user's question relies on two separate parts of a lease review, a small top-k value might miss the second half of the context.

## AI Tool Plan
* We will utilize Claude to help generate the modular pipeline architecture blocks. We will input our character-based chunking boundaries into the prompt to generate the clean text split loops for `ingest.py` and provide the persistent collection parameters to generate the database initialization code for `store.py`.
