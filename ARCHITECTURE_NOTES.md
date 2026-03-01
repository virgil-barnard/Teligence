# Teligence GPT Architecture Notes

This document explains the major architectural decisions in this repository and what they imply for capability, performance, and limits.


## 1) Model at a Glance

The model is a decoder-only GPT implemented in TensorFlow with:

- multi-layer Transformer blocks
- grouped-query attention (GQA)
- optional fused QKV projection
- optional FlashAttention-style training kernel
- local sliding-window attention semantics
- ring-cache incremental inference
- tokenizer abstraction (byte/char)

Primary implementation files:

- `modeling.py`
- `train_utils.py`
- `data_utils.py`
- `tokenizer.py`


## 2) Key Architectural Decisions

## 2.1 Local Sliding-Window Attention

### Decision

Training attention is constrained to a window `W` (`attn_window`) instead of full `T x T` attention over sequence length `T`.

### Why

- makes attention cost effectively linear in sequence length (for fixed window)
- allows larger `T` or larger batch at same memory budget
- aligns training behavior with ring-cache inference behavior

### Implications

- model cannot directly attend beyond the last `W` tokens in a single forward pass
- long-range dependencies beyond `W` must be compressed into hidden state indirectly, not directly attended
- often better throughput and memory efficiency than full attention


## 2.2 Ring-Cache Inference

### Decision

During autoregressive generation, each layer stores key/value tensors in a circular buffer of size `W`.

### Why

- keeps inference memory bounded by `O(layers * heads * W * d_head)`
- constant-time cache growth (no unbounded KV accumulation)

### Implications

- latency and memory remain stable for long generation
- direct attention horizon is capped at `W`
- behavior matches local training objective well


## 2.3 GQA (Grouped Query Attention)

### Decision

Use fewer KV heads than query heads (`n_kv_head < n_head`), with query heads grouped per KV head.

### Why

- reduces KV memory and bandwidth cost
- reduces compute in parts of attention path

### Implications

- usually better efficiency with small/acceptable quality tradeoff
- especially helpful for inference and cache footprint


## 2.4 Fused QKV Projection

### Decision

Compute Q, K, V from one linear projection when enabled.

### Why

- fewer kernel launches / better memory locality
- often modest throughput improvement

### Implications

- mostly an efficiency optimization
- minimal modeling behavior change


## 2.5 Positional Encoding Choice (RoPE default, ALiBi optional)

### Decision

Architecture supports both RoPE and ALiBi; default is RoPE.

### Why

- RoPE: strong default for decoder models
- ALiBi: can be robust for extrapolation and simple distance biasing

### Implications

- extrapolation behavior depends on encoding and hyperparameters
- attention bias semantics differ; should be benchmarked per domain


## 2.6 FlashAttention-Style Training Kernel

### Decision

Training uses an online-softmax attention implementation with `tf.while_loop` blocks instead of explicitly materializing full attention matrices.

### Why

- lower peak memory usage
- better scalability for longer contexts/windows

### Implications

- more complex implementation/debugging than naive attention
- potential backend-specific performance variability


## 2.7 RMSNorm + SwiGLU MLP

### Decision

Use RMSNorm and optionally SwiGLU feed-forward blocks.

### Why

- RMSNorm is common in modern efficient decoder stacks
- SwiGLU often improves quality-per-parameter vs plain ReLU MLP

### Implications

- generally stable optimization
- slightly more complex than plain LayerNorm/ReLU setup


## 2.8 Explicit AdamW-Style Weight Decay

### Decision

Apply weight decay explicitly after gradient update to rank>=2 parameters.

### Why

- keeps regularization behavior controlled and explicit

### Implications

- easier to reason about exactly which weights decay
- should be tuned alongside LR/warmup schedule


## 2.9 Tokenizer Abstraction

### Decision

Tokenizer is modular (`tokenizer.py`) with byte and char strategies.

### Why

- allows domain-specific strategy testing without touching core training loop

### Implications

- easy benchmark comparisons across token granularities
- vocabulary size and sequence statistics vary significantly by tokenizer


## 3) Can This Work Like Infinite Context / Persistent Memory?

Short answer: **not in true infinite-context sense**.

### What it can do

- generate indefinitely with bounded memory usage due to ring cache
- keep a moving local attention window over recent tokens

### What it cannot do (currently)

- directly attend to arbitrarily old tokens once they fall outside `attn_window`
- maintain exact long-term token-level memory over unbounded history

### Practical interpretation

It is **streaming-friendly** and **bounded-memory**, but not infinite-context retrieval.


## 4) If You Want Longer/"Memory-like" Behavior

Potential next upgrades:

- increase `attn_window` (simple, expensive)
- hierarchical memory summary tokens (compress old context)
- retrieval augmentation (external vector store / key-value memory)
- recurrent memory modules across segments
- chunked recurrence (Transformer-XL style ideas)

These can approximate long-term memory while preserving bounded compute.


## 5) Performance/Quality Tradeoff Matrix

## 5.1 Good for

- throughput-conscious experimentation
- streaming generation with stable memory
- architecture ablations where bounded context is acceptable

## 5.2 Risk areas

- tasks requiring exact long-range dependency > `attn_window`
- domains where rare long-distance references dominate quality
- overfitting to short-context heuristics if window too small


## 6) Evaluation Guidance

To evaluate architectural advantage fairly:

- hold parameter count as constant as possible
- hold token budget constant
- compare across at least 2-3 text domains
- track both quality and systems metrics

Recommended metrics already supported:

- `val_nll`, `val_bpc`, `val_ppl`
- `test_nll`, `test_bpc`, `test_ppl`
- `tok_per_s`

Recommended TensorBoard views:

- `train/loss` vs `train/tok_per_s`
- `eval/val_bpc` and `eval/best_val_bpc`
- attention probe images over time
- weight histogram drift for stability signals


## 7) Current Repo Positioning

At this stage, the architecture is best described as:

- efficient, bounded-context GPT for rapid iteration
- strongly instrumented for debugging (TensorBoard, attention probes)
- modular for tokenizer and dataset/domain benchmarking

It is a good platform for testing where local-attention + ring-cache designs win on efficiency, and where they lose on long-range recall.
