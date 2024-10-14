# DRAM Attention

DRAM Attention uses an LRU cache mechanism to efficiently manage the KV cache between main memory (DRAM) and GPU memory (HBM/VRAM). By storing the KV cache in main memory and selectively loading only the most relevant tokens into GPU memory when needed, it enables LLM inference with very long inputs using limited GPU memory while maintaining good performance.

## Getting Started

To install DRAM Attention:
```
git clone git+https://github.com/long-context/dram-attention
cd dram-attention
pip install -e .
```

## Evaluation

The evaluation results on `niah_multikey_2`, a simple key-value retrieval task from the [RULER benchmark](https://arxiv.org/abs/2404.06654), are as follows:

| Sequence Length | Average Recall |
|---------------:|---------------:|
|          8,192 |          1.00 |
|         16,384 |          1.00 |
|         32,768 |          1.00 |
|         65,536 |          0.99 |
|        131,072 |          0.94 |

To run the evaluation, go to `evaluation/dram-attention` and follow the instructions in `README.md`.

## API
The `DRAMAttention` class manages key-value cache data in both main memory and GPU memory. It uses two caches:

1. Local cache: Stored in HBM (GPU memory) for the most recent tokens (suffix tokens)
2. Prefix LRU cache: Managed by the `LRUCache` class and resides in DRAM (main memory)

At each decoding step, a small number of "important" pages are transferred from DRAM to HBM for attention computation.

Here's how to use it:

```python
import torch
from dram_attention import DRAMAttention

# Set up DRAM Attention
attention = DRAMAttention(
    max_output_length=64,         # Maximum length of generated sequence
    lru_hbm_cache_size=32*1024,   # Size of the HBM LRU cache
    local_cache_size=4096,        # Size of local cache for recent tokens
    page_size=16,                 # Size of each page in the LRU cache
    top_k=4096,                   # Number of top tokens to retrieve from LRU cache
    n_kv_heads=8,                 # Number of key-value heads
    head_dim=128,                 # Dimension of each head
    device="cuda",                # Device to store cache tensors
    dtype=torch.bfloat16          # Data type for cache tensors
)

# Prefill stage: Process the entire input prompt
output = attention(xq, xk, xv, start_pos=0, stage="prefill")

# Generate stage: Generate one token at a time
output = attention(xq, xk, xv, start_pos=pos, stage="generate")
```
DRAMAttention works in two stages:
- `prefill`: Initial pass that processes the input sequence and initializes both local and LRU caches
- `generate`: Generates new tokens one at a time, using both the local cache for recent tokens and selectively loading relevant pages from DRAM to HBM via the LRU cache

## Prior Work

DRAM Attention builds on two projects: [InfLLM](https://github.com/thunlp/InfLLM) and [Quest](https://github.com/mit-han-lab/Quest). A detailed write-up is coming soon!