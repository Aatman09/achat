# achat — GPT-2 from scratch, with KV-cache inference

A from-scratch PyTorch implementation of GPT-2, trained on FineWeb + Shakespeare,
extended with a **custom KV-cache inference engine** and benchmarked against the
no-cache baseline.

## What's in here

| File              | What it does                                                    |
|-------------------|-----------------------------------------------------------------|
| `model.py`        | GPT-2 architecture (CausalSelfAttention, MLP, Block, GPT)        |
| `train.py`        | Pretraining on FineWeb shards (bf16 autocast, grad-accum, clip) |
| `dataloader2.py`  | Sharded FineWeb dataloader                                      |
| `inference.py`    | Naive generation (recomputes K/V every step)                    |
| `model_kv.py`     | GPT-2 with KV cache plumbed through attention + Block + GPT     |
| `inference_kv.py` | Cached generation for the fine-tuned shakespeare model          |
| `benchmark.py`    | KV-cache vs no-cache: tokens/sec, peak memory, speedup          |

## The KV cache, in one paragraph

Naive generation re-runs the full sequence through every transformer block on
every new token: at step `t`, you recompute K and V for all `t` past tokens,
even though they haven't changed. KV caching saves K and V per layer once they're
computed, so each decode step does just one new token's worth of attention work.
Quadratic per-step cost becomes linear.

In `model_kv.py`, the cache is a list of `(k, v)` tensors, one per block:
- On the first call (`kv_caches=None`), the prefill runs as normal and the
  returned `new_kv_caches` are the K/V for every block.
- On each decode call, you pass a single new token plus the cache; each block
  concatenates the new K/V onto the past and returns the updated cache.
- `is_causal=True` only when `q.shape[2] == k.shape[2]` (prefill); during
  decode, the lone query attends to all past keys, no mask needed.
- Position IDs are continued from `past_length` so `wpe` doesn't reset to 0.

## Results

Run `python benchmark.py --new-tokens 500` to reproduce:

```
[no-cache      ] 500 new tokens in <X>s | <Y> tok/s | peak mem <M> MB
[kv-cache      ] 500 new tokens in <X>s | <Y> tok/s | peak mem <M> MB
Speedup       : <Nx>
```

(Fill in numbers after running on the target GPU.)

## Reproducing

```bash
pip install torch transformers tiktoken numpy matplotlib
python benchmark.py --new-tokens 500
# or:
python inference_kv.py   # requires shakespeare_gpt.pth from training
```
