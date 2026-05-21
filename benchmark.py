"""Benchmark: KV-cache inference vs no-cache (recompute-everything) inference.

Loads GPT-2 (pretrained from HF) into two implementations:
  - model.GPT       : crops idx, recomputes all K/V every step
  - model_kv.GPT    : passes one new token + cache every step

Reports total time, tokens/sec, peak GPU memory.
"""
import argparse
import time
import torch
import torch.nn.functional as F
import tiktoken

from model import GPT as GPT_NoCache
from model_kv import GPT as GPT_KV


@torch.no_grad()
def generate_no_cache(model, idx, max_new_tokens, temperature=1.0):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -1024:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_token], dim=1)
    return idx


@torch.no_grad()
def generate_with_cache(model, idx, max_new_tokens, temperature=1.0):
    return model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature)


def run(model, generate_fn, prompt_tokens, max_new_tokens, device, label):
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        _ = generate_fn(model, prompt_tokens, 4)

    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        out = generate_fn(model, prompt_tokens, max_new_tokens)
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()

    elapsed = t1 - t0
    tok_per_s = max_new_tokens / elapsed
    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if device == "cuda" else 0.0

    print(f"[{label:14s}] {max_new_tokens} new tokens in {elapsed:.2f}s | {tok_per_s:7.2f} tok/s | peak mem {peak_mem_mb:8.1f} MB")
    return elapsed, tok_per_s, peak_mem_mb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-tokens", type=int, default=500)
    parser.add_argument("--prompt", type=str, default="The capital of France is")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Generating {args.new_tokens} new tokens. Prompt: {args.prompt!r}\n")

    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(args.prompt)
    prompt_tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    # ---- no-cache path
    model_nc = GPT_NoCache.from_pretrained("gpt2").to(device).eval()
    e_nc, tps_nc, mem_nc = run(model_nc, generate_no_cache, prompt_tokens, args.new_tokens, device, "no-cache")
    del model_nc
    if device == "cuda":
        torch.cuda.empty_cache()

    # ---- KV-cache path
    model_kv = GPT_KV.from_pretrained("gpt2").to(device).eval()
    e_kv, tps_kv, mem_kv = run(model_kv, generate_with_cache, prompt_tokens, args.new_tokens, device, "kv-cache")
    del model_kv
    if device == "cuda":
        torch.cuda.empty_cache()

    print()
    print(f"Speedup       : {e_nc / e_kv:.2f}x ({tps_kv / tps_nc:.2f}x tok/s)")
    print(f"Time saved    : {(e_nc - e_kv):.2f}s")
    print(f"Memory delta  : kv-cache uses {mem_kv - mem_nc:+.1f} MB vs no-cache")


if __name__ == "__main__":
    main()
