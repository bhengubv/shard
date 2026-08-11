# Shard

Drop-in KV cache compression for LLM inference. **10× memory reduction** on Llama-3.1-8B-Instruct at 8K context, **11.2×** at 32K, with quality numbers indistinguishable from FP16. Full write-up: [research journal](https://krishgarg.com/shard) / [portfolio write-up](https://kirri-website.vercel.app/shard.html).

Shard is a `transformers.Cache` subclass you pass to `model.generate(..., past_key_values=cache)`. It compresses the K/V cache during prefill, runs attention directly on the compressed format (no FP16 K materialization), and uses a per-token 8-bit quantizer for decode that is bit-exact lossless across 750/750 measured tokens.

The compression is asymmetric. Keys go through PCA in their no-RoPE basis — Llama's K matrix is effectively rank-192 out of 1024 once you undo the rotation. Values get a Hadamard rotation followed by k-means vector quantization. Attention computes Q·K directly on int4 PCA coefficients via a per-pair relative-Δ RoPE identity, so K never gets reconstructed.

## Results

All on Llama-3.1-8B-Instruct, single B200.

| metric | result |
|---|---|
| Compression @ 8K context | **10.0×** |
| Compression @ 32K context | **11.2×** |
| NIAH recall (4K–32K, 20 needles) | **1.000** |
| LongBench-E avg Δ vs FP16 (8 tasks) | **−0.05** |
| WikiText-2 PPL | **6.47** vs 6.45 FP16 (+0.26%) |
| 8-bit streaming match vs FP16 | **750/750** |
| Decode throughput | 0.4–0.5× FP16 |

Throughput is below FP16. The implementation is a memory-capacity win, not a latency win. See the [blog](https://krishgarg.com/shard) for the profiler trace and what's left to fix.

## Install

```bash
pip install -e .
```

Requires `torch` and `transformers`. Triton is picked up automatically if installed (used by the int4 K matmul kernel and the all-heads VQ-V kernel); CPU/PyTorch fallbacks exist for every kernel.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from shard import Cache, enable_llama_fused_attention

model_id = "meta-llama/Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, dtype="float16", device_map="auto")
tok = AutoTokenizer.from_pretrained(model_id)

enable_llama_fused_attention(model)     # fused compressed-K attention
cache = Cache.from_model(model)
cache._streaming = True                 # 8-bit lossless decode streaming
cache._stream_bits = 8

ids = tok("Once upon a time", return_tensors="pt").to(model.device)
out = model.generate(**ids, max_new_tokens=128, past_key_values=cache)
```

## How it works

| segment | method | why |
|---|---|---|
| Prefill middle | PCA(K) + Hadamard+VQ(V) | exploits low-rank K and rotation-friendly V |
| Sink (4) + window (last 64) | FP16 | attention sink + recency, preserved exactly |
| Decode stream | TurboQuant-style Lloyd-Max | data-oblivious, no drift on long generations |
| Q·K | per-pair relative-Δ RoPE on int4 coefficients | no FP16 K reconstruction |
| weighted-sum V | inverse Hadamard pulled out past the sum | one transform at the end, not per token |

Full derivation, ablations, and the failed approaches are in the [blog](https://krishgarg.com/shard).

## Reproducing the benchmarks

End-to-end benchmark (NIAH, LongBench-E, WikiText PPL, compression-by-context, streaming match, decode throughput). Runs on Modal with a B200:

```bash
modal run benchmarks/benchmark.py
```

GPU kernel verification (every Triton kernel against its PyTorch reference, plus an attention end-to-end equivalence check):

```bash
modal run benchmarks/kernel_test.py
```

## Layout

```
src/shard/
  cache.py            # Cache — main API, prefill compression + decode buffer
  attention.py        # fused compressed-K attention + Llama monkey-patch
  triton_kernels.py   # int4 K matmul, all-heads VQ-V sum, fused-attention primitives
  streaming.py        # Lloyd-Max + optional QJL for per-token decode quantization
  rope.py             # RoPE utilities, DP bit allocation, bit-packing

benchmarks/
  benchmark.py        # full E2E suite
  kernel_test.py      # per-kernel correctness checks
```

## License

MIT.
