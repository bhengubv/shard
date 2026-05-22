"""Minimal Shard usage. Loads Llama-3.1-8B-Instruct, swaps in a Shard Cache,
generates 64 tokens, and prints the compression ratio + memory savings.

Requires a CUDA GPU with ~16 GB free (model weights) and `pip install -e .` first.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from shard import Cache, enable_llama_fused_attention


MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
PROMPT = "Write a short poem about the ocean."


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, device_map="auto"
    )
    model.eval()

    enable_llama_fused_attention(model)

    cache = Cache.from_model(model)
    cache._streaming = True
    cache._stream_bits = 8

    inputs = tok(PROMPT, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            past_key_values=cache,
            pad_token_id=tok.eos_token_id,
        )

    text = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(text)
    print()
    print(f"compression ratio: {cache.compress_ratio():.2f}×")
    print(f"compressed cache size: {cache.memory_bytes() / 1024 / 1024:.1f} MB")
    print(f"fp16 equivalent: {cache.fp16_equivalent_bytes() / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
