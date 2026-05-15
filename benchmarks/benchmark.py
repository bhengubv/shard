"""Full E2E comparison vs TurboQuant with all Shard features enabled:
fused compressed-K attention + TurboQuant-style streaming + lazy-per-layer.

TurboQuant paper headline on Llama-3.1-8B-Instruct:
  - Compression: 4.5x (3.5-bit) / 6x (3-bit)
  - NIAH: 0.997 recall (matches FP16 0.997)
  - LongBench-E: near-lossless Δ from FP16
"""
import modal
import os
import json
import subprocess
import time
from pathlib import Path

app = modal.App("shard-tq-e2e-final")
vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
result_vol = modal.Volume.from_name("shard-benchmark-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "accelerate", "datasets",
                 "sentencepiece", "protobuf", "numpy",
                 "rouge-score", "pandas", "pyarrow", "huggingface-hub")
    .add_local_dir("src/shard", remote_path="/root/shard")
)


@app.function(image=image, gpu="B200", timeout=14400,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              memory=131072, volumes={"/hf-cache": vol, "/results": result_vol})
def run(sections="all", longbench_samples_per_task=50, longbench_tasks="all",
        full_fused=False, profile_ctx_len=8192, profile_tokens=16,
        profile_warmup_tokens=4, profile_decode_only=False):
    import sys, re, string, math, json, zipfile, random, platform
    sys.path.insert(0, "/root")
    os.environ["HF_HOME"] = "/hf-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/hf-cache"

    import numpy as np
    import torch, pandas as pd
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from shard.cache import Cache
    from shard.rope import get_rope_inv_freq
    from shard.attention import enable_llama_fused_attention
    from huggingface_hub import hf_hub_download
    from datasets import load_dataset
    from rouge_score import rouge_scorer

    device = "cuda"
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    hf_token = os.environ.get("HF_TOKEN")
    seed = 42
    requested_sections = {s.strip() for s in sections.split(",") if s.strip()}
    if "all" in requested_sections:
        requested_sections = {
            "niah", "longbench", "ppl", "compression", "streaming", "throughput"
        }
    selected_longbench_tasks = {
        t.strip() for t in longbench_tasks.split(",") if t.strip()
    }
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    print(f"\n{'='*72}")
    print(f"  SHARD vs TURBOQUANT — FULL E2E WITH ALL FEATURES ENABLED")
    print(f"  Model: {model_id}")
    print(f"  Features: fused attention + streaming(8-bit) + lazy-per-layer")
    print(f"  Full-fused compressed decode: {'ON' if full_fused else 'OFF'}")
    if full_fused:
        print("  WARNING: full-fused decode is experimental and currently slower "
              "than the default path.", flush=True)
    print(f"{'='*72}\n")

    tok = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16, device_map="auto", token=hf_token)
    model.eval()
    vol.commit()

    # Enable the fused attention monkey-patch
    ok = enable_llama_fused_attention(model)
    print(f"Fused attention patch: {'INSTALLED' if ok else 'FAILED'}", flush=True)

    inv_freq = get_rope_inv_freq(model)
    hd = getattr(model.config, "head_dim",
                 model.config.hidden_size // model.config.num_attention_heads)
    nl = model.config.num_hidden_layers
    nkv = getattr(model.config, "num_key_value_heads",
                  model.config.num_attention_heads)
    cache_config = {
        "k_target_cr": 20,
        "max_rank_k": 192,
        "v_vq_group_size": 4,
        "v_vq_codebook_size": 256,
        "residual_window": 64,
        "streaming": True,
        "stream_bits": 8,
        "stream_qjl": False,
        "full_fused": full_fused,
        "profile_ctx_len": profile_ctx_len,
        "profile_tokens": profile_tokens,
        "profile_warmup_tokens": profile_warmup_tokens,
        "profile_decode_only": profile_decode_only,
    }
    results = {
        "metadata": {
            "benchmark": "shard_e2e",
            "model_id": model_id,
            "seed": seed,
            "device": device,
            "gpu_name": torch.cuda.get_device_name(0),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "cuda": torch.version.cuda,
            "cache_config": cache_config,
            "model_config": {
                "num_hidden_layers": nl,
                "num_attention_heads": model.config.num_attention_heads,
                "num_key_value_heads": nkv,
                "hidden_size": model.config.hidden_size,
                "head_dim": hd,
            },
            "notes": {
                "turboquant_numbers": "Paper reference only; TurboQuant is not run in this script.",
                "ppl": "Continuation PPL after compressed prefill, not full sliding-window PPL.",
            },
            "requested_sections": sorted(requested_sections),
            "longbench_samples_per_task": longbench_samples_per_task,
            "longbench_tasks": (
                "all" if "all" in selected_longbench_tasks
                else sorted(selected_longbench_tasks)
            ),
        },
        "niah": {"records": [], "summary": {}},
        "longbench_e": {"records": [], "summary": {}},
        "ppl": {"records": [], "summary": {}},
        "compression": {"records": [], "summary": {}},
        "streaming": {"records": [], "summary": {}},
        "throughput": {"records": [], "summary": {}},
        "profile": {"records": [], "summary": {}},
    }

    def want(section):
        return section in requested_sections

    def checkpoint(section):
        results["metadata"]["last_completed_section"] = section
        path = f"/results/shard_e2e_benchmark_{section}.json"
        latest_path = "/results/shard_e2e_benchmark_latest_partial.json"
        payload = json.dumps(results, indent=2) + "\n"
        with open(path, "w") as f:
            f.write(payload)
        with open(latest_path, "w") as f:
            f.write(payload)
        result_vol.commit()
        print(f"  checkpoint: {path}", flush=True)

    def make_cache(streaming=True, stream_bits=8):
        c = Cache(head_dim=hd, num_layers=nl, n_kv_heads=nkv,
                        device=device, k_target_cr=cache_config["k_target_cr"],
                        max_rank_k=cache_config["max_rank_k"],
                        v_vq_group_size=cache_config["v_vq_group_size"],
                        v_vq_codebook_size=cache_config["v_vq_codebook_size"],
                        residual_window=cache_config["residual_window"],
                        rope_inv_freq=inv_freq)
        c._streaming = streaming
        c._stream_bits = stream_bits
        c._stream_qjl = cache_config["stream_qjl"]
        c._full_fused = cache_config["full_fused"]
        return c

    def fp16_cache_bytes(seq_len, batch_size=1):
        return batch_size * nl * 2 * nkv * seq_len * hd * 2

    def cuda_peak_mb():
        return torch.cuda.max_memory_allocated() / 1024 / 1024

    def full_attention_mask(input_ids):
        return torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)

    def greedy_decode_from_prefill(prefill_out, cache=None, n_tokens=128):
        next_ids = prefill_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        past_key_values = None if cache is not None else prefill_out.past_key_values
        generated = []
        for _ in range(n_tokens):
            generated.append(int(next_ids[0, 0]))
            kwargs = {"use_cache": True}
            if cache is not None:
                kwargs["past_key_values"] = cache
            else:
                kwargs["past_key_values"] = past_key_values
            with torch.no_grad():
                out = model(next_ids, **kwargs)
            next_ids = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            if cache is None:
                past_key_values = out.past_key_values
        return generated

    def force_compress_cache(cache):
        """Trigger one-time Shard prefill compression outside timed decode regions."""
        if hasattr(cache, "compress"):
            cache.compress()
        torch.cuda.synchronize()

    _inp = tok("hello", return_tensors="pt").to(device)
    with torch.no_grad():
        model.generate(input_ids=_inp.input_ids,
                       attention_mask=full_attention_mask(_inp.input_ids),
                       max_new_tokens=3, do_sample=False,
                       pad_token_id=tok.eos_token_id)

    fw = load_dataset("HuggingFaceFW/fineweb", "sample-10BT",
                      split="train", streaming=True)
    texts = []
    for item in fw:
        texts.append(item["text"])
        if len(texts) >= 80:
            break
    haystack = " ".join(texts)
    hay_ids = tok(haystack, return_tensors="pt").input_ids[0]

    needle = ("\nThe best thing to do in San Francisco is eat a sandwich "
              "and sit in Dolores Park on a sunny day.\n")
    question = "\n\nWhat is the best thing to do in San Francisco?\nAnswer:"
    needle_ids = tok(needle, return_tensors="pt").input_ids[0]
    q_ids = tok(question, return_tensors="pt").input_ids[0]

    niah = {}
    if want("niah"):
        # ==============================================================
        # 1. NIAH (Fu et al. setup) — target: beat 0.997 at higher cr
        # ==============================================================
        print(f"\n--- 1. NEEDLE-IN-A-HAYSTACK ---")
        print(f"  Target: TurboQuant=0.997 at ~4-6x compression")

        for ctx_len in [4096, 8192, 16384, 32768]:
            if hay_ids.shape[0] < ctx_len:
                continue
            hits, total, crs = 0, 0, []
            for depth in [0.1, 0.3, 0.5, 0.7, 0.9]:
                budget = ctx_len - needle_ids.shape[0] - q_ids.shape[0] - 10
                pos = int(budget * depth)
                full = torch.cat([hay_ids[:pos], needle_ids,
                                  hay_ids[pos:budget], q_ids]).unsqueeze(0).to(device)
                cache = make_cache()
                with torch.no_grad():
                    out = model.generate(full, attention_mask=full_attention_mask(full),
                                          max_new_tokens=30, do_sample=False,
                                          past_key_values=cache,
                                          pad_token_id=tok.eos_token_id)
                ans = tok.decode(out[0][full.shape[1]:],
                                 skip_special_tokens=True).lower()
                hit = "sandwich" in ans and ("dolores" in ans or "park" in ans)
                hits += int(hit)
                total += 1
                cr = cache.clean_compress_ratio()
                runtime_cr = cache.compress_ratio()
                crs.append(cr)
                results["niah"]["records"].append({
                    "ctx_len": ctx_len,
                    "depth": depth,
                    "seed": seed,
                    "method": "shard",
                    "correct": bool(hit),
                    "response": ans,
                    "compression_ratio": cr,
                    "compression_ratio_kind": "clean_storage",
                    "runtime_compression_ratio": runtime_cr,
                    "input_tokens": int(full.shape[1]),
                })
                del cache
                torch.cuda.empty_cache()
            rec = hits / total
            niah[ctx_len] = (rec, sum(crs) / len(crs))
            results["niah"]["summary"][str(ctx_len)] = {
                "recall": rec,
                "hits": hits,
                "total": total,
                "compression_ratio_mean": sum(crs) / len(crs),
            }
            print(f"  {ctx_len:>6}: recall={rec:.3f}  cr={sum(crs)/len(crs):.1f}x",
                  flush=True)
        checkpoint("niah")

    # ==============================================================
    # 2. LongBench-E (full subset) — target: lossless Δ from FP16
    # ==============================================================
    if want("longbench"):
        print(f"\n--- 2. LONGBENCH-E SUBSET ---")
        print(f"  Target: near-zero Δ from FP16 baseline (TurboQuant claim)")

    def normalize(s):
        s = s.lower()
        s = re.sub(r"\b(a|an|the)\b", " ", s)
        s = "".join(ch for ch in s if ch not in string.punctuation)
        return re.sub(r"\s+", " ", s).strip()

    def f1_score(pred, truths):
        pred = normalize(pred)
        best = 0.0
        for t in truths if isinstance(truths, (list, tuple)) else [truths]:
            t = normalize(str(t))
            if not pred or not t:
                continue
            pt, tt = pred.split(), t.split()
            common = set(pt) & set(tt)
            if not common:
                continue
            ncom = sum(min(pt.count(w), tt.count(w)) for w in common)
            if ncom == 0:
                continue
            p, r = ncom / len(pt), ncom / len(tt)
            best = max(best, 2 * p * r / (p + r))
        return best

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    def rouge_l(pred, truths):
        best = 0.0
        for t in truths if isinstance(truths, (list, tuple)) else [truths]:
            best = max(best, scorer.score(str(t), pred)['rougeL'].fmeasure)
        return best

    if want("longbench"):
        zip_path = hf_hub_download(repo_id="zai-org/LongBench", repo_type="dataset",
                                    filename="data.zip", token=hf_token)
        zf = zipfile.ZipFile(zip_path, "r")
        names = set(zf.namelist())
    else:
        zf = None
        names = set()

    def load_task(task):
        for suf in [f"{task}_e", task]:
            p = f"data/{suf}.jsonl"
            if p in names:
                with zf.open(p) as f:
                    rows = [json.loads(line) for line in f if line.strip()]
                return pd.DataFrame(rows)
        return None

    tasks = [
        ("qasper", "f1", 128),
        ("multifieldqa_en", "f1", 64),
        ("hotpotqa", "f1", 32),
        ("2wikimqa", "f1", 32),
        ("gov_report", "rouge", 512),
        ("multi_news", "rouge", 512),
        ("triviaqa", "f1", 32),
        ("samsum", "rouge", 128),
    ]
    if "all" not in selected_longbench_tasks:
        tasks = [task for task in tasks if task[0] in selected_longbench_tasks]
    MAX_CTX = 7500

    lb_results = {}
    for task_name, metric, max_new in (tasks if want("longbench") else []):
        df = load_task(task_name)
        if df is None:
            continue
        n = min(longbench_samples_per_task, len(df))
        df = df.sample(n=n, random_state=seed).reset_index(drop=True)
        fp_scores, cp_scores, crs = [], [], []
        for i in range(n):
            ex = df.iloc[i]
            prompt = str(ex.get("input", ex.get("prompt", "")))
            context = str(ex.get("context", ""))
            ans = ex["answers"] if "answers" in ex else ex.get("answer", "")
            if hasattr(ans, 'tolist'):
                ans = ans.tolist()
            if not isinstance(ans, (list, tuple)):
                ans = [ans]
            full_text = (f"{context}\n\n{prompt}\n\nAnswer:"
                         if context else f"{prompt}\n\nAnswer:")
            ids = tok(full_text, return_tensors="pt", truncation=True,
                       max_length=MAX_CTX).input_ids.to(device)
            if ids.shape[1] < 512:
                continue

            with torch.no_grad():
                out = model.generate(ids, attention_mask=full_attention_mask(ids),
                                      max_new_tokens=max_new,
                                      do_sample=False,
                                      pad_token_id=tok.eos_token_id)
            fp_pred = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

            cache = make_cache()
            with torch.no_grad():
                outc = model.generate(ids, attention_mask=full_attention_mask(ids),
                                       max_new_tokens=max_new,
                                       do_sample=False,
                                       pad_token_id=tok.eos_token_id,
                                       past_key_values=cache)
            cp_pred = tok.decode(outc[0][ids.shape[1]:], skip_special_tokens=True)
            crs.append(cache.clean_compress_ratio())
            runtime_cr = cache.compress_ratio()
            del cache
            torch.cuda.empty_cache()

            scorer_fn = f1_score if metric == "f1" else rouge_l
            fp_score = scorer_fn(fp_pred, ans)
            cp_score = scorer_fn(cp_pred, ans)
            fp_scores.append(fp_score)
            cp_scores.append(cp_score)
            results["longbench_e"]["records"].append({
                "task": task_name,
                "metric": metric,
                "sample_index": i,
                "input_tokens": int(ids.shape[1]),
                "max_new_tokens": max_new,
                "answers": [str(a) for a in ans],
                "fp16_prediction": fp_pred,
                "shard_prediction": cp_pred,
                "fp16_score": fp_score,
                "shard_score": cp_score,
                "delta": cp_score - fp_score,
                "compression_ratio": crs[-1] if crs else None,
                "compression_ratio_kind": "clean_storage",
                "runtime_compression_ratio": runtime_cr,
            })

        if not fp_scores:
            continue
        fp_avg = 100 * sum(fp_scores) / len(fp_scores)
        cp_avg = 100 * sum(cp_scores) / len(cp_scores)
        cr_avg = sum(crs) / len(crs) if crs else 0
        lb_results[task_name] = (fp_avg, cp_avg, cr_avg)
        results["longbench_e"]["summary"][task_name] = {
            "metric": metric,
            "n": len(fp_scores),
            "fp16": fp_avg,
            "shard": cp_avg,
            "delta": cp_avg - fp_avg,
            "compression_ratio_mean": cr_avg,
        }
        print(f"  {task_name:<20}: FP16={fp_avg:5.2f}  "
              f"Shard={cp_avg:5.2f}  Δ={cp_avg-fp_avg:+5.2f}  "
              f"cr={cr_avg:.1f}x", flush=True)
        checkpoint(f"longbench_{task_name}")
    if want("longbench"):
        checkpoint("longbench")

    # ==============================================================
    # 3. WikiText-2 PPL
    # ==============================================================
    fp_ppl, cp_ppl = [], []
    if want("ppl"):
        print(f"\n--- 3. WIKITEXT-2 PPL ---")
        wt2 = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        wt2_text = "\n".join([t for t in wt2["text"] if t.strip()])
        wt2_ids = tok(wt2_text, return_tensors="pt").input_ids[0].to(device)

        import torch.nn.functional as F
        n_windows = min(10, (wt2_ids.shape[0] - 256) // 2048)
        for wi in range(n_windows):
            start = wi * 2048
            ids = wt2_ids[start:start + 2304].unsqueeze(0)
            if ids.shape[1] < 2304:
                continue
            with torch.no_grad():
                logits_fp = model(ids, attention_mask=full_attention_mask(ids)).logits
            shift_fp = logits_fp[:, 2047:-1, :].contiguous()
            labels = ids[:, 2048:]
            loss_fp = F.cross_entropy(
                shift_fp.reshape(-1, shift_fp.size(-1)),
                labels.reshape(-1)).item()

            cache = make_cache()
            ctx = ids[:, :2048]
            evl = ids[:, 2048:]
            with torch.no_grad():
                model(ctx, attention_mask=full_attention_mask(ctx),
                      use_cache=True, past_key_values=cache)
            cache.compress()
            pos_ids = torch.arange(2048, 2304, device=device).unsqueeze(0)
            with torch.no_grad():
                out_c = model(evl, past_key_values=cache, use_cache=True,
                               position_ids=pos_ids)
            shift_c = out_c.logits[:, :-1, :].contiguous()
            evl_labels = evl[:, 1:]
            ml = min(shift_c.shape[1], evl_labels.shape[1])
            loss_c = F.cross_entropy(
                shift_c[:, :ml].reshape(-1, shift_c.size(-1)),
                evl_labels[:, :ml].reshape(-1)).item()
            fp_ppl.append(math.exp(loss_fp))
            cp_ppl.append(math.exp(loss_c))
            results["ppl"]["records"].append({
                "window": wi,
                "context_tokens": 2048,
                "eval_tokens": int(ml),
                "fp16_loss": loss_fp,
                "shard_loss": loss_c,
                "fp16_ppl": math.exp(loss_fp),
                "shard_ppl": math.exp(loss_c),
                "compression_ratio": cache.clean_compress_ratio(),
                "compression_ratio_kind": "clean_storage",
                "runtime_compression_ratio": cache.compress_ratio(),
            })
            del cache
            torch.cuda.empty_cache()

        if fp_ppl:
            fp_avg = sum(fp_ppl) / len(fp_ppl)
            cp_avg = sum(cp_ppl) / len(cp_ppl)
            results["ppl"]["summary"] = {
                "dataset": "wikitext-2-raw-v1",
                "windows": len(fp_ppl),
                "context_tokens": 2048,
                "eval_tokens_per_window": 256,
                "fp16_ppl": fp_avg,
                "shard_ppl": cp_avg,
                "relative_delta_pct": 100 * (cp_avg - fp_avg) / fp_avg,
            }
            print(f"  FP16 PPL: {fp_avg:.2f}", flush=True)
            print(f"  Shard PPL: {cp_avg:.2f}  "
                  f"(Δ={100*(cp_avg-fp_avg)/fp_avg:+.2f}%)", flush=True)
        checkpoint("ppl")

    # ==============================================================
    # 4. Compression by context length
    # ==============================================================
    if want("compression"):
        print(f"\n--- 4. COMPRESSION BY CONTEXT ---")
        for sl in [4096, 8192, 16384, 32768]:
            if hay_ids.shape[0] < sl:
                continue
            ids = hay_ids[:sl].unsqueeze(0).to(device)
            cache = make_cache(streaming=False)  # measure prefill compression
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                model.generate(ids, attention_mask=full_attention_mask(ids),
                               max_new_tokens=2, do_sample=False,
                               past_key_values=cache,
                               pad_token_id=tok.eos_token_id)
            peak_mb = cuda_peak_mb()
            cr = cache.clean_compress_ratio()
            mem = cache.clean_memory_bytes() / 1024 / 1024
            runtime_cr = cache.compress_ratio()
            runtime_mem = cache.memory_bytes() / 1024 / 1024
            fp16_mb = fp16_cache_bytes(sl) / 1024 / 1024
            record = {
                "seq_len": sl,
                "fp16_cache_mb": fp16_mb,
                "shard_stored_mb": mem,
                "shard_runtime_mb": runtime_mem,
                "peak_allocated_mb": peak_mb,
                "compression_ratio": cr,
                "runtime_compression_ratio": runtime_cr,
                "effective_bits_per_element": 16 / cr if cr else None,
            }
            results["compression"]["records"].append(record)
            results["compression"]["summary"][str(sl)] = record
            print(f"  seq={sl:>6}: {cr:.1f}x  stored={mem:.0f}MB", flush=True)
            del cache
            torch.cuda.empty_cache()
        checkpoint("compression")

    # ==============================================================
    # 5. Lossless streaming quality (TurboQuant-style)
    # ==============================================================
    tm, tt = 0, 0
    if want("streaming"):
        print(f"\n--- 5. STREAMING QUALITY (8-bit lossless) ---")
        print(f"  Target: 100% match vs FP16 at 150 decode tokens")
        prefix = hay_ids[:8000]
        prompts = [
            "\n\nQ: Capital of France?\nA:",
            "\n\ndef reverse_string(s):",
            "\n\nThe scientific method involves",
            "\n\nOnce upon a time in a kingdom",
            "\n\nSolve: 2x+5=13. x=",
        ]
        for p in prompts:
            p_ids = tok(p, return_tensors="pt").input_ids[0]
            full = torch.cat([prefix, p_ids]).unsqueeze(0).to(device)
            with torch.no_grad():
                out_fp = model.generate(full, attention_mask=full_attention_mask(full),
                                         max_new_tokens=150, do_sample=False,
                                         pad_token_id=tok.eos_token_id)
            fp = out_fp[0][full.shape[1]:].tolist()
            cache = make_cache(streaming=True, stream_bits=8)
            with torch.no_grad():
                out_c = model.generate(full, attention_mask=full_attention_mask(full),
                                        max_new_tokens=150, do_sample=False,
                                        past_key_values=cache,
                                        pad_token_id=tok.eos_token_id)
            cp = out_c[0][full.shape[1]:].tolist()
            m = sum(1 for a, b in zip(fp, cp) if a == b)
            tm += m; tt += min(len(fp), len(cp))
            results["streaming"]["records"].append({
                "prompt": p,
                "stream_bits": 8,
                "tokens_compared": min(len(fp), len(cp)),
                "matching_tokens": m,
                "match_rate": m / min(len(fp), len(cp)) if min(len(fp), len(cp)) else 0,
                "fp16_tokens": fp,
                "shard_tokens": cp,
            })
            del cache
            torch.cuda.empty_cache()
        results["streaming"]["summary"] = {
            "stream_bits": 8,
            "matching_tokens": tm,
            "tokens_compared": tt,
            "match_rate": tm / tt if tt else 0,
        }
        print(f"  Streaming 8-bit: {tm}/{tt} ({100*tm/tt:.1f}%)", flush=True)
        checkpoint("streaming")

    # ==============================================================
    # 6. Decode throughput (tok/s) vs FP16 baseline
    # ==============================================================
    tput = {}
    N_DECODE = 128
    N_WARMUP = 8
    if want("throughput"):
        print(f"\n--- 6. DECODE THROUGHPUT (tok/s) ---")
        print(f"  reports prefill, one-time compression, and steady-state decode separately")
        for ctx_len in [2048, 4096, 8192]:
            if hay_ids.shape[0] < ctx_len:
                continue
            ids = hay_ids[:ctx_len].unsqueeze(0).to(device)

            # FP16 baseline
            with torch.no_grad():
                warm = model(ids, attention_mask=full_attention_mask(ids),
                             use_cache=True)
            greedy_decode_from_prefill(warm, n_tokens=N_WARMUP)
            del warm
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad():
                fp_prefill = model(ids, attention_mask=full_attention_mask(ids),
                                   use_cache=True)
            torch.cuda.synchronize()
            fp_prefill_s = time.time() - t0
            fp_peak_mb = cuda_peak_mb()

            torch.cuda.synchronize()
            t0 = time.time()
            fp_tokens = greedy_decode_from_prefill(fp_prefill, n_tokens=N_DECODE)
            torch.cuda.synchronize()
            fp_decode_s = time.time() - t0
            fp_tps = N_DECODE / fp_decode_s
            del fp_prefill
            torch.cuda.empty_cache()

            # Shard (fused attn + streaming)
            c_warm = make_cache()
            with torch.no_grad():
                sh_warm = model(ids, attention_mask=full_attention_mask(ids),
                                use_cache=True, past_key_values=c_warm)
            greedy_decode_from_prefill(sh_warm, cache=c_warm, n_tokens=N_WARMUP)
            del c_warm
            del sh_warm
            torch.cuda.empty_cache()
            cache = make_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad():
                sh_prefill = model(ids, attention_mask=full_attention_mask(ids),
                                   use_cache=True, past_key_values=cache)
            torch.cuda.synchronize()
            sh_prefill_s = time.time() - t0
            sh_peak_prefill_mb = cuda_peak_mb()

            torch.cuda.synchronize()
            t0 = time.time()
            force_compress_cache(cache)
            torch.cuda.synchronize()
            sh_compress_s = time.time() - t0
            sh_peak_compress_mb = cuda_peak_mb()

            torch.cuda.synchronize()
            t0 = time.time()
            sh_tokens = greedy_decode_from_prefill(sh_prefill, cache=cache,
                                                   n_tokens=N_DECODE)
            torch.cuda.synchronize()
            sh_decode_s = time.time() - t0
            sh_tps = N_DECODE / sh_decode_s
            sh_peak_total_mb = cuda_peak_mb()
            cr = cache.clean_compress_ratio()
            runtime_cr = cache.compress_ratio()
            token_matches = sum(1 for a, b in zip(fp_tokens, sh_tokens) if a == b)
            del sh_prefill
            del cache
            torch.cuda.empty_cache()

            tput[ctx_len] = (fp_tps, sh_tps, cr)
            ratio = sh_tps / fp_tps
            results["throughput"]["records"].append({
                "ctx_len": ctx_len,
                "decode_tokens": N_DECODE,
                "warmup_tokens": N_WARMUP,
                "fp16_prefill_s": fp_prefill_s,
                "shard_prefill_s": sh_prefill_s,
                "shard_compress_s": sh_compress_s,
                "fp16_decode_s": fp_decode_s,
                "shard_decode_s": sh_decode_s,
                "fp16_tok_per_s": fp_tps,
                "shard_tok_per_s": sh_tps,
                "speedup_vs_fp16": ratio,
                "compression_ratio": cr,
                "compression_ratio_kind": "clean_storage",
                "runtime_compression_ratio": runtime_cr,
                "fp16_peak_prefill_mb": fp_peak_mb,
                "shard_peak_prefill_mb": sh_peak_prefill_mb,
                "shard_peak_compress_mb": sh_peak_compress_mb,
                "shard_peak_total_mb": sh_peak_total_mb,
                "decode_token_matches": token_matches,
                "decode_token_match_rate": token_matches / N_DECODE,
            })
            results["throughput"]["summary"][str(ctx_len)] = (
                results["throughput"]["records"][-1]
            )
            print(f"  ctx={ctx_len:>5}: FP16={fp_tps:6.2f} tok/s  "
                  f"Shard={sh_tps:6.2f} tok/s  ({ratio:.2f}× FP16, cr={cr:.1f}×)",
                  flush=True)
        checkpoint("throughput")

    # ==============================================================
    # 7. Decode profiler trace
    # ==============================================================
    if want("profile"):
        print(f"\n--- 7. DECODE PROFILER ---")
        print("  profiling FP16 decode and Shard steady decode "
              f"at ctx={profile_ctx_len}, tokens={profile_tokens}")
        if profile_decode_only:
            print("  decode-only mode: skipping mixed prefill/compression profile")
        ctx_len = int(profile_ctx_len)
        n_profile_tokens = int(profile_tokens)
        n_warmup_tokens = int(profile_warmup_tokens)
        if hay_ids.shape[0] < ctx_len:
            ctx_len = min(4096, hay_ids.shape[0])
        ids = hay_ids[:ctx_len].unsqueeze(0).to(device)

        # Warm both paths before collecting a trace so one-time setup does not dominate.
        with torch.no_grad():
            warm = model(ids, attention_mask=full_attention_mask(ids), use_cache=True)
        greedy_decode_from_prefill(warm, n_tokens=n_warmup_tokens)
        del warm
        torch.cuda.empty_cache()

        cache = make_cache()
        with torch.no_grad():
            sh_warm = model(ids, attention_mask=full_attention_mask(ids),
                            use_cache=True, past_key_values=cache)
        force_compress_cache(cache)
        greedy_decode_from_prefill(sh_warm, cache=cache, n_tokens=n_warmup_tokens)
        del sh_warm
        del cache
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        profile_path = "/results/shard_decode_profile.json"
        table_path = "/results/shard_decode_profile_top.txt"
        decode_only_profile_path = "/results/shard_decode_only_profile.json"
        decode_only_table_path = "/results/shard_decode_only_profile_top.txt"
        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        token_matches = None
        shard_cr = None
        full_fused_attempts = 0
        full_fused_hits = 0
        full_fused_errors = 0
        last_full_fused_error = None
        if not profile_decode_only:
            with torch.profiler.profile(
                activities=activities,
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
            ) as prof:
                with torch.profiler.record_function("fp16_prefill"):
                    with torch.no_grad():
                        fp_prefill = model(ids, attention_mask=full_attention_mask(ids),
                                           use_cache=True)
                with torch.profiler.record_function("fp16_decode_loop"):
                    fp_tokens = greedy_decode_from_prefill(fp_prefill,
                                                           n_tokens=n_profile_tokens)
                del fp_prefill
                torch.cuda.empty_cache()

                cache = make_cache()
                with torch.profiler.record_function("shard_prefill"):
                    with torch.no_grad():
                        sh_prefill = model(ids, attention_mask=full_attention_mask(ids),
                                           use_cache=True, past_key_values=cache)
                with torch.profiler.record_function("shard_compress_once"):
                    force_compress_cache(cache)
                with torch.profiler.record_function("shard_decode_loop"):
                    sh_tokens = greedy_decode_from_prefill(sh_prefill, cache=cache,
                                                           n_tokens=n_profile_tokens)
                shard_cr = cache.clean_compress_ratio()
                shard_runtime_cr = cache.compress_ratio()
                full_fused_attempts = getattr(cache, "_full_fused_attempts", 0)
                full_fused_hits = getattr(cache, "_full_fused_hits", 0)
                full_fused_errors = getattr(cache, "_full_fused_errors", 0)
                last_full_fused_error = getattr(cache, "_last_full_fused_error", None)
                if full_fused:
                    print("  full-fused counters: "
                          f"attempts={full_fused_attempts} "
                          f"hits={full_fused_hits} "
                          f"errors={full_fused_errors}", flush=True)
                    if last_full_fused_error:
                        print(f"  last full-fused error: {last_full_fused_error}",
                              flush=True)
                del sh_prefill
                del cache
                torch.cuda.empty_cache()

            prof.export_chrome_trace(profile_path)
            top_ops = prof.key_averages().table(
                sort_by="cuda_time_total",
                row_limit=40,
            )
            with open(table_path, "w") as f:
                f.write(top_ops)
                f.write("\n")
            token_matches = sum(1 for a, b in zip(fp_tokens, sh_tokens) if a == b)

        # A second, cleaner trace: compression/prefill happen before profiling, so
        # this table only shows the decode loops we are trying to optimize.
        with torch.no_grad():
            fp_decode_prefill = model(ids, attention_mask=full_attention_mask(ids),
                                      use_cache=True)
        cache = make_cache()
        with torch.no_grad():
            sh_decode_prefill = model(ids, attention_mask=full_attention_mask(ids),
                                      use_cache=True, past_key_values=cache)
        force_compress_cache(cache)
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as decode_prof:
            with torch.profiler.record_function("fp16_decode_only"):
                fp_decode_only_tokens = greedy_decode_from_prefill(
                    fp_decode_prefill, n_tokens=n_profile_tokens)
            with torch.profiler.record_function("shard_decode_only"):
                sh_decode_only_tokens = greedy_decode_from_prefill(
                    sh_decode_prefill, cache=cache, n_tokens=n_profile_tokens)
        shard_runtime_cr = locals().get("shard_runtime_cr")
        if shard_cr is None:
            shard_cr = cache.clean_compress_ratio()
            shard_runtime_cr = cache.compress_ratio()
        full_fused_attempts = max(
            full_fused_attempts, getattr(cache, "_full_fused_attempts", 0))
        full_fused_hits = max(
            full_fused_hits, getattr(cache, "_full_fused_hits", 0))
        full_fused_errors = max(
            full_fused_errors, getattr(cache, "_full_fused_errors", 0))
        last_full_fused_error = (
            last_full_fused_error
            or getattr(cache, "_last_full_fused_error", None)
        )
        decode_prof.export_chrome_trace(decode_only_profile_path)
        decode_top_ops = decode_prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=40,
        )
        with open(decode_only_table_path, "w") as f:
            f.write(decode_top_ops)
            f.write("\n")
        decode_only_token_matches = sum(
            1 for a, b in zip(fp_decode_only_tokens, sh_decode_only_tokens) if a == b)
        del fp_decode_prefill
        del sh_decode_prefill
        del cache
        torch.cuda.empty_cache()

        results["profile"]["summary"] = {
            "ctx_len": ctx_len,
            "decode_tokens": n_profile_tokens,
            "warmup_tokens": n_warmup_tokens,
            "full_fused": full_fused,
            "full_fused_attempts": full_fused_attempts,
            "full_fused_hits": full_fused_hits,
            "full_fused_errors": full_fused_errors,
            "last_full_fused_error": last_full_fused_error,
            "profile_path": profile_path,
            "top_ops_path": table_path,
            "decode_only_profile_path": decode_only_profile_path,
            "decode_only_top_ops_path": decode_only_table_path,
            "compression_ratio": shard_cr,
            "compression_ratio_kind": "clean_storage",
            "runtime_compression_ratio": shard_runtime_cr,
            "compression_is_profiled_separately": True,
            "decode_token_matches": token_matches,
            "decode_token_match_rate": (
                None if token_matches is None else token_matches / n_profile_tokens),
            "decode_only_token_matches": decode_only_token_matches,
            "decode_only_token_match_rate": (
                decode_only_token_matches / n_profile_tokens),
        }
        if not profile_decode_only:
            results["profile"]["records"].append({
                "name": "decode_trace",
                "chrome_trace": profile_path,
                "top_ops": table_path,
            })
        results["profile"]["records"].append({
            "name": "decode_only_trace",
            "chrome_trace": decode_only_profile_path,
            "top_ops": decode_only_table_path,
        })
        result_vol.commit()
        if not profile_decode_only:
            print(f"  wrote trace: {profile_path}", flush=True)
            print(f"  wrote top ops: {table_path}", flush=True)
        print(f"  wrote decode-only trace: {decode_only_profile_path}", flush=True)
        print(f"  wrote decode-only top ops: {decode_only_table_path}", flush=True)
        checkpoint("profile")

    # ==============================================================
    # SUMMARY TABLE
    # ==============================================================
    print(f"\n{'='*72}")
    print(f"  FINAL SHARD vs TURBOQUANT SUMMARY")
    print(f"{'='*72}")
    print(f"\n  NIAH (TurboQuant=0.997 at ~4-6x):")
    for k, (r, c) in niah.items():
        print(f"    {k:>6}: Shard recall={r:.3f} at {c:.1f}x")

    if lb_results:
        avg_fp = sum(v[0] for v in lb_results.values()) / len(lb_results)
        avg_cp = sum(v[1] for v in lb_results.values()) / len(lb_results)
        avg_cr = sum(v[2] for v in lb_results.values()) / len(lb_results)
        results["longbench_e"]["macro_average"] = {
            "tasks": len(lb_results),
            "fp16": avg_fp,
            "shard": avg_cp,
            "delta": avg_cp - avg_fp,
            "compression_ratio_mean": avg_cr,
        }
        print(f"\n  LongBench-E average ({len(lb_results)} tasks):")
        print(f"    FP16: {avg_fp:.2f} | Shard: {avg_cp:.2f} "
              f"(Δ={avg_cp-avg_fp:+.2f}) at {avg_cr:.1f}x")
    if tput:
        print(f"\n  Decode throughput (tok/s):")
        for k, (fp_tps, sh_tps, cr) in tput.items():
            print(f"    ctx={k:>5}: FP16={fp_tps:6.2f}  Shard={sh_tps:6.2f}  "
                  f"({sh_tps/fp_tps:.2f}× FP16, {cr:.1f}× memory)")
    print(f"\n{'='*72}\n")
    checkpoint("final")
    return {
        "status": "complete",
        "latest_partial": "/results/shard_e2e_benchmark_latest_partial.json",
        "final": "/results/shard_e2e_benchmark_final.json",
        "last_completed_section": results["metadata"].get("last_completed_section"),
    }


@app.local_entrypoint()
def main(
    sections: str = "all",
    longbench_samples_per_task: int = 50,
    longbench_tasks: str = "all",
    full_fused: bool = False,
    profile_ctx_len: int = 8192,
    profile_tokens: int = 16,
    profile_warmup_tokens: int = 4,
    profile_decode_only: bool = False,
    background: bool = False,
):
    kwargs = {
        "sections": sections,
        "longbench_samples_per_task": longbench_samples_per_task,
        "longbench_tasks": longbench_tasks,
        "full_fused": full_fused,
        "profile_ctx_len": profile_ctx_len,
        "profile_tokens": profile_tokens,
        "profile_warmup_tokens": profile_warmup_tokens,
        "profile_decode_only": profile_decode_only,
    }
    if background:
        call = run.spawn(**kwargs)
        print("Spawned background benchmark.")
        print(f"Function call: {call.object_id}")
        print("Progress checkpoints will be written to Modal Volume "
              "`shard-benchmark-results`.")
        return

    results = run.remote(**kwargs)
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True).strip())
    except Exception:
        sha = None
        dirty = None
    if "metadata" not in results:
        print("Remote benchmark completed.")
        print(f"Latest checkpoint: {results.get('latest_partial')}")
        print(f"Final checkpoint: {results.get('final')}")
        return

    results["metadata"]["git"] = {"sha": sha, "dirty": dirty}
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_path = out_dir / f"shard_e2e_benchmark_{ts}.json"
    latest_path = out_dir / "shard_e2e_benchmark_latest.json"
    text = json.dumps(results, indent=2)
    out_path.write_text(text + "\n")
    latest_path.write_text(text + "\n")
    print(f"Wrote {out_path}")
