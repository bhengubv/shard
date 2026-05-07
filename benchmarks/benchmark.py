"""End-to-end benchmark: NIAH + LongBench-E on Llama-3.1-8B-Instruct. Modal+B200."""
import modal
import os

app = modal.App("shard-e2e")
vol = modal.Volume.from_name("hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "accelerate", "datasets",
                 "sentencepiece", "protobuf", "numpy",
                 "rouge-score", "pandas", "pyarrow", "huggingface-hub")
    .add_local_dir("src/shard", remote_path="/root/shard")
)


@app.function(image=image, gpu="B200", timeout=14400,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              memory=131072, volumes={"/hf-cache": vol})
def run():
    import sys, re, string, math, json, zipfile
    sys.path.insert(0, "/root")
    os.environ["HF_HOME"] = "/hf-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/hf-cache"

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

    tok = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16, device_map="auto", token=hf_token)
    model.eval()
    vol.commit()

    enable_llama_fused_attention(model)
    inv_freq = get_rope_inv_freq(model)
    hd = getattr(model.config, "head_dim",
                 model.config.hidden_size // model.config.num_attention_heads)
    nl = model.config.num_hidden_layers
    nkv = getattr(model.config, "num_key_value_heads",
                  model.config.num_attention_heads)

    def make_cache():
        return Cache(head_dim=hd, num_layers=nl, n_kv_heads=nkv,
                     device=device, k_target_cr=20, max_rank_k=192,
                     v_vq_group_size=4, v_vq_codebook_size=256,
                     residual_window=64, rope_inv_freq=inv_freq)

    # NIAH
    print(f"\n--- NIAH ---")
    fw = load_dataset("HuggingFaceFW/fineweb", "sample-10BT",
                      split="train", streaming=True)
    texts = []
    for item in fw:
        texts.append(item["text"])
        if len(texts) >= 80: break
    haystack = " ".join(texts)
    hay_ids = tok(haystack, return_tensors="pt").input_ids[0]

    needle = ("\nThe best thing to do in San Francisco is eat a sandwich "
              "and sit in Dolores Park on a sunny day.\n")
    question = "\n\nWhat is the best thing to do in San Francisco?\nAnswer:"
    needle_ids = tok(needle, return_tensors="pt").input_ids[0]
    q_ids = tok(question, return_tensors="pt").input_ids[0]

    for ctx_len in [4096, 8192, 16384, 32768]:
        if hay_ids.shape[0] < ctx_len: continue
        hits, total, crs = 0, 0, []
        for depth in [0.1, 0.3, 0.5, 0.7, 0.9]:
            budget = ctx_len - needle_ids.shape[0] - q_ids.shape[0] - 10
            pos = int(budget * depth)
            full = torch.cat([hay_ids[:pos], needle_ids,
                              hay_ids[pos:budget], q_ids]).unsqueeze(0).to(device)
            cache = make_cache()
            with torch.no_grad():
                out = model.generate(full, max_new_tokens=30, do_sample=False,
                                      past_key_values=cache,
                                      pad_token_id=tok.eos_token_id)
            ans = tok.decode(out[0][full.shape[1]:], skip_special_tokens=True).lower()
            hit = "sandwich" in ans and ("dolores" in ans or "park" in ans)
            hits += int(hit); total += 1
            crs.append(cache.compress_ratio())
            del cache; torch.cuda.empty_cache()
        rec = hits / total
        print(f"  {ctx_len:>6}: recall={rec:.3f}  cr={sum(crs)/len(crs):.1f}x", flush=True)

    # LongBench-E
    print(f"\n--- LONGBENCH-E ---")

    def normalize(s):
        s = s.lower()
        s = re.sub(r"\b(a|an|the)\b", " ", s)
        s = "".join(ch for ch in s if ch not in string.punctuation)
        return re.sub(r"\s+", " ", s).strip()

    def f1_score(pred, truths):
        pred = normalize(pred); best = 0.0
        for t in truths if isinstance(truths, (list, tuple)) else [truths]:
            t = normalize(str(t))
            if not pred or not t: continue
            pt, tt = pred.split(), t.split()
            common = set(pt) & set(tt)
            if not common: continue
            ncom = sum(min(pt.count(w), tt.count(w)) for w in common)
            if ncom == 0: continue
            p, r = ncom / len(pt), ncom / len(tt)
            best = max(best, 2 * p * r / (p + r))
        return best

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    def rouge_l(pred, truths):
        best = 0.0
        for t in truths if isinstance(truths, (list, tuple)) else [truths]:
            best = max(best, scorer.score(str(t), pred)['rougeL'].fmeasure)
        return best

    zip_path = hf_hub_download(repo_id="zai-org/LongBench", repo_type="dataset",
                                filename="data.zip", token=hf_token)
    zf = zipfile.ZipFile(zip_path, "r")
    names = set(zf.namelist())

    def load_task(task):
        for suf in [f"{task}_e", task]:
            p = f"data/{suf}.jsonl"
            if p in names:
                with zf.open(p) as f:
                    rows = [json.loads(line) for line in f if line.strip()]
                return pd.DataFrame(rows)
        return None

    tasks = [("qasper", "f1", 128), ("multifieldqa_en", "f1", 64),
             ("hotpotqa", "f1", 32), ("2wikimqa", "f1", 32),
             ("gov_report", "rouge", 512), ("multi_news", "rouge", 512),
             ("triviaqa", "f1", 32), ("samsum", "rouge", 128)]
    MAX_CTX = 7500

    for task_name, metric, max_new in tasks:
        df = load_task(task_name)
        if df is None: continue
        n = min(15, len(df))
        fp_scores, cp_scores, crs = [], [], []
        for i in range(n):
            ex = df.iloc[i]
            prompt = str(ex.get("input", ex.get("prompt", "")))
            context = str(ex.get("context", ""))
            ans = ex["answers"] if "answers" in ex else ex.get("answer", "")
            if hasattr(ans, 'tolist'): ans = ans.tolist()
            if not isinstance(ans, (list, tuple)): ans = [ans]
            full_text = (f"{context}\n\n{prompt}\n\nAnswer:"
                         if context else f"{prompt}\n\nAnswer:")
            ids = tok(full_text, return_tensors="pt", truncation=True,
                       max_length=MAX_CTX).input_ids.to(device)
            if ids.shape[1] < 512: continue

            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                                      pad_token_id=tok.eos_token_id)
            fp_pred = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

            cache = make_cache()
            with torch.no_grad():
                outc = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                                       pad_token_id=tok.eos_token_id,
                                       past_key_values=cache)
            cp_pred = tok.decode(outc[0][ids.shape[1]:], skip_special_tokens=True)
            crs.append(cache.compress_ratio())
            del cache; torch.cuda.empty_cache()

            scorer_fn = f1_score if metric == "f1" else rouge_l
            fp_scores.append(scorer_fn(fp_pred, ans))
            cp_scores.append(scorer_fn(cp_pred, ans))

        if not fp_scores: continue
        fp_avg = 100 * sum(fp_scores) / len(fp_scores)
        cp_avg = 100 * sum(cp_scores) / len(cp_scores)
        cr_avg = sum(crs) / len(crs) if crs else 0
        print(f"  {task_name:<20}: FP16={fp_avg:5.2f}  "
              f"Shard={cp_avg:5.2f}  Δ={cp_avg-fp_avg:+5.2f}  "
              f"cr={cr_avg:.1f}x", flush=True)


@app.local_entrypoint()
def main():
    run.remote()
