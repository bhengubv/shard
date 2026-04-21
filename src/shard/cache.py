"""KV cache compression: per-layer online PCA for K + Hadamard+VQ for V."""
import math
import torch
from transformers.cache_utils import Cache as HFCache, CacheLayerMixin

from .rope import (dp_bit_allocation, quantize_group, dequantize_group,
                   undo_rope, reapply_rope, get_rope_inv_freq,
                   pack_nbits, unpack_nbits)

def hadamard(x):
    d = x.shape[-1]
    h = 1
    while h < d:
        x = x.unflatten(-1, (-1, h * 2))
        a, b = x[..., :h], x[..., h:]
        x = torch.cat([a + b, a - b], dim=-1).flatten(-2)
        h *= 2
    return x / math.sqrt(d)

def kmeans(data, k, n_iter=30, seed=42):
    n, d = data.shape
    dev = data.device
    g = torch.Generator(device=dev).manual_seed(seed)
    centroids = data[torch.randperm(n, generator=g, device=dev)[:k]].clone()
    for it in range(n_iter):
        idx = torch.cdist(data, centroids).argmin(dim=1)
        new = torch.zeros_like(centroids)
        counts = torch.zeros(k, device=dev, dtype=data.dtype)
        new.index_add_(0, idx, data)
        counts.index_add_(0, idx, torch.ones(n, device=dev, dtype=data.dtype))
        nz = counts > 0
        new[nz] /= counts[nz].unsqueeze(1)
        new[~nz] = centroids[~nz]
        if (new - centroids).abs().max().item() < 1e-4 and it >= 5:
            centroids = new; break
        centroids = new
    return centroids, idx

def vq_encode(v, group_size, codebook_size):
    bh, seq, hd = v.shape
    ng = hd // group_size
    ch_max = v.abs().amax(dim=1).clamp(min=1e-8)
    flat = (v / ch_max.unsqueeze(1)).reshape(bh * seq, ng, group_size).reshape(-1, group_size)
    centroids, idx = kmeans(flat.float(), codebook_size, n_iter=30)
    bits = max(1, (codebook_size - 1).bit_length())
    packed, n_orig = pack_nbits(idx.to(torch.uint8).long(), bits)
    return {"centroids": centroids.half(), "packed": packed,
            "ch_maxabs": ch_max.half(), "shape": (bh, seq, hd),
            "group_size": group_size, "bits": bits, "n_orig": n_orig}

def vq_decode(qd, device):
    bh, seq, hd = qd["shape"]
    gs = qd["group_size"]
    idx = unpack_nbits(qd["packed"].to(device), qd["bits"], qd["n_orig"])
    idx = idx[:bh * seq * (hd // gs)].long()
    centroids = qd["centroids"].to(device).float()
    ch_max = qd["ch_maxabs"].to(device).float()
    return centroids[idx].reshape(bh, seq, hd) * ch_max.unsqueeze(1)


class CacheLayer(CacheLayerMixin):
    def __init__(self, layer_idx):
        self.layer_idx = layer_idx
        self._keys = None
        self._values = None
        self._prefilled = False
        self._seq_len = 0
        self._cache_ref = None
        self._needs_decompress = False

    def _lazy_evict(self):
        c = self._cache_ref
        if c is None or not c._lazy_per_layer: return
        last = c._last_accessed_layer
        if last != -1 and last != self.layer_idx:
            prev = c.layers[last]
            if getattr(prev, "_gpu_compressed", None) is not None:
                prev._keys = None
                prev._values = None
                prev._needs_decompress = True
        c._last_accessed_layer = self.layer_idx

    def _ensure_dequant(self):
        if not self._needs_decompress: return
        gc = getattr(self, "_gpu_compressed", None)
        if gc is None: return

        packed = gc["k_coeffs"]
        lo = (packed & 0x0F).to(torch.float16) - 8.0
        hi = ((packed >> 4) & 0x0F).to(torch.float16) - 8.0
        flat = torch.stack([lo, hi], dim=1).flatten()
        n_total, rank = gc["k_coeffs_shape"]
        coeffs = flat[:n_total * rank].reshape(n_total, rank) * gc["k_coeff_scale"].half()
        basis = gc["k_basis_q"].to(torch.float16) * gc["k_basis_scale"].half()
        recon = coeffs @ basis.T + gc["k_mean"].half()
        del basis, coeffs, flat, lo, hi

        B, n_comp = gc["batch"], gc["n_compress"]
        nkv, hd = self._cache_ref._n_kv_heads, self._cache_ref.head_dim
        k_layer = recon.view(B, n_comp, nkv, hd)
        del recon

        if self._cache_ref._rope_inv_freq is not None:
            pos = torch.arange(gc["compress_start"], gc["compress_start"] + n_comp, device=packed.device)
            k_mid = reapply_rope(k_layer.permute(0, 2, 1, 3), pos, self._cache_ref._rope_inv_freq).half()
        else:
            k_mid = k_layer.permute(0, 2, 1, 3).half()

        v_mid = hadamard(vq_decode(gc["v_qdata"], str(packed.device))).view(B, nkv, n_comp, hd).half()

        parts_k = [gc["sink_k"], k_mid, gc["win_k"]]
        parts_v = [gc["sink_v"], v_mid, gc["win_v"]]
        if "decode_k" in gc: parts_k.append(gc["decode_k"]); parts_v.append(gc["decode_v"])

        self._keys = torch.cat(parts_k, dim=2)
        self._values = torch.cat(parts_v, dim=2)
        self._needs_decompress = False

    @property
    def keys(self):
        self._lazy_evict(); self._ensure_dequant(); return self._keys
    @keys.setter
    def keys(self, v): self._keys = v

    @property
    def values(self):
        self._lazy_evict(); self._ensure_dequant(); return self._values
    @values.setter
    def values(self, v): self._values = v

    def update(self, key_states, value_states, *args, **kwargs):
        seq_new = key_states.shape[2]
        c = self._cache_ref

        if seq_new > 1 and not self._prefilled:
            self._prefilled = True
            self._seq_len = seq_new
            self._keys = key_states
            self._values = value_states
            return key_states, value_states

        if self.layer_idx == 0 and c is not None and not c._compressed:
            c.compress()

        self._lazy_evict()
        gc = getattr(self, "_gpu_compressed", None)

        if gc is not None:
            if "decode_k" in gc:
                gc["decode_k"] = torch.cat([gc["decode_k"], key_states], dim=2)
                gc["decode_v"] = torch.cat([gc["decode_v"], value_states], dim=2)
            else:
                gc["decode_k"] = key_states
                gc["decode_v"] = value_states
            self._seq_len += seq_new
            self._needs_decompress = True
            self._ensure_dequant()
            return self._keys, self._values

        self._ensure_dequant()
        self._seq_len += seq_new
        if self._keys is not None:
            self._keys = torch.cat([self._keys, key_states], dim=2)
            self._values = torch.cat([self._values, value_states], dim=2)
        else:
            self._keys = key_states
            self._values = value_states
        return self._keys, self._values

    def get_seq_length(self): return self._seq_len
    def get_max_cache_shape(self): return None
    def get_mask_sizes(self, query_length): return self._seq_len + query_length, 0
    def lazy_initialization(self, *args, **kwargs): pass

    def reorder_cache(self, beam_idx):
        if self._needs_decompress: _ = self.keys
        if self._keys is not None:
            self._keys = self._keys.index_select(0, beam_idx.to(self._keys.device))
        if self._values is not None:
            self._values = self._values.index_select(0, beam_idx.to(self._values.device))


class Cache(HFCache):
    """KV cache compression. Prefill: per-layer online PCA on K + Hadamard+VQ on V."""
    def __init__(self, head_dim=128, num_layers=32, n_kv_heads=8, device="cuda",
                 k_target_cr=20.0, max_rank_k=192,
                 v_vq_group_size=4, v_vq_codebook_size=256,
                 sink_tokens=4, residual_window=64, rope_inv_freq=None):
        self.head_dim = head_dim
        self._num_layers = num_layers
        self._n_kv_heads = n_kv_heads
        self._per_layer_dim = n_kv_heads * head_dim
        self._k_target_cr = k_target_cr
        self._max_rank_k = max_rank_k
        self._v_vq_group_size = v_vq_group_size
        self._v_vq_codebook_size = v_vq_codebook_size
        self._sink_tokens = sink_tokens
        self._residual_window = residual_window
        self._rope_inv_freq = rope_inv_freq
        self._lazy_per_layer = True
        self._last_accessed_layer = -1
        self._compressed = False
        self._compressed_layers = {}
        super().__init__(layers=[CacheLayer(i) for i in range(num_layers)])
        for l in self.layers:
            l._cache_ref = self

    @classmethod
    def from_model(cls, model, k_target_cr=16.0, **kwargs):
        cfg = model.config
        hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        return cls(head_dim=hd, num_layers=cfg.num_hidden_layers,
                    n_kv_heads=getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
                    device=str(next(model.parameters()).device),
                    k_target_cr=k_target_cr,
                    rope_inv_freq=get_rope_inv_freq(model), **kwargs)

    def compress(self):
        if self._compressed: return
        k0 = self.layers[0].keys
        if k0 is None: return
        B, _, seq, _ = k0.shape
        sink, win = self._sink_tokens, self._residual_window
        if seq <= sink + win + 4: return

        start, end = sink, seq - win
        n_comp = end - start
        positions = torch.arange(seq, device=k0.device)[start:end]

        for li in range(self._num_layers):
            torch.manual_seed(42 + li)
            layer = self.layers[li]
            k_full = layer.keys
            k = self._compress_k(k_full[:, :, start:end, :], positions, B, n_comp)
            self._compressed_layers[li] = {
                **k,
                "sink_k": k_full[:, :, :sink, :].clone(),
                "win_k": k_full[:, :, end:, :].clone(),
                "n_compress": n_comp, "batch": B, "compress_start": start,
            }

        self._compressed = True

    def _compress_k(self, k_mid, positions, B, n_comp):
        k = (undo_rope(k_mid.float(), positions, self._rope_inv_freq)
             if self._rope_inv_freq is not None else k_mid.float())
        flat = k.permute(0, 2, 1, 3).reshape(B * n_comp, self._per_layer_dim)
        n, p = flat.shape
        mean = flat.mean(dim=0)
        rank = min(self._max_rank_k, p, n - 1)
        _, S, V = torch.svd_lowrank(flat - mean, q=rank, niter=24)
        basis = V[:, :rank]
        eigvals = (S[:rank].float() ** 2) / n

        budget = max(64, int(self._per_layer_dim * 16 / self._k_target_cr))
        alloc = dp_bit_allocation(eigvals, budget)
        keep = sum(1 for b in alloc if b > 0)
        if keep > 0:
            basis = basis[:, :keep].contiguous()
            alloc = alloc[:keep]

        col_max = basis.abs().max(dim=0).values.clamp(min=1e-6)
        scale = col_max / 127.0
        basis_q = (basis / scale).round().clamp(-127, 127).to(torch.int8)
        proj = (flat - mean.float()) @ (basis_q.float() * scale.float())[:, :keep]
        return {"k_basis_q": basis_q, "k_scale": scale.half(),
                "k_mean": mean.half(), "k_alloc": alloc,
                "k_bytes": quantize_group(proj, alloc),
                "k_n_total": B * n_comp}
