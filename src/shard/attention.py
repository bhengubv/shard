"""Fused compressed-K attention.
Q_rope(p_q) · K_rope(p_t) on int4 PCA coefficients without materializing FP16 K.
"""
import math
import torch

from .rope import rotate_half

def _unpack_int4(packed, n_total, rank):
    lo = (packed & 0x0F).to(torch.float16) - 8.0
    hi = ((packed >> 4) & 0x0F).to(torch.float16) - 8.0
    return torch.stack([lo, hi], dim=1).flatten()[:n_total * rank].reshape(n_total, rank)

def compressed_scores(q_rope, gc, pos_q, n_q, nkv, hd, inv_freq):
    B = q_rope.shape[0]
    dev = q_rope.device
    n_total, rank = gc["k_coeffs_shape"]
    n_comp = gc["n_compress"]
    gqa = n_q // nkv
    hd2 = hd // 2

    freqs = torch.tensor([pos_q], device=dev).float().unsqueeze(-1) * inv_freq.to(dev).float()
    cos_q = torch.cat([freqs, freqs], dim=-1).cos().to(torch.float16).view(1, 1, 1, hd)
    sin_q = torch.cat([freqs, freqs], dim=-1).sin().to(torch.float16).view(1, 1, 1, hd)
    q_n = q_rope * cos_q - rotate_half(q_rope) * sin_q
    a, b = q_n[..., :hd2].contiguous(), q_n[..., hd2:].contiguous()

    k_basis = gc["k_basis_q"].to(torch.float16) * gc["k_basis_scale"].to(torch.float16)
    bph = k_basis.view(nkv, hd, rank)
    bc, bd = bph[:, :hd2, :], bph[:, hd2:, :]
    mean = gc["k_mean"].to(torch.float16).view(nkv, hd)
    mc, md = mean[:, :hd2], mean[:, hd2:]
    kv_idx = torch.arange(n_q, device=dev) // gqa
    bc, bd, mc, md = bc[kv_idx], bd[kv_idx], mc[kv_idx], md[kv_idx]

    coeff_scale = gc["k_coeff_scale"].to(torch.float16)
    bc_e, bd_e = bc.unsqueeze(0).unsqueeze(2), bd.unsqueeze(0).unsqueeze(2)
    A = (a.unsqueeze(-1) * bc_e + b.unsqueeze(-1) * bd_e) * coeff_scale
    Bm = (b.unsqueeze(-1) * bc_e - a.unsqueeze(-1) * bd_e) * coeff_scale
    A_mean = a * mc.unsqueeze(0).unsqueeze(2) + b * md.unsqueeze(0).unsqueeze(2)
    B_mean = b * mc.unsqueeze(0).unsqueeze(2) - a * md.unsqueeze(0).unsqueeze(2)

    # Fast path (B=1, Triton available): skip int4 unpack, matmul-in-kernel
    from .triton_kernels import int4_matmul, HAS_TRITON
    if B == 1 and HAS_TRITON and q_rope.is_cuda:
        half_rank = (rank + 1) // 2
        b_packed = gc["k_coeffs"].view(n_comp, half_rank)
        M = n_q * hd2
        A_nos = (a.unsqueeze(-1) * bc_e + b.unsqueeze(-1) * bd_e)
        Bm_nos = (b.unsqueeze(-1) * bc_e - a.unsqueeze(-1) * bd_e)
        u_flat = int4_matmul(A_nos.view(M, rank), b_packed, coeff_scale, n_comp, rank)
        v_flat = int4_matmul(Bm_nos.view(M, rank), b_packed, coeff_scale, n_comp, rank)
        U = u_flat.view(1, n_q, 1, hd2, n_comp) + A_mean.unsqueeze(-1)
        V = v_flat.view(1, n_q, 1, hd2, n_comp) + B_mean.unsqueeze(-1)
    else:
        coefs = _unpack_int4(gc["k_coeffs"], n_total, rank).to(dev).view(B, n_comp, rank)
        U = torch.einsum("bhspr,bcr->bhspc", A, coefs) + A_mean.unsqueeze(-1)
        V = torch.einsum("bhspr,bcr->bhspc", Bm, coefs) + B_mean.unsqueeze(-1)

    pos_t = torch.arange(gc["compress_start"], gc["compress_start"] + n_comp, device=dev)
    phase = (pos_t - pos_q).float().unsqueeze(-1) * inv_freq.to(dev).float()
    cos_d = phase.cos().to(torch.float16)
    sin_d = phase.sin().to(torch.float16)
    scores = torch.einsum("bhspc,cp->bhsc", U, cos_d) + torch.einsum("bhspc,cp->bhsc", V, sin_d)
    return scores / math.sqrt(hd)

def _v_weighted_sum_compressed(weights_mid, gc, B, nkv, n_comp, hd):
    """Fused V path: codebook lookup -> weighted sum -> inverse Hadamard at the end."""
    from .cache import hadamard
    from .rope import unpack_nbits

    qd = gc["v_qdata"]
    gs = qd["group_size"]
    ng = hd // gs
    idx = unpack_nbits(qd["packed"].to(weights_mid.device), qd["bits"], qd["n_orig"])
    idx = idx[:B * nkv * n_comp * ng].view(B * nkv, n_comp, ng).long()
    centroids = qd["centroids"].to(weights_mid.device).float()
    ch_max = qd["ch_maxabs"].to(weights_mid.device).float()
    gqa = weights_mid.shape[1] // nkv

    v_rot = centroids[idx].view(B * nkv, n_comp, hd) * ch_max.unsqueeze(1)
    v_rot = v_rot.view(B, nkv, n_comp, hd).repeat_interleave(gqa, dim=1)
    out_rot = torch.einsum("bhsk,bhkd->bhsd", weights_mid, v_rot)
    return hadamard(out_rot.to(torch.float16)).float()


def fused_attention_compressed(q_rope, layer, pos_q, inv_freq, attn_mask):
    """Fused-K scoring + fused-V weighted sum (sink/window/decode/stream stay fp16)."""
    gc = layer._gpu_compressed
    if gc is None: return None
    B, n_q, _, hd = q_rope.shape
    nkv = layer._cache_ref._n_kv_heads
    n_comp = gc["n_compress"]
    gqa = n_q // nkv

    def _score(q, k):
        return torch.einsum("bhsd,bhkd->bhsk",
                             q.float(),
                             k.repeat_interleave(gqa, dim=1).float()) / math.sqrt(hd)

    comp_scores = compressed_scores(q_rope, gc, pos_q, n_q, nkv, hd, inv_freq).float()
    parts = [_score(q_rope, gc["sink_k"]), comp_scores, _score(q_rope, gc["win_k"])]
    if "decode_k" in gc:
        parts.append(_score(q_rope, gc["decode_k"]))
    if gc.get("stream_n", 0) > 0:
        sk, _ = layer._dequant_stream(gc, B, nkv, hd)
        parts.append(_score(q_rope, sk))

    scores = torch.cat(parts, dim=-1)
    if attn_mask is not None: scores = scores + attn_mask
    weights = torch.softmax(scores, dim=-1, dtype=torch.float32)

    sink_n = gc["sink_k"].shape[2]
    win_n = gc["win_k"].shape[2]
    w_sink = weights[..., :sink_n]
    w_comp = weights[..., sink_n:sink_n + n_comp]
    w_win = weights[..., sink_n + n_comp:sink_n + n_comp + win_n]
    off = sink_n + n_comp + win_n

    out = _v_weighted_sum_compressed(w_comp, gc, B, nkv, n_comp, hd)
    def _add_fp16(w, v):
        v_exp = v.repeat_interleave(gqa, dim=1).float()
        return torch.einsum("bhsk,bhkd->bhsd", w, v_exp)

    out = out + _add_fp16(w_sink, gc["sink_v"])
    out = out + _add_fp16(w_win, gc["win_v"])
    if "decode_v" in gc:
        dv = gc["decode_v"]
        out = out + _add_fp16(weights[..., off:off + dv.shape[2]], dv)
        off += dv.shape[2]
    if gc.get("stream_n", 0) > 0:
        _, sv = layer._dequant_stream(gc, B, nkv, hd)
        out = out + _add_fp16(weights[..., off:off + sv.shape[2]], sv)
    return out.to(q_rope.dtype)


def enable_llama_fused_attention(model):
    """Monkey-patch LlamaAttention.forward to route Cache through fused path on decode."""
    from .cache import Cache
    try:
        from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb
    except ImportError:
        return False

    original = LlamaAttention.forward

    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                 past_key_values=None, cache_position=None, **kwargs):
        is_shard = isinstance(past_key_values, Cache)
        layer = past_key_values.layers[self.layer_idx] if is_shard else None
        fuse = (is_shard and layer is not None
                and getattr(layer, "_gpu_compressed", None) is not None
                and hidden_states.shape[1] == 1
                and past_key_values._rope_inv_freq is not None)
        if not fuse:
            return original(self, hidden_states=hidden_states,
                             position_embeddings=position_embeddings,
                             attention_mask=attention_mask,
                             past_key_values=past_key_values,
                             cache_position=cache_position, **kwargs)

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        q = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        k = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        past_key_values.update(k, v, self.layer_idx,
                                {"sin": sin, "cos": cos, "cache_position": cache_position})
        pos_q = int(cache_position[-1].item()) if cache_position is not None \
                else past_key_values.get_seq_length() - 1
        out = fused_attention_compressed(q, layer, pos_q,
                                          past_key_values._rope_inv_freq, attention_mask)
        if out is None:
            return original(self, hidden_states=hidden_states,
                             position_embeddings=position_embeddings,
                             attention_mask=attention_mask,
                             past_key_values=past_key_values,
                             cache_position=cache_position, **kwargs)
        out = out.transpose(1, 2).contiguous().view(*input_shape, -1)
        return self.o_proj(out), None

    LlamaAttention.forward = forward
    return True
