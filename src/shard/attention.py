"""Fused compressed-K attention.
Computes Q_rope(p_q) · K_rope(p_t) on int4 PCA coefficients without materializing
FP16 K. For rotate_half RoPE, splitting q = [a, b], k_n = [c, d]:
    pair_i(Δ) = (a_i c_i + b_i d_i) cos(θ_i Δ) + (b_i c_i − a_i d_i) sin(θ_i Δ)
With k_n = K_coefs @ K_basis.T + K_mean, both become rank-R inner products
against int4 K_coefs plus a mean term. V takes the standard fp16 path.
"""
import math
import torch

from .rope import rotate_half

def _unpack_int4(packed, n_total, rank):
    lo = (packed & 0x0F).to(torch.float16) - 8.0
    hi = ((packed >> 4) & 0x0F).to(torch.float16) - 8.0
    return torch.stack([lo, hi], dim=1).flatten()[:n_total * rank].reshape(n_total, rank)

def compressed_scores(q_rope, gc, pos_q, n_q, nkv, hd, inv_freq):
    # returns (B, n_q, 1, n_comp) fp32 scaled scores
    B = q_rope.shape[0]
    dev = q_rope.device
    n_total, rank = gc["k_coeffs_shape"]
    n_comp = gc["n_compress"]
    gqa = n_q // nkv
    hd2 = hd // 2

    # undo RoPE on Q at pos_q
    freqs = torch.tensor([pos_q], device=dev).float().unsqueeze(-1) * inv_freq.to(dev).float()
    cos_q = torch.cat([freqs, freqs], dim=-1).cos().to(torch.float16).view(1, 1, 1, hd)
    sin_q = torch.cat([freqs, freqs], dim=-1).sin().to(torch.float16).view(1, 1, 1, hd)
    q_n = q_rope * cos_q - rotate_half(q_rope) * sin_q
    a, b = q_n[..., :hd2].contiguous(), q_n[..., hd2:].contiguous()

    # split K_basis into (nkv, hd/2, R) halves; broadcast to Q heads via GQA
    k_basis = gc.get("k_basis_f16")
    if k_basis is None:
        k_basis = gc["k_basis_q"].to(torch.float16) * gc["k_basis_scale"].to(torch.float16)
    bph = k_basis.view(nkv, hd, rank)
    bc, bd = bph[:, :hd2, :], bph[:, hd2:, :]
    mean = gc.get("k_mean_f16")
    if mean is None:
        mean = gc["k_mean"].to(torch.float16)
    mean = mean.view(nkv, hd)
    mc, md = mean[:, :hd2], mean[:, hd2:]
    kv_idx = torch.arange(n_q, device=dev) // gqa
    bc, bd, mc, md = bc[kv_idx], bd[kv_idx], mc[kv_idx], md[kv_idx]  # (n_q, ...)

    coeff_scale = gc.get("k_coeff_scale_f16")
    if coeff_scale is None:
        coeff_scale = gc["k_coeff_scale"]
    bc_e, bd_e = bc.unsqueeze(0).unsqueeze(2), bd.unsqueeze(0).unsqueeze(2)
    A_mean = a * mc.unsqueeze(0).unsqueeze(2) + b * md.unsqueeze(0).unsqueeze(2)
    B_mean = b * mc.unsqueeze(0).unsqueeze(2) - a * md.unsqueeze(0).unsqueeze(2)

    # Fast path (B=1, Triton available): skip int4 unpack, matmul-in-kernel
    from .triton_kernels import int4_matmul, HAS_TRITON
    if B == 1 and HAS_TRITON and q_rope.is_cuda:
        # pack-friendly shape: (n_comp, rank/2) uint8
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
        A = (a.unsqueeze(-1) * bc_e + b.unsqueeze(-1) * bd_e) * coeff_scale
        Bm = (b.unsqueeze(-1) * bc_e - a.unsqueeze(-1) * bd_e) * coeff_scale
        coefs = _unpack_int4(gc["k_coeffs"], n_total, rank).to(dev).view(B, n_comp, rank)
        U = torch.einsum("bhspr,bcr->bhspc", A, coefs) + A_mean.unsqueeze(-1)
        V = torch.einsum("bhspr,bcr->bhspc", Bm, coefs) + B_mean.unsqueeze(-1)

    # per-pair cos(Δ)/sin(Δ) then reduce over pair dim
    pos_t = torch.arange(gc["compress_start"], gc["compress_start"] + n_comp, device=dev)
    phase = (pos_t - pos_q).float().unsqueeze(-1) * inv_freq.to(dev).float()
    cos_d = phase.cos().to(torch.float16)
    sin_d = phase.sin().to(torch.float16)
    scores = torch.einsum("bhspc,cp->bhsc", U, cos_d) + torch.einsum("bhspc,cp->bhsc", V, sin_d)
    return scores / math.sqrt(hd)

def _v_weighted_sum_compressed(weights_mid, gc, B, nkv, n_comp, hd):
    """Fused V path: codebook lookup → weighted sum → inverse Hadamard at the end.
    Hadamard linearity: H(Σ_t w_t · v_t) = Σ_t w_t · H(v_t). On CUDA with B=1 we
    use the all-KV-head vq_weighted_sum kernel (single launch). Otherwise
    fall back to batched PyTorch einsum.
    """
    from .cache import hadamard
    from .rope import unpack_nbits
    from .triton_kernels import HAS_TRITON

    qd = gc["v_qdata"]
    gs = qd["group_size"]
    ng = hd // gs
    gqa = weights_mid.shape[1] // nkv
    S = weights_mid.shape[2]

    idx = unpack_nbits(qd["packed"].to(weights_mid.device), qd["bits"], qd["n_orig"])
    centroids = qd["centroids"].to(weights_mid.device)
    ch_max = qd["ch_maxabs"].to(weights_mid.device)
    idx_l = idx[:B * nkv * n_comp * ng].view(B * nkv, n_comp, ng).long()
    v_rot = centroids.float()[idx_l].view(B * nkv, n_comp, hd) * ch_max.float().unsqueeze(1)
    v_rot = v_rot.view(B, nkv, n_comp, hd).repeat_interleave(gqa, dim=1)
    out_rot = torch.einsum("bhsk,bhkd->bhsd", weights_mid, v_rot)
    return hadamard(out_rot.to(torch.float16)).float()


def _compressed_score_parts(q_rope, gc, pos_q, n_q, nkv, hd, inv_freq):
    """Same as compressed_scores but returns the intermediates needed by the
    full-fused kernel: (U_raw, V_raw, mean_c, mean_s, cos_d, sin_d) — NOT reduced
    yet. U_raw/V_raw are (n_q, hd/2, n_comp) fp16; mean_{c,s} are (n_q, hd/2) fp16;
    cos_d/sin_d are (hd/2, n_comp) fp16.
    """
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
    a, b = q_n[..., :hd2].contiguous(), q_n[..., hd2:].contiguous()  # (1, n_q, 1, hd/2)

    k_basis = gc.get("k_basis_f16")
    if k_basis is None:
        k_basis = gc["k_basis_q"].to(torch.float16) * gc["k_basis_scale"].to(torch.float16)
    bph = k_basis.view(nkv, hd, rank)
    bc, bd = bph[:, :hd2, :], bph[:, hd2:, :]
    mean = gc.get("k_mean_f16")
    if mean is None:
        mean = gc["k_mean"].to(torch.float16)
    mean = mean.view(nkv, hd)
    mc, md = mean[:, :hd2], mean[:, hd2:]
    kv_idx = torch.arange(n_q, device=dev) // gqa
    bc, bd, mc, md = bc[kv_idx], bd[kv_idx], mc[kv_idx], md[kv_idx]

    coeff_scale = gc.get("k_coeff_scale_f16")
    if coeff_scale is None:
        coeff_scale = gc["k_coeff_scale"]
    bc_e, bd_e = bc.unsqueeze(0).unsqueeze(2), bd.unsqueeze(0).unsqueeze(2)
    A_nos = (a.unsqueeze(-1) * bc_e + b.unsqueeze(-1) * bd_e)
    Bm_nos = (b.unsqueeze(-1) * bc_e - a.unsqueeze(-1) * bd_e)
    mean_c = (a * mc.unsqueeze(0).unsqueeze(2)).squeeze(0).squeeze(1) \
             + (b * md.unsqueeze(0).unsqueeze(2)).squeeze(0).squeeze(1)  # (n_q, hd/2)
    mean_s = (b * mc.unsqueeze(0).unsqueeze(2)).squeeze(0).squeeze(1) \
             - (a * md.unsqueeze(0).unsqueeze(2)).squeeze(0).squeeze(1)

    from .triton_kernels import int4_matmul, HAS_TRITON
    half_rank = (rank + 1) // 2
    b_packed = gc["k_coeffs"].view(n_comp, half_rank)
    M = n_q * hd2
    u_flat = int4_matmul(A_nos.view(M, rank), b_packed, coeff_scale, n_comp, rank)
    v_flat = int4_matmul(Bm_nos.view(M, rank), b_packed, coeff_scale, n_comp, rank)
    U_raw = u_flat.view(n_q, hd2, n_comp).contiguous()
    V_raw = v_flat.view(n_q, hd2, n_comp).contiguous()

    pos_t = torch.arange(gc["compress_start"], gc["compress_start"] + n_comp, device=dev)
    phase = (pos_t - pos_q).float().unsqueeze(-1) * inv_freq.to(dev).float()  # (n_comp, hd/2)
    cos_d = phase.cos().to(torch.float16).T.contiguous()  # (hd/2, n_comp)
    sin_d = phase.sin().to(torch.float16).T.contiguous()
    return U_raw, V_raw, mean_c.contiguous(), mean_s.contiguous(), cos_d, sin_d


def fused_attention_compressed_full(q_rope, layer, pos_q, inv_freq, attn_mask):
    """Fully-fused decode attention: one Triton kernel fuses (cos/sin score reduction
    + online softmax + VQ V weighted sum) for the compressed segment. Non-comp fp16
    segments (sink/window/decode/stream) are combined via flash-style mixing so the
    global softmax is exact. B=1 only (decode path)."""
    from .cache import hadamard
    from .triton_kernels import score_softmax_vq

    gc = layer._gpu_compressed
    if gc is None: return None
    B, n_q, _, hd = q_rope.shape
    if B != 1: return None
    nkv = layer._cache_ref._n_kv_heads
    n_comp = gc["n_compress"]
    gqa = n_q // nkv

    U_raw, V_raw, mean_c, mean_s, cos_d, sin_d = _compressed_score_parts(
        q_rope, gc, pos_q, n_q, nkv, hd, inv_freq)

    # VQ V data for comp segment: (nkv, n_comp, G) indices; run per-KV-head
    from .rope import unpack_nbits
    qd = gc["v_qdata"]
    gs = qd["group_size"]
    ng = hd // gs
    idx = gc.get("v_indices_u8")
    centroids = gc.get("v_centroids_f16")
    ch_max_all = gc.get("v_ch_f16")
    if idx is None or centroids is None or ch_max_all is None:
        idx = unpack_nbits(qd["packed"].to(q_rope.device), qd["bits"], qd["n_orig"])
        idx = idx[:nkv * n_comp * ng].view(nkv, n_comp, ng).to(torch.uint8).contiguous()
        centroids = qd["centroids"].to(q_rope.device).to(torch.float16).contiguous()
        ch_max_all = qd["ch_maxabs"].to(q_rope.device).view(nkv, hd).to(torch.float16)

    inv_sqrt_hd = 1.0 / math.sqrt(hd)
    out_comp_rot = torch.empty(n_q, hd, dtype=torch.float32, device=q_rope.device)
    m_comp = torch.empty(n_q, dtype=torch.float32, device=q_rope.device)
    l_comp = torch.empty(n_q, dtype=torch.float32, device=q_rope.device)
    for kv in range(nkv):
        q_slice = slice(kv * gqa, (kv + 1) * gqa)
        out_k, m_k, l_k = score_softmax_vq(
            U_raw[q_slice].contiguous(),
            V_raw[q_slice].contiguous(),
            mean_c[q_slice].contiguous(),
            mean_s[q_slice].contiguous(),
            cos_d, sin_d,
            idx[kv], centroids, ch_max_all[kv].contiguous(),
            inv_sqrt_hd,
        )
        out_comp_rot[q_slice] = out_k
        m_comp[q_slice] = m_k
        l_comp[q_slice] = l_k

    out_comp = hadamard(out_comp_rot.to(torch.float16)).float()  # (n_q, hd) standard space

    # Non-comp: gather scores + values for sink / win / decode / stream.
    def _score(q, k):
        return torch.einsum("bhsd,bhkd->bhsk",
                             q.float(),
                             k.repeat_interleave(gqa, dim=1).float()) * inv_sqrt_hd

    nc_parts_k = [gc["sink_k"], gc["win_k"]]
    nc_parts_v = [gc["sink_v"], gc["win_v"]]
    if "decode_k" in gc:
        nc_parts_k.append(gc["decode_k"]); nc_parts_v.append(gc["decode_v"])
    if gc.get("stream_n", 0) > 0:
        sk, sv = layer._dequant_stream(gc, B, nkv, hd)
        nc_parts_k.append(sk); nc_parts_v.append(sv)

    k_nc = torch.cat(nc_parts_k, dim=2)     # (1, nkv, N_nc, hd)
    v_nc = torch.cat(nc_parts_v, dim=2)
    s_nc = _score(q_rope, k_nc).squeeze(0).squeeze(1)  # (n_q, N_nc)
    # Decode: single query past all keys → causal mask is all-zero. Skip it.
    m_nc = s_nc.max(dim=-1).values
    p_nc = (s_nc - m_nc.unsqueeze(-1)).exp()
    l_nc = p_nc.sum(dim=-1)
    v_nc_exp = v_nc.repeat_interleave(gqa, dim=1).squeeze(0).float()  # (n_q, N_nc, hd)
    out_nc = torch.einsum("qk,qkd->qd", p_nc, v_nc_exp)                # (n_q, hd)

    # Flash-style mixing across comp + non-comp
    M = torch.maximum(m_comp, m_nc)
    a_c = (m_comp - M).exp()
    a_n = (m_nc - M).exp()
    Z = (a_c * l_comp + a_n * l_nc).unsqueeze(-1)
    out = (a_c.unsqueeze(-1) * out_comp + a_n.unsqueeze(-1) * out_nc) / Z
    return out.view(1, n_q, 1, hd).to(q_rope.dtype)


def fused_attention_compressed(q_rope, layer, pos_q, inv_freq, attn_mask):
    """Fused-K scoring on int4 coefficients + fused-V weighted sum (no fp16 K or V materialized
    for the compressed middle segment). Sink/window/decode/stream run through the fp16 path."""
    gc = layer._gpu_compressed
    if gc is None: return None
    B, n_q, _, hd = q_rope.shape
    nkv = layer._cache_ref._n_kv_heads
    n_comp = gc["n_compress"]
    gqa = n_q // nkv

    def _static_expanded(key):
        cache_key = f"{key}_gqa{gqa}_f32"
        cached = gc.get(cache_key)
        if cached is None:
            cached = gc[key].repeat_interleave(gqa, dim=1).float()
            gc[cache_key] = cached
        return cached

    def _score(q, k):
        return torch.einsum("bhsd,bhkd->bhsk",
                             q.float(),
                             k.repeat_interleave(gqa, dim=1).float()) / math.sqrt(hd)

    comp_scores = compressed_scores(q_rope, gc, pos_q, n_q, nkv, hd, inv_freq).float()
    q_fp32 = q_rope.float()
    inv_sqrt_hd = 1.0 / math.sqrt(hd)

    def _score_expanded(k_exp):
        return torch.einsum("bhsd,bhkd->bhsk", q_fp32, k_exp) * inv_sqrt_hd

    stream = None
    if gc.get("stream_n", 0) > 0:
        stream = layer._dequant_stream(gc, B, nkv, hd)

    parts = [_score_expanded(_static_expanded("sink_k")),
             comp_scores,
             _score_expanded(_static_expanded("win_k"))]
    if "decode_k" in gc:
        parts.append(_score(q_rope, gc["decode_k"]))
    if stream is not None:
        sk, _ = stream
        parts.append(_score(q_rope, sk))

    scores = torch.cat(parts, dim=-1)
    if attn_mask is not None: scores = scores + attn_mask
    weights = torch.softmax(scores, dim=-1, dtype=torch.float32)

    sink_n = gc["sink_k"].shape[2]
    win_n = gc["win_k"].shape[2]
    # Slice weights into segments in the same order as parts
    w_sink = weights[..., :sink_n]
    w_comp = weights[..., sink_n:sink_n + n_comp]
    w_win = weights[..., sink_n + n_comp:sink_n + n_comp + win_n]
    off = sink_n + n_comp + win_n

    out = _v_weighted_sum_compressed(w_comp, gc, B, nkv, n_comp, hd)
    # add contributions from fp16 segments
    def _add_fp16(w, v):
        v_exp = v.repeat_interleave(gqa, dim=1).float()
        return torch.einsum("bhsk,bhkd->bhsd", w, v_exp)

    def _add_expanded(w, v_exp):
        return torch.einsum("bhsk,bhkd->bhsd", w, v_exp)

    out = out + _add_expanded(w_sink, _static_expanded("sink_v"))
    out = out + _add_expanded(w_win, _static_expanded("win_v"))
    if "decode_v" in gc:
        dv = gc["decode_v"]
        out = out + _add_fp16(weights[..., off:off + dv.shape[2]], dv)
        off += dv.shape[2]
    if stream is not None:
        _, sv = stream
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
        past_key_values._fused_decode_update = True
        try:
            past_key_values.update(k, v, self.layer_idx,
                                    {"sin": sin, "cos": cos, "cache_position": cache_position})
        finally:
            past_key_values._fused_decode_update = False
        pos_q = int(cache_position[-1].item()) if cache_position is not None \
                else past_key_values.get_seq_length() - 1
        out = None
        if getattr(past_key_values, "_full_fused", False):
            past_key_values._full_fused_attempts = (
                getattr(past_key_values, "_full_fused_attempts", 0) + 1)
            try:
                out = fused_attention_compressed_full(q, layer, pos_q,
                                                       past_key_values._rope_inv_freq,
                                                       attention_mask)
                if out is not None:
                    past_key_values._full_fused_hits = (
                        getattr(past_key_values, "_full_fused_hits", 0) + 1)
            except Exception as exc:
                past_key_values._full_fused_errors = (
                    getattr(past_key_values, "_full_fused_errors", 0) + 1)
                past_key_values._last_full_fused_error = repr(exc)
                out = None
        if out is None:
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
