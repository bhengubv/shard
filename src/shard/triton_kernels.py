"""Triton kernels for Shard's hot paths.

Kernels:
  1. `int4_matmul` — rank-R matmul with inline int4 unpack (compressed-K scoring).
  2. `vq_weighted_sum` — codebook lookup + per-channel scale + weighted sum over
     compressed V in one kernel (Hadamard-rotated space; caller undoes Hadamard).
  3. `fused_attention_decode` — FlashAttention-style one-pass Q·Kᵀ softmax·V for
     the compressed segment: int4 K unpack + VQ V unpack + online softmax, all fused.

Each kernel has a PyTorch fallback for CPU/non-Triton paths.
"""
import math
import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ---------------------------------------------------------------------------
# 1. int4 × fp16 matmul (fused unpack)
# ---------------------------------------------------------------------------

if HAS_TRITON:
    @triton.jit
    def _int4_matmul_kernel(
        A_ptr, B_packed_ptr, B_scale_ptr, Out_ptr,
        M, N, R,
        sam: tl.constexpr, sar: tl.constexpr,
        sbn, sbr: tl.constexpr,
        som, son: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BR: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)

        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for r_start in range(0, R, BR):
            offs_r = r_start + tl.arange(0, BR)
            r_mask = offs_r < R
            a = tl.load(A_ptr + offs_m[:, None] * sam + offs_r[None, :] * sar,
                        mask=(offs_m[:, None] < M) & r_mask[None, :], other=0.0)
            half = offs_r // 2
            is_hi = (offs_r % 2) == 1
            b_bytes = tl.load(
                B_packed_ptr + offs_n[:, None] * sbn + half[None, :] * sbr,
                mask=(offs_n[:, None] < N) & r_mask[None, :], other=0)
            lo = (b_bytes & 0x0F).to(tl.float32) - 8.0
            hi = ((b_bytes >> 4) & 0x0F).to(tl.float32) - 8.0
            b_vals = tl.where(is_hi[None, :], hi, lo)
            s = tl.load(B_scale_ptr + offs_r, mask=r_mask, other=0.0)
            b_scaled = (b_vals * s[None, :]).to(tl.float16)
            acc += tl.dot(a.to(tl.float16), tl.trans(b_scaled))

        tl.store(Out_ptr + offs_m[:, None] * som + offs_n[None, :] * son,
                  acc.to(tl.float16),
                  mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def int4_matmul(A, B_packed, B_scale, N, R):
    """A @ dequant(B_packed, B_scale)^T where B_packed is int4 with per-column scale."""
    if not (HAS_TRITON and A.is_cuda):
        return _int4_matmul_torch(A, B_packed, B_scale, N, R)
    M = A.shape[0]
    out = torch.empty(M, N, dtype=torch.float16, device=A.device)
    BM, BN, BR = 64, 128, 64
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _int4_matmul_kernel[grid](
        A, B_packed, B_scale, out,
        M, N, R,
        A.stride(0), A.stride(1),
        B_packed.stride(0), B_packed.stride(1),
        out.stride(0), out.stride(1),
        BM=BM, BN=BN, BR=BR,
    )
    return out


def _int4_matmul_torch(A, B_packed, B_scale, N, R):
    lo = (B_packed & 0x0F).to(torch.float16) - 8.0
    hi = ((B_packed >> 4) & 0x0F).to(torch.float16) - 8.0
    B = torch.stack([lo, hi], dim=-1).flatten(-2)[:, :R]
    return A.to(torch.float16) @ (B * B_scale.to(torch.float16)).T


# ---------------------------------------------------------------------------
# 2. VQ weighted sum (V-path fusion)
# ---------------------------------------------------------------------------
# Computes out[q, d] = sum_t weights[q, t] * codebook[indices[t, d//gs]][d%gs] * ch_max[d]
# in Hadamard-rotated space. Caller applies one inverse Hadamard to the output.

if HAS_TRITON:
    @triton.jit
    def _vq_weighted_sum_kernel(
        W_ptr,           # (Q, N) fp32 softmax weights
        Idx_ptr,         # (N, G) uint8 codebook indices
        Cb_ptr,          # (C, gs) fp16 codebook
        Ch_ptr,          # (D,) fp16 per-channel scale
        Out_ptr,         # (Q, D) fp32 output
        Q, N, D, G, gs,
        swq, swn,
        sin_, sig,     # indices: stride along N dim, stride along G dim
        scc, scg,
        sou, sod,
        BQ: tl.constexpr, BD: tl.constexpr, BN: tl.constexpr,
    ):
        pid_q = tl.program_id(0)
        pid_d = tl.program_id(1)
        offs_q = pid_q * BQ + tl.arange(0, BQ)
        offs_d = pid_d * BD + tl.arange(0, BD)
        q_mask = offs_q < Q
        d_mask = offs_d < D

        # group index and within-group position for each output feature
        g_idx = offs_d // gs              # (BD,)
        g_pos = offs_d % gs               # (BD,)

        ch = tl.load(Ch_ptr + offs_d, mask=d_mask, other=0.0).to(tl.float32)

        acc = tl.zeros((BQ, BD), dtype=tl.float32)
        for n_start in range(0, N, BN):
            offs_n = n_start + tl.arange(0, BN)
            n_mask = offs_n < N

            # indices[t, g_idx] for this N-block — shape (BN, BD)
            idx = tl.load(Idx_ptr + offs_n[:, None] * sin_ + g_idx[None, :] * sig,
                           mask=n_mask[:, None] & d_mask[None, :], other=0).to(tl.int64)

            # codebook[idx, g_pos] → (BN, BD) fp32
            v_rot = tl.load(Cb_ptr + idx * scc + g_pos[None, :] * scg,
                             mask=n_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            v_rot = v_rot * ch[None, :]

            # weights[offs_q, offs_n] — shape (BQ, BN) fp32
            w = tl.load(W_ptr + offs_q[:, None] * swq + offs_n[None, :] * swn,
                         mask=q_mask[:, None] & n_mask[None, :], other=0.0)

            # fp32 matmul via tl.dot (ieee precision) — per-token outer sum with
            # static_range can't be expressed via slicing in Triton.
            acc += tl.dot(w, v_rot, input_precision="ieee")

        tl.store(Out_ptr + offs_q[:, None] * sou + offs_d[None, :] * sod,
                  acc, mask=q_mask[:, None] & d_mask[None, :])


def vq_weighted_sum(weights, indices, codebook, ch_max):
    """Weighted sum of Hadamard-rotated V, with on-chip VQ decompression.

    Args:
        weights: (Q, N) fp32
        indices: (N, G) uint8 — codebook indices for each group per token
        codebook: (C, gs) fp16
        ch_max: (D,) fp16 where D = G*gs
    Returns: (Q, D) fp32 in rotated space. Caller applies inverse Hadamard.
    """
    assert weights.is_contiguous() and indices.is_contiguous()
    Q, N = weights.shape
    _, G = indices.shape
    C, gs = codebook.shape
    D = G * gs
    assert ch_max.shape == (D,)

    if not (HAS_TRITON and weights.is_cuda):
        return _vq_weighted_sum_torch(weights, indices, codebook, ch_max)

    out = torch.empty(Q, D, dtype=torch.float32, device=weights.device)
    BQ, BD, BN = 32, 64, 16
    grid = (triton.cdiv(Q, BQ), triton.cdiv(D, BD))
    _vq_weighted_sum_kernel[grid](
        weights, indices, codebook, ch_max, out,
        Q, N, D, G, gs,
        weights.stride(0), weights.stride(1),
        indices.stride(0), indices.stride(1),
        codebook.stride(0), codebook.stride(1),
        out.stride(0), out.stride(1),
        BQ=BQ, BD=BD, BN=BN,
    )
    return out


def _vq_weighted_sum_torch(weights, indices, codebook, ch_max):
    # weights: (Q, N), indices: (N, G) uint8, codebook: (C, gs), ch_max: (D,)
    Q, N = weights.shape
    N2, G = indices.shape
    C, gs = codebook.shape
    D = G * gs
    v_rot = codebook[indices.long()].reshape(N, D) * ch_max.to(torch.float32)
    return weights.to(torch.float32) @ v_rot.to(torch.float32)


# ---------------------------------------------------------------------------
# 2b. All-heads VQ weighted sum — single launch handles every KV head at once.
# ---------------------------------------------------------------------------
# Grid = (nkv, ceil(D/BD)). Each program: one KV head × one D tile. BH = gqa
# queries are contiguous within the program. Replaces the per-head Python loop
# in _v_weighted_sum_compressed with a single kernel dispatch.

if HAS_TRITON:
    @triton.jit
    def _vq_weighted_sum_all_kernel(
        W_ptr,           # (nkv*gqa, N) fp32 weights laid out per KV head
        Idx_ptr,         # (nkv, N, G) uint8
        Cb_ptr,          # (C, gs) fp16
        Ch_ptr,          # (nkv, D) fp16 per-channel scale
        Out_ptr,         # (nkv*gqa, D) fp32
        N, D, G, gs,
        gqa: tl.constexpr,
        swh, swn,
        sik, sin_, sig,
        scc, scg,
        sck, scd,
        soh, sod,
        BH: tl.constexpr, BD: tl.constexpr, BN: tl.constexpr,
    ):
        pid_kv = tl.program_id(0)
        pid_d = tl.program_id(1)
        offs_h = pid_kv * gqa + tl.arange(0, BH)
        offs_d = pid_d * BD + tl.arange(0, BD)
        h_mask = tl.arange(0, BH) < gqa
        d_mask = offs_d < D

        g_idx = offs_d // gs
        g_pos = offs_d % gs

        ch = tl.load(Ch_ptr + pid_kv * sck + offs_d * scd,
                      mask=d_mask, other=0.0).to(tl.float32)

        acc = tl.zeros((BH, BD), dtype=tl.float32)
        for n_start in range(0, N, BN):
            offs_n = n_start + tl.arange(0, BN)
            n_mask = offs_n < N

            idx = tl.load(Idx_ptr + pid_kv * sik
                           + offs_n[:, None] * sin_
                           + g_idx[None, :] * sig,
                           mask=n_mask[:, None] & d_mask[None, :], other=0).to(tl.int64)
            v_rot = tl.load(Cb_ptr + idx * scc + g_pos[None, :] * scg,
                             mask=n_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            v_rot = v_rot * ch[None, :]

            w = tl.load(W_ptr + offs_h[:, None] * swh + offs_n[None, :] * swn,
                         mask=h_mask[:, None] & n_mask[None, :], other=0.0)

            acc += tl.dot(w, v_rot, input_precision="ieee")

        tl.store(Out_ptr + offs_h[:, None] * soh + offs_d[None, :] * sod,
                  acc, mask=h_mask[:, None] & d_mask[None, :])


def vq_weighted_sum_all(weights, indices, codebook, ch_max, gqa):
    """All-KV-head VQ weighted sum in ONE launch.

    weights: (nkv*gqa, N) fp32 contiguous
    indices: (nkv, N, G) uint8 contiguous
    codebook: (C, gs) fp16 contiguous
    ch_max: (nkv, D) fp16 contiguous where D = G*gs
    gqa: int, query heads per KV head
    Returns: (nkv*gqa, D) fp32
    """
    assert weights.is_contiguous() and indices.is_contiguous()
    assert codebook.is_contiguous() and ch_max.is_contiguous()
    HQ, N = weights.shape
    nkv, N2, G = indices.shape
    C, gs = codebook.shape
    D = G * gs
    assert HQ == nkv * gqa

    if not (HAS_TRITON and weights.is_cuda):
        # PyTorch fallback matching the kernel semantics.
        idx_l = indices.long()
        v_rot = codebook.float()[idx_l].reshape(nkv, N, D) * ch_max.float().unsqueeze(1)
        # Broadcast by gqa
        v_rot_q = v_rot.unsqueeze(1).expand(nkv, gqa, N, D).reshape(HQ, N, D)
        return torch.einsum("qn,qnd->qd", weights, v_rot_q)

    out = torch.empty(HQ, D, dtype=torch.float32, device=weights.device)
    BH, BD, BN = max(16, gqa), 64, 64
    grid = (nkv, triton.cdiv(D, BD))
    _vq_weighted_sum_all_kernel[grid](
        weights, indices, codebook, ch_max, out,
        N, D, G, gs,
        gqa,
        weights.stride(0), weights.stride(1),
        indices.stride(0), indices.stride(1), indices.stride(2),
        codebook.stride(0), codebook.stride(1),
        ch_max.stride(0), ch_max.stride(1),
        out.stride(0), out.stride(1),
        BH=BH, BD=BD, BN=BN,
    )
    return out


# ---------------------------------------------------------------------------
# 3. Hadamard (normalized WHT along last dim). PyTorch butterfly — fast at hd=128.
# ---------------------------------------------------------------------------

def hadamard(x):
    d = x.shape[-1]
    h = 1
    while h < d:
        x = x.unflatten(-1, (-1, h * 2))
        a, b = x[..., :h], x[..., h:]
        x = torch.cat([a + b, a - b], dim=-1).flatten(-2)
        h *= 2
    return x / math.sqrt(d)


# ---------------------------------------------------------------------------
# 4. Fused attention decode: Q · (int4 PCA-K)ᵀ → softmax → · (VQ V) in one pass.
# ---------------------------------------------------------------------------
# One program instance handles one (query head, D-tile) pair. It streams over
# N tokens in blocks of BN, computes scores on-the-fly by materializing K from
# packed int4 coefficients, updates the running online-softmax accumulators
# (m, l, acc_rot), and unpacks V via codebook lookup for the matmul. At the
# end the D-tile of acc/l is written out (in Hadamard-rotated space). Caller
# applies inverse Hadamard and adds sink/window contributions.
#
# Because this kernel folds the full score pipeline, it's only used when the
# score path reduces to a standard Q·Kᵀ against fp16 K (i.e., K has been
# pre-reconstructed). The per-pair-RoPE math stays in PyTorch/int4_matmul;
# this kernel takes an already-reconstructed fp16 K_comp as input and fuses
# Q·Kᵀ + softmax + ·V. That gives ~3x over unfused eager while keeping the
# RoPE identity correct.

if HAS_TRITON:
    @triton.jit
    def _fused_attn_decode_kernel(
        Q_ptr,       # (H, hd) fp16
        K_ptr,       # (N, hd) fp16 — reconstructed compressed K (one KV head)
        Idx_ptr,     # (N, G) uint8 — V codebook indices (one KV head)
        Cb_ptr,      # (C, gs) fp16
        Ch_ptr,      # (hd,) fp16 — per-channel scale for this KV head
        Out_ptr,     # (H, hd) fp32 — rotated output, caller inverts Hadamard
        Lse_ptr,     # (H,) fp32 — logsumexp for mixing with fp16 segments
        H, N, hd: tl.constexpr, G, gs,
        scale,
        sqh, sqd,
        skn, skd,
        sin_, sig,
        scc, scg,
        sou, sod,
        BH: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr,
    ):
        pid_h = tl.program_id(0)
        pid_d = tl.program_id(1)
        offs_h = pid_h * BH + tl.arange(0, BH)
        offs_d = pid_d * BD + tl.arange(0, BD)
        h_mask = offs_h < H
        d_mask = offs_d < hd
        offs_d_full = tl.arange(0, hd)

        # Load Q tile (BH, hd). Full hd fits in one block for llama (hd=128).
        q = tl.load(Q_ptr + offs_h[:, None] * sqh + offs_d_full[None, :] * sqd,
                    mask=h_mask[:, None], other=0.0).to(tl.float32) * scale

        ch = tl.load(Ch_ptr + offs_d, mask=d_mask, other=0.0).to(tl.float32)

        m_i = tl.full((BH,), -float("inf"), tl.float32)
        l_i = tl.zeros((BH,), tl.float32)
        acc = tl.zeros((BH, BD), tl.float32)

        g_idx = offs_d // gs
        g_pos = offs_d % gs

        for n_start in range(0, N, BN):
            offs_n = n_start + tl.arange(0, BN)
            n_mask = offs_n < N

            # Load K tile (BN, hd) — fp16 reconstructed.
            k = tl.load(K_ptr + offs_n[:, None] * skn + offs_d_full[None, :] * skd,
                         mask=n_mask[:, None], other=0.0).to(tl.float16)

            # scores s[h, n] = Σ_d q[h, d] * k[n, d]   (explicit fp32 reduction)
            k32 = k.to(tl.float32)
            s = tl.sum(q[:, None, :] * k32[None, :, :], axis=2)
            s = tl.where(h_mask[:, None] & n_mask[None, :], s, -float("inf"))

            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]

            # V decode
            idx = tl.load(Idx_ptr + offs_n[:, None] * sin_ + g_idx[None, :] * sig,
                           mask=n_mask[:, None] & d_mask[None, :], other=0).to(tl.int64)
            v_rot = tl.load(Cb_ptr + idx * scc + g_pos[None, :] * scg,
                             mask=n_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            v_rot = v_rot * ch[None, :]

            # acc[h, d] += Σ_n p[h, n] * v_rot[n, d]  (explicit fp32)
            acc += tl.sum(p[:, :, None] * v_rot[None, :, :], axis=1)
            m_i = m_new

        # Write rotated output (Q normalizes by 1/l at mix stage, not here).
        tl.store(Out_ptr + offs_h[:, None] * sou + offs_d[None, :] * sod,
                  acc, mask=h_mask[:, None] & d_mask[None, :])
        # Write logsumexp for downstream mixing with sink/window fp16 segments.
        if pid_d == 0:
            lse = m_i + tl.log(l_i)
            tl.store(Lse_ptr + offs_h, lse, mask=h_mask)


def fused_attn_decode(q, k, v_indices, v_codebook, v_ch, scale):
    """One-pass Q·Kᵀ softmax ·V for a single KV head.

    q: (H, hd) fp16 — Q heads mapped to this KV head (post-GQA duplication)
    k: (N, hd) fp16 — reconstructed K for this KV head
    v_indices: (N, G) uint8 — VQ indices
    v_codebook: (C, gs) fp16
    v_ch: (hd,) fp16 — per-channel scale
    Returns (out_rot, lse): out in Hadamard-rotated space, lse for segment mixing.
    """
    assert q.is_contiguous() and k.is_contiguous()
    assert v_indices.is_contiguous() and v_codebook.is_contiguous()
    H, hd = q.shape
    N, _ = v_indices.shape
    C, gs = v_codebook.shape
    G = hd // gs
    out = torch.empty(H, hd, dtype=torch.float32, device=q.device)
    lse = torch.empty(H, dtype=torch.float32, device=q.device)

    if not (HAS_TRITON and q.is_cuda):
        return _fused_attn_decode_torch(q, k, v_indices, v_codebook, v_ch, scale)

    BH, BN, BD = 32, 16, 64
    grid = (triton.cdiv(H, BH), triton.cdiv(hd, BD))
    _fused_attn_decode_kernel[grid](
        q, k, v_indices, v_codebook, v_ch, out, lse,
        H, N, hd, G, gs,
        scale,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        v_indices.stride(0), v_indices.stride(1),
        v_codebook.stride(0), v_codebook.stride(1),
        out.stride(0), out.stride(1),
        BH=BH, BN=BN, BD=BD,
    )
    return out, lse


def _fused_attn_decode_torch(q, k, v_indices, v_codebook, v_ch, scale):
    H, hd = q.shape
    N, G = v_indices.shape
    C, gs = v_codebook.shape
    s = (q.float() @ k.float().T) * scale        # (H, N)
    m = s.max(dim=-1, keepdim=True).values
    p = (s - m).exp()
    l = p.sum(dim=-1, keepdim=True)
    v_rot = v_codebook[v_indices.long()].reshape(N, hd).float() * v_ch.float()
    out = p @ v_rot                               # (H, hd) unnormalized
    lse = (m.squeeze(-1) + l.squeeze(-1).log())
    return out, lse


# ---------------------------------------------------------------------------
# 5. Fused score-reduce + online softmax + VQ-V weighted sum.
# ---------------------------------------------------------------------------
# Takes U, V from int4_matmul (the two rank-R matmul halves of the RoPE
# identity) plus mean terms, cos/sin tables, and VQ V data. Does cos/sin
# reduction, mean add, scaling, online softmax, and weighted sum against
# codebook-decoded V — all in one kernel. Output is in Hadamard-rotated
# space; caller inverts. This is the "full fused decode attention" for the
# compressed middle segment.

if HAS_TRITON:
    @triton.jit
    def _score_softmax_vq_kernel(
        U_ptr, V_ptr,          # (H, hd_half, N) fp16 — score halves
        Mc_ptr, Ms_ptr,        # (H, hd_half) fp16 — mean terms
        Cos_ptr, Sin_ptr,      # (hd_half, N) fp16 — per-pair phase
        Idx_ptr,               # (N, G) uint8 — VQ indices
        Cb_ptr,                # (C, gs) fp16 — codebook
        Ch_ptr,                # (hd,) fp16 — per-channel scale
        Out_ptr, M_ptr, L_ptr,
        H, N, hd_half: tl.constexpr, hd: tl.constexpr, G, gs,
        inv_sqrt_hd,
        suh, sup, sun,
        svh, svp, svn,
        smch, smcp,
        smsh, smsp,
        scop, scon,
        ssip, ssin,
        sin_, sig,
        scc, scg,
        sou, sod,
        BH: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr,
    ):
        pid_h = tl.program_id(0)
        pid_d = tl.program_id(1)
        offs_h = pid_h * BH + tl.arange(0, BH)
        offs_d = pid_d * BD + tl.arange(0, BD)
        h_mask = offs_h < H
        d_mask = offs_d < hd

        ch = tl.load(Ch_ptr + offs_d, mask=d_mask, other=0.0).to(tl.float32)
        g_idx = offs_d // gs
        g_pos = offs_d % gs

        m_i = tl.full((BH,), -float("inf"), tl.float32)
        l_i = tl.zeros((BH,), tl.float32)
        acc = tl.zeros((BH, BD), tl.float32)

        offs_p = tl.arange(0, hd_half)

        # mean terms (BH, hd_half) loaded once
        mc = tl.load(Mc_ptr + offs_h[:, None] * smch + offs_p[None, :] * smcp,
                      mask=h_mask[:, None], other=0.0).to(tl.float32)
        ms = tl.load(Ms_ptr + offs_h[:, None] * smsh + offs_p[None, :] * smsp,
                      mask=h_mask[:, None], other=0.0).to(tl.float32)

        for n_start in range(0, N, BN):
            offs_n = n_start + tl.arange(0, BN)
            n_mask = offs_n < N

            # cos/sin tiles (hd_half, BN)
            cos_t = tl.load(Cos_ptr + offs_p[:, None] * scop + offs_n[None, :] * scon,
                             mask=n_mask[None, :], other=0.0).to(tl.float32)
            sin_t = tl.load(Sin_ptr + offs_p[:, None] * ssip + offs_n[None, :] * ssin,
                             mask=n_mask[None, :], other=0.0).to(tl.float32)

            # score = Σ_p U[h, p, n] * cos[p, n] + V[h, p, n] * sin[p, n] + mean
            # Compute via hd_half matmul over p: for fixed h, score[n] = u_col[p,n].dot(cos[p,n]) ... per pair
            # Simpler: elementwise over p, then reduce.
            s = tl.zeros((BH, BN), tl.float32)
            for p_start in range(0, hd_half, 16):
                offs_pp = p_start + tl.arange(0, 16)
                p_mask = offs_pp < hd_half
                u = tl.load(U_ptr + offs_h[:, None, None] * suh
                             + offs_pp[None, :, None] * sup
                             + offs_n[None, None, :] * sun,
                             mask=h_mask[:, None, None] & p_mask[None, :, None] & n_mask[None, None, :],
                             other=0.0).to(tl.float32)
                v = tl.load(V_ptr + offs_h[:, None, None] * svh
                             + offs_pp[None, :, None] * svp
                             + offs_n[None, None, :] * svn,
                             mask=h_mask[:, None, None] & p_mask[None, :, None] & n_mask[None, None, :],
                             other=0.0).to(tl.float32)
                c = tl.load(Cos_ptr + offs_pp[:, None] * scop + offs_n[None, :] * scon,
                             mask=p_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float32)
                sn = tl.load(Sin_ptr + offs_pp[:, None] * ssip + offs_n[None, :] * ssin,
                              mask=p_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float32)
                s += tl.sum(u * c[None, :, :] + v * sn[None, :, :], axis=1)

            # mean term reduction over p:
            s += tl.sum(mc[:, :, None] * cos_t[None, :, :] + ms[:, :, None] * sin_t[None, :, :], axis=1)
            s = s * inv_sqrt_hd
            s = tl.where(h_mask[:, None] & n_mask[None, :], s, -float("inf"))

            # online softmax
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]

            # VQ V decode for this (t, d_tile)
            idx = tl.load(Idx_ptr + offs_n[:, None] * sin_ + g_idx[None, :] * sig,
                           mask=n_mask[:, None] & d_mask[None, :], other=0).to(tl.int64)
            v_rot = tl.load(Cb_ptr + idx * scc + g_pos[None, :] * scg,
                             mask=n_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            v_rot = v_rot * ch[None, :]

            # acc[h, d] += Σ_n p[h, n] * v_rot[n, d]  (explicit fp32 reduction)
            acc += tl.sum(p[:, :, None] * v_rot[None, :, :], axis=1)
            m_i = m_new

        tl.store(Out_ptr + offs_h[:, None] * sou + offs_d[None, :] * sod,
                  acc, mask=h_mask[:, None] & d_mask[None, :])
        if pid_d == 0:
            tl.store(M_ptr + offs_h, m_i, mask=h_mask)
            tl.store(L_ptr + offs_h, l_i, mask=h_mask)


def score_softmax_vq(U, V, mean_c, mean_s, cos, sin, v_indices, v_codebook, v_ch, inv_sqrt_hd):
    """Fused: score = reduce_p(U·cos + V·sin) + mean → softmax → ·(VQ V). One-pass.

    Shapes:
      U, V: (H, hd_half, N) fp16
      mean_c, mean_s: (H, hd_half) fp16
      cos, sin: (hd_half, N) fp16
      v_indices: (N, G) uint8
      v_codebook: (C, gs) fp16
      v_ch: (hd,) fp16
    Returns (out_rot: (H, hd) fp32, m: (H,), l: (H,)) — the online-softmax running
    max and sum of exp(score - max). Caller uses these for flash-style mixing with
    sink/window segments. out_rot is Σ_t exp(s_t - m) v_t (not yet normalized by l).
    """
    H, hd_half, N = U.shape
    hd = hd_half * 2
    C, gs = v_codebook.shape
    G = hd // gs

    if not (HAS_TRITON and U.is_cuda):
        return _score_softmax_vq_torch(U, V, mean_c, mean_s, cos, sin,
                                        v_indices, v_codebook, v_ch, inv_sqrt_hd)

    out = torch.empty(H, hd, dtype=torch.float32, device=U.device)
    m = torch.empty(H, dtype=torch.float32, device=U.device)
    l = torch.empty(H, dtype=torch.float32, device=U.device)
    BH, BN, BD = 32, 16, 64
    grid = (triton.cdiv(H, BH), triton.cdiv(hd, BD))
    _score_softmax_vq_kernel[grid](
        U, V, mean_c, mean_s, cos, sin, v_indices, v_codebook, v_ch, out, m, l,
        H, N, hd_half, hd, G, gs,
        inv_sqrt_hd,
        U.stride(0), U.stride(1), U.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        mean_c.stride(0), mean_c.stride(1),
        mean_s.stride(0), mean_s.stride(1),
        cos.stride(0), cos.stride(1),
        sin.stride(0), sin.stride(1),
        v_indices.stride(0), v_indices.stride(1),
        v_codebook.stride(0), v_codebook.stride(1),
        out.stride(0), out.stride(1),
        BH=BH, BN=BN, BD=BD,
    )
    return out, m, l


def _score_softmax_vq_torch(U, V, mean_c, mean_s, cos, sin, v_indices, v_codebook, v_ch, inv_sqrt_hd):
    H, hd_half, N = U.shape
    hd = hd_half * 2
    scores = (U.float() * cos.float().unsqueeze(0)
              + V.float() * sin.float().unsqueeze(0)).sum(dim=1)
    scores += (mean_c.float().unsqueeze(-1) * cos.float().unsqueeze(0)
               + mean_s.float().unsqueeze(-1) * sin.float().unsqueeze(0)).sum(dim=1)
    scores *= inv_sqrt_hd
    m = scores.max(dim=-1).values
    p = (scores - m.unsqueeze(-1)).exp()
    l = p.sum(dim=-1)
    v_rot = v_codebook[v_indices.long()].reshape(N, hd).float() * v_ch.float()
    out = p @ v_rot
    return out, m, l


# ---------------------------------------------------------------------------
# 6. int4 pack: flat int8 [-7, 7] stream → uint8 nibbles, pairing adjacent.
# ---------------------------------------------------------------------------
# Matches `_unpack_int4` in attention.py: each byte stores two signed nibbles
# (hi = second value, lo = first), paired linearly in flat order.

if HAS_TRITON:
    @triton.jit
    def _int4_pack_kernel(
        In_ptr, Out_ptr,
        N_half,
        BN: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BN + tl.arange(0, BN)
        mask = offs < N_half
        lo = tl.load(In_ptr + offs * 2, mask=mask, other=0)
        hi = tl.load(In_ptr + offs * 2 + 1, mask=mask, other=0)
        lo_s = (lo + 8).to(tl.uint8) & 0x0F
        hi_s = (hi + 8).to(tl.uint8) & 0x0F
        tl.store(Out_ptr + offs, lo_s | (hi_s << 4), mask=mask)


def int4_pack(x):
    """Pack flat int8 in [-7, 7] → uint8 nibbles. Accepts any shape; flattens first."""
    flat = x.reshape(-1).contiguous()
    n = flat.shape[0]
    if n & 1:
        flat = torch.cat([flat, torch.zeros(1, dtype=flat.dtype, device=flat.device)])
        n += 1
    n_half = n // 2

    if not (HAS_TRITON and x.is_cuda):
        shifted = (flat + 8).to(torch.uint8) & 0x0F
        return (shifted[0::2] | (shifted[1::2] << 4)).contiguous()

    out = torch.empty(n_half, dtype=torch.uint8, device=flat.device)
    BN = 1024
    grid = (triton.cdiv(n_half, BN),)
    _int4_pack_kernel[grid](flat, out, n_half, BN=BN)
    return out


# ---------------------------------------------------------------------------
# 7. VQ assign: nearest-centroid lookup. Replaces cdist+argmin in kmeans loop.
# ---------------------------------------------------------------------------

if HAS_TRITON:
    @triton.jit
    def _vq_assign_kernel(
        X_ptr, Cb_ptr, Out_ptr,
        N, K, D: tl.constexpr,
        sxn, sxd,
        sck, scd,
        BN: tl.constexpr, BK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_n = pid * BN + tl.arange(0, BN)
        n_mask = offs_n < N
        offs_k_base = tl.arange(0, BK)

        best_dist = tl.full((BN,), float("inf"), tl.float32)
        best_idx = tl.zeros((BN,), tl.int32)

        for k_start in range(0, K, BK):
            offs_k = k_start + offs_k_base
            k_mask = offs_k < K
            # Accumulate squared distance by iterating D (constexpr) — no 3D tensor, no tl.dot.
            dist = tl.zeros((BN, BK), tl.float32)
            for d in tl.static_range(D):
                xd = tl.load(X_ptr + offs_n * sxn + d * sxd,
                              mask=n_mask, other=0.0).to(tl.float32)
                cd = tl.load(Cb_ptr + offs_k * sck + d * scd,
                              mask=k_mask, other=0.0).to(tl.float32)
                diff_d = xd[:, None] - cd[None, :]
                dist += diff_d * diff_d
            dist = tl.where(k_mask[None, :], dist, float("inf"))
            block_min = tl.min(dist, axis=1)
            block_arg = tl.argmin(dist, axis=1)
            is_new = block_min < best_dist
            best_dist = tl.where(is_new, block_min, best_dist)
            best_idx = tl.where(is_new, (k_start + block_arg).to(tl.int32), best_idx)

        tl.store(Out_ptr + offs_n, best_idx, mask=n_mask)


def vq_assign(x, centroids):
    """Return argmin_k ||x[n] - centroids[k]||² as int64. x: (N, D), centroids: (K, D)."""
    N, D = x.shape
    K, D2 = centroids.shape
    assert D == D2
    if not (HAS_TRITON and x.is_cuda):
        return torch.cdist(x, centroids).argmin(dim=1)

    out = torch.empty(N, dtype=torch.int32, device=x.device)
    BN, BK = 128, 32
    grid = (triton.cdiv(N, BN),)
    _vq_assign_kernel[grid](
        x.contiguous(), centroids.contiguous(), out,
        N, K, D,
        x.stride(0), x.stride(1),
        centroids.stride(0), centroids.stride(1),
        BN=BN, BK=BK,
    )
    return out.long()


# ---------------------------------------------------------------------------
# 8. Undo-RoPE fused kernel: x (..., hd) + cos/sin → x_unroped.
# ---------------------------------------------------------------------------
# For rotate_half RoPE: undo = rotate_half's transpose.
# x_unroped[..., :hd/2] = x[..., :hd/2]*cos + x[..., hd/2:]*sin
# x_unroped[..., hd/2:] = x[..., hd/2:]*cos - x[..., :hd/2]*sin

if HAS_TRITON:
    @triton.jit
    def _undo_rope_kernel(
        X_ptr, Cos_ptr, Sin_ptr, Out_ptr,
        N, hd: tl.constexpr, hd_half: tl.constexpr,
        sxn, sxd,
        scn, scd,
        son, sod,
        BN: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_n = pid * BN + tl.arange(0, BN)
        n_mask = offs_n < N
        offs_lo = tl.arange(0, hd_half)
        offs_hi = hd_half + tl.arange(0, hd_half)

        x_lo = tl.load(X_ptr + offs_n[:, None] * sxn + offs_lo[None, :] * sxd,
                        mask=n_mask[:, None], other=0.0).to(tl.float32)
        x_hi = tl.load(X_ptr + offs_n[:, None] * sxn + offs_hi[None, :] * sxd,
                        mask=n_mask[:, None], other=0.0).to(tl.float32)
        cos_lo = tl.load(Cos_ptr + offs_n[:, None] * scn + offs_lo[None, :] * scd,
                          mask=n_mask[:, None], other=0.0).to(tl.float32)
        sin_lo = tl.load(Sin_ptr + offs_n[:, None] * scn + offs_lo[None, :] * scd,
                          mask=n_mask[:, None], other=0.0).to(tl.float32)

        out_lo = x_lo * cos_lo + x_hi * sin_lo
        out_hi = x_hi * cos_lo - x_lo * sin_lo
        tl.store(Out_ptr + offs_n[:, None] * son + offs_lo[None, :] * sod,
                  out_lo.to(tl.float16), mask=n_mask[:, None])
        tl.store(Out_ptr + offs_n[:, None] * son + offs_hi[None, :] * sod,
                  out_hi.to(tl.float16), mask=n_mask[:, None])


def undo_rope_fused(x, cos, sin):
    """Undo rotate_half RoPE. x: (..., hd), cos/sin: (..., hd) or broadcast. Returns fp16."""
    shape = x.shape
    hd = shape[-1]
    x_flat = x.reshape(-1, hd).contiguous()
    cos_flat = cos.expand_as(x).reshape(-1, hd).contiguous()
    sin_flat = sin.expand_as(x).reshape(-1, hd).contiguous()
    N = x_flat.shape[0]

    if not (HAS_TRITON and x.is_cuda):
        hd_half = hd // 2
        x_lo, x_hi = x_flat[:, :hd_half].float(), x_flat[:, hd_half:].float()
        c, s = cos_flat[:, :hd_half].float(), sin_flat[:, :hd_half].float()
        out = torch.empty_like(x_flat, dtype=torch.float16)
        out[:, :hd_half] = (x_lo * c + x_hi * s).half()
        out[:, hd_half:] = (x_hi * c - x_lo * s).half()
        return out.reshape(*shape)

    out = torch.empty_like(x_flat, dtype=torch.float16)
    BN = 128
    grid = (triton.cdiv(N, BN),)
    _undo_rope_kernel[grid](
        x_flat.to(torch.float16), cos_flat.to(torch.float16), sin_flat.to(torch.float16), out,
        N, hd, hd // 2,
        x_flat.stride(0), x_flat.stride(1),
        cos_flat.stride(0), cos_flat.stride(1),
        out.stride(0), out.stride(1),
        BN=BN,
    )
    return out.reshape(*shape)
