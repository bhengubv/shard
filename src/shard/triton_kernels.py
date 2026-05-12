"""Triton kernels for Shard's hot paths."""
import math
import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


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
# VQ weighted sum (V-path fusion)
# ---------------------------------------------------------------------------

if HAS_TRITON:
    @triton.jit
    def _vq_weighted_sum_kernel(
        W_ptr,
        Idx_ptr,
        Cb_ptr,
        Ch_ptr,
        Out_ptr,
        Q, N, D, G, gs,
        swq, swn,
        sig, sin_,
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

        g_idx = offs_d // gs
        g_pos = offs_d % gs

        ch = tl.load(Ch_ptr + offs_d, mask=d_mask, other=0.0).to(tl.float32)

        acc = tl.zeros((BQ, BD), dtype=tl.float32)
        for n_start in range(0, N, BN):
            offs_n = n_start + tl.arange(0, BN)
            n_mask = offs_n < N

            idx = tl.load(Idx_ptr + offs_n[:, None] * sin_ + g_idx[None, :] * sig,
                           mask=n_mask[:, None] & d_mask[None, :], other=0).to(tl.int64)

            v_rot = tl.load(Cb_ptr + idx * scc + g_pos[None, :] * scg,
                             mask=n_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            v_rot = v_rot * ch[None, :]

            w = tl.load(W_ptr + offs_q[:, None] * swq + offs_n[None, :] * swn,
                         mask=q_mask[:, None] & n_mask[None, :], other=0.0)

            acc += tl.dot(w, v_rot, input_precision="ieee")

        tl.store(Out_ptr + offs_q[:, None] * sou + offs_d[None, :] * sod,
                  acc, mask=q_mask[:, None] & d_mask[None, :])


def vq_weighted_sum(weights, indices, codebook, ch_max):
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
    Q, N = weights.shape
    N2, G = indices.shape
    C, gs = codebook.shape
    D = G * gs
    v_rot = codebook[indices.long()].reshape(N, D) * ch_max.to(torch.float32)
    return weights.to(torch.float32) @ v_rot.to(torch.float32)


def hadamard(x):
    d = x.shape[-1]
    h = 1
    while h < d:
        x = x.unflatten(-1, (-1, h * 2))
        a, b = x[..., :h], x[..., h:]
        x = torch.cat([a + b, a - b], dim=-1).flatten(-2)
        h *= 2
    return x / math.sqrt(d)


# Online-softmax + VQ-V flash-style primitive (not live-routed by default).
if HAS_TRITON:
    @triton.jit
    def _fused_attn_decode_kernel(
        Q_ptr, K_ptr, Idx_ptr, Cb_ptr, Ch_ptr,
        Out_ptr, Lse_ptr,
        H, N, hd: tl.constexpr, G, gs,
        scale,
        sqh, sqd,
        skn, skd,
        sig, sin_,
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

            k = tl.load(K_ptr + offs_n[:, None] * skn + offs_d_full[None, :] * skd,
                         mask=n_mask[:, None], other=0.0).to(tl.float16)

            s = tl.dot(q, tl.trans(k.to(tl.float32)), input_precision="ieee")
            s = tl.where(h_mask[:, None] & n_mask[None, :], s, -float("inf"))

            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]

            idx = tl.load(Idx_ptr + offs_n[:, None] * sin_ + g_idx[None, :] * sig,
                           mask=n_mask[:, None] & d_mask[None, :], other=0).to(tl.int64)
            v_rot = tl.load(Cb_ptr + idx * scc + g_pos[None, :] * scg,
                             mask=n_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            v_rot = v_rot * ch[None, :]

            acc += tl.dot(p, v_rot, input_precision="ieee")
            m_i = m_new

        tl.store(Out_ptr + offs_h[:, None] * sou + offs_d[None, :] * sod,
                  acc, mask=h_mask[:, None] & d_mask[None, :])
        if pid_d == 0:
            lse = m_i + tl.log(l_i)
            tl.store(Lse_ptr + offs_h, lse, mask=h_mask)


def fused_attn_decode(q, k, v_indices, v_codebook, v_ch, scale):
    """One-pass Q·Kᵀ softmax ·V (primitive; takes pre-reconstructed fp16 K)."""
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
    s = (q.float() @ k.float().T) * scale
    m = s.max(dim=-1, keepdim=True).values
    p = (s - m).exp()
    l = p.sum(dim=-1, keepdim=True)
    v_rot = v_codebook[v_indices.long()].reshape(N, hd).float() * v_ch.float()
    out = p @ v_rot
    lse = (m.squeeze(-1) + l.squeeze(-1).log())
    return out, lse


# Full-fused score-reduce + softmax + VQ V (primitive).
if HAS_TRITON:
    @triton.jit
    def _score_softmax_vq_kernel(
        U_ptr, V_ptr,
        Mc_ptr, Ms_ptr,
        Cos_ptr, Sin_ptr,
        Idx_ptr, Cb_ptr, Ch_ptr,
        Out_ptr, M_ptr, L_ptr,
        H, N, hd_half: tl.constexpr, hd: tl.constexpr, G, gs,
        inv_sqrt_hd,
        suh, sup, sun,
        svh, svp, svn,
        smch, smcp,
        smsh, smsp,
        scop, scon,
        ssip, ssin,
        sig, sin_,
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

        mc = tl.load(Mc_ptr + offs_h[:, None] * smch + offs_p[None, :] * smcp,
                      mask=h_mask[:, None], other=0.0).to(tl.float32)
        ms = tl.load(Ms_ptr + offs_h[:, None] * smsh + offs_p[None, :] * smsp,
                      mask=h_mask[:, None], other=0.0).to(tl.float32)

        for n_start in range(0, N, BN):
            offs_n = n_start + tl.arange(0, BN)
            n_mask = offs_n < N

            cos_t = tl.load(Cos_ptr + offs_p[:, None] * scop + offs_n[None, :] * scon,
                             mask=n_mask[None, :], other=0.0).to(tl.float32)
            sin_t = tl.load(Sin_ptr + offs_p[:, None] * ssip + offs_n[None, :] * ssin,
                             mask=n_mask[None, :], other=0.0).to(tl.float32)

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

            s += tl.sum(mc[:, :, None] * cos_t[None, :, :] + ms[:, :, None] * sin_t[None, :, :], axis=1)
            s = s * inv_sqrt_hd
            s = tl.where(h_mask[:, None] & n_mask[None, :], s, -float("inf"))

            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]

            idx = tl.load(Idx_ptr + offs_n[:, None] * sin_ + g_idx[None, :] * sig,
                           mask=n_mask[:, None] & d_mask[None, :], other=0).to(tl.int64)
            v_rot = tl.load(Cb_ptr + idx * scc + g_pos[None, :] * scg,
                             mask=n_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            v_rot = v_rot * ch[None, :]

            acc += tl.dot(p, v_rot, input_precision="ieee")
            m_i = m_new

        tl.store(Out_ptr + offs_h[:, None] * sou + offs_d[None, :] * sod,
                  acc, mask=h_mask[:, None] & d_mask[None, :])
        if pid_d == 0:
            tl.store(M_ptr + offs_h, m_i, mask=h_mask)
            tl.store(L_ptr + offs_h, l_i, mask=h_mask)


def score_softmax_vq(U, V, mean_c, mean_s, cos, sin, v_indices, v_codebook, v_ch, inv_sqrt_hd):
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
