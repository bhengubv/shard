"""Modal-hosted kernel verification. Runs each Triton kernel on B200 GPU with
random inputs, compares to the PyTorch reference, and reports max|diff|. Also
does an end-to-end attention equivalence check (fused vs eager) so we catch
wiring-level bugs that unit tests miss."""
import modal

app = modal.App("shard-kernel-tests")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "triton", "transformers")
    .add_local_dir("src/shard", remote_path="/root/shard")
)


@app.function(image=image, gpu="B200", timeout=900)
def run():
    import sys
    sys.path.insert(0, "/root")
    import torch
    import math
    from shard.triton_kernels import (
        int4_matmul, _int4_matmul_torch,
        vq_weighted_sum, _vq_weighted_sum_torch,
        vq_weighted_sum_all,
        fused_attn_decode, _fused_attn_decode_torch,
        score_softmax_vq, _score_softmax_vq_torch,
        int4_pack, vq_assign, undo_rope_fused, hadamard,
        HAS_TRITON,
    )

    torch.manual_seed(0)
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print(f"HAS_TRITON={HAS_TRITON}  tf32_disabled")
    print(f"{'='*70}")

    def diff(a, b, name):
        d = (a.float() - b.float()).abs()
        mx, mn = d.max().item(), d.mean().item()
        ok = mx < 1e-2
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] {name:<35} max={mx:.6e} mean={mn:.6e}")
        return ok

    all_ok = True

    # 1. int4_matmul
    print("\n1. int4_matmul")
    M, N, R = 128, 512, 192
    A = torch.randn(M, R, device=dev, dtype=torch.float16) * 0.1
    B_q = torch.randint(-7, 8, (N, R), device=dev, dtype=torch.int8)
    B_scale = torch.rand(R, device=dev, dtype=torch.float16) * 0.1
    B_shifted = (B_q + 8).to(torch.uint8)
    flat = B_shifted.reshape(-1)
    if flat.shape[0] % 2:
        flat = torch.cat([flat, torch.zeros(1, dtype=torch.uint8, device=dev)])
    B_packed = (flat[0::2] | (flat[1::2] << 4)).view(N, R // 2)
    out_trt = int4_matmul(A, B_packed, B_scale, N, R)
    out_ref = _int4_matmul_torch(A, B_packed, B_scale, N, R)
    all_ok &= diff(out_trt, out_ref, "int4_matmul")

    # 2. vq_weighted_sum
    print("\n2. vq_weighted_sum")
    Q, N, C, gs = 32, 512, 256, 4
    G = 32  # groups
    D = G * gs  # 128
    w = torch.rand(Q, N, device=dev, dtype=torch.float32)
    w = torch.softmax(w, dim=-1)
    idx = torch.randint(0, C, (N, G), device=dev, dtype=torch.uint8)
    cb = torch.randn(C, gs, device=dev, dtype=torch.float16) * 0.3
    ch = torch.rand(D, device=dev, dtype=torch.float16) * 0.5 + 0.1
    out_trt = vq_weighted_sum(w, idx, cb, ch)
    out_ref = _vq_weighted_sum_torch(w, idx, cb, ch)
    all_ok &= diff(out_trt, out_ref, "vq_weighted_sum")

    # 2b. vq_weighted_sum_all  — single launch, all KV heads
    print("\n2b. vq_weighted_sum_all")
    nkv_t, gqa_t = 8, 4
    HQ = nkv_t * gqa_t
    D_all = 128
    G_all = 32
    gs_all = 4
    N_all = 512
    C_all = 256
    w_all = torch.rand(HQ, N_all, device=dev, dtype=torch.float32)
    w_all = torch.softmax(w_all, dim=-1)
    idx_all = torch.randint(0, C_all, (nkv_t, N_all, G_all), device=dev, dtype=torch.uint8)
    cb_all = torch.randn(C_all, gs_all, device=dev, dtype=torch.float16) * 0.3
    ch_all = (torch.rand(nkv_t, D_all, device=dev, dtype=torch.float16) * 0.5 + 0.1)
    out_all = vq_weighted_sum_all(w_all, idx_all, cb_all, ch_all, gqa_t)
    # Reference: per-kv-head
    refs = []
    for kv in range(nkv_t):
        w_kv = w_all[kv * gqa_t:(kv + 1) * gqa_t]
        refs.append(_vq_weighted_sum_torch(w_kv, idx_all[kv], cb_all, ch_all[kv]))
    out_ref_all = torch.cat(refs, dim=0)
    all_ok &= diff(out_all, out_ref_all, "vq_weighted_sum_all")

    # 3. fused_attn_decode
    print("\n3. fused_attn_decode")
    H, N, hd = 32, 512, 128
    q = torch.randn(H, hd, device=dev, dtype=torch.float16) * 0.1
    k = torch.randn(N, hd, device=dev, dtype=torch.float16) * 0.1
    v_idx = torch.randint(0, C, (N, G), device=dev, dtype=torch.uint8)
    v_cb = torch.randn(C, gs, device=dev, dtype=torch.float16) * 0.3
    v_ch = torch.rand(hd, device=dev, dtype=torch.float16) * 0.5 + 0.1
    scale = 1.0 / math.sqrt(hd)
    out_t, lse_t = fused_attn_decode(q, k, v_idx, v_cb, v_ch, scale)
    out_r, lse_r = _fused_attn_decode_torch(q, k, v_idx, v_cb, v_ch, scale)
    all_ok &= diff(out_t, out_r, "fused_attn_decode.out")
    all_ok &= diff(lse_t, lse_r, "fused_attn_decode.lse")

    # 4. score_softmax_vq
    print("\n4. score_softmax_vq")
    hd_half = hd // 2
    U = torch.randn(H, hd_half, N, device=dev, dtype=torch.float16) * 0.1
    V = torch.randn(H, hd_half, N, device=dev, dtype=torch.float16) * 0.1
    mean_c = torch.randn(H, hd_half, device=dev, dtype=torch.float16) * 0.05
    mean_s = torch.randn(H, hd_half, device=dev, dtype=torch.float16) * 0.05
    cos_d = torch.randn(hd_half, N, device=dev, dtype=torch.float16) * 0.5
    sin_d = torch.randn(hd_half, N, device=dev, dtype=torch.float16) * 0.5
    inv_sqrt_hd = 1.0 / math.sqrt(hd)
    out_t, m_t, l_t = score_softmax_vq(U, V, mean_c, mean_s, cos_d, sin_d,
                                         v_idx, v_cb, v_ch, inv_sqrt_hd)
    out_r, m_r, l_r = _score_softmax_vq_torch(U, V, mean_c, mean_s, cos_d, sin_d,
                                                v_idx, v_cb, v_ch, inv_sqrt_hd)
    all_ok &= diff(out_t, out_r, "score_softmax_vq.out")
    all_ok &= diff(m_t, m_r, "score_softmax_vq.m")
    all_ok &= diff(l_t, l_r, "score_softmax_vq.l")

    # 5. int4_pack
    print("\n5. int4_pack")
    n = 1024
    r = 192
    q_in = torch.randint(-7, 8, (n, r), device=dev, dtype=torch.int8)
    packed_trt = int4_pack(q_in)
    # reference: match the legacy packing used in cache.py
    flat = (q_in + 8).to(torch.uint8).reshape(-1)
    if flat.shape[0] % 2:
        flat = torch.cat([flat, torch.zeros(1, dtype=torch.uint8, device=dev)])
    packed_ref = flat[0::2] | (flat[1::2] << 4)
    ok = (packed_trt == packed_ref).all().item()
    print(f"  [{'PASS' if ok else 'FAIL'}] int4_pack (bitwise equality: {ok})")
    all_ok &= ok

    # 6. vq_assign
    print("\n6. vq_assign")
    Np, Dp, Kp = 10000, 4, 256
    data = torch.randn(Np, Dp, device=dev, dtype=torch.float32)
    centroids = torch.randn(Kp, Dp, device=dev, dtype=torch.float32)
    idx_trt = vq_assign(data, centroids)
    idx_ref = torch.cdist(data, centroids).argmin(dim=1)
    mismatch = (idx_trt != idx_ref).sum().item()
    ok = mismatch == 0
    print(f"  [{'PASS' if ok else 'FAIL'}] vq_assign (mismatches: {mismatch}/{Np})")
    all_ok &= ok

    # 7. undo_rope_fused
    print("\n7. undo_rope_fused")
    x = torch.randn(64, hd, device=dev, dtype=torch.float16) * 0.1
    positions = torch.arange(64, device=dev).float()
    inv_freq = 1.0 / (500000.0 ** (torch.arange(0, hd, 2, device=dev).float() / hd))
    freqs = positions.unsqueeze(-1) * inv_freq.unsqueeze(0)
    cos = torch.cat([freqs, freqs], dim=-1).cos().to(torch.float16)
    sin = torch.cat([freqs, freqs], dim=-1).sin().to(torch.float16)
    out_trt = undo_rope_fused(x, cos, sin)
    # Reference: standard rotate_half undo
    h2 = hd // 2
    x_lo, x_hi = x[:, :h2].float(), x[:, h2:].float()
    c, s = cos[:, :h2].float(), sin[:, :h2].float()
    out_ref = torch.empty_like(x, dtype=torch.float16)
    out_ref[:, :h2] = (x_lo * c + x_hi * s).half()
    out_ref[:, h2:] = (x_hi * c - x_lo * s).half()
    all_ok &= diff(out_trt, out_ref, "undo_rope_fused")

    # 8. END-TO-END: fused attention vs eager
    print("\n8. end-to-end fused attention vs eager reference")
    from shard.cache import Cache, hadamard, vq_encode
    from shard.attention import compressed_scores, _v_weighted_sum_compressed
    from shard.rope import reapply_rope

    B_e, n_q, nkv, hd_e = 1, 32, 8, 128
    rank, n_comp = 192, 512
    gqa = n_q // nkv
    inv_freq_e = 1.0 / (500000.0 ** (torch.arange(0, hd_e, 2, dtype=torch.float32, device=dev) / hd_e))

    # Build fake compressed K state
    k_mean = torch.randn(nkv * hd_e, device=dev, dtype=torch.float16) * 0.1
    k_basis_fp = torch.randn(nkv * hd_e, rank, device=dev, dtype=torch.float16) * 0.5
    k_coeffs_fp = torch.randint(-7, 8, (B_e * n_comp, rank), device=dev, dtype=torch.float16)
    k_coeff_scale = torch.rand(rank, device=dev, dtype=torch.float16) * 0.1 + 0.01
    k_basis_scale = torch.rand(rank, device=dev, dtype=torch.float16) * 0.1 + 0.01
    k_basis_q = torch.clamp(torch.round(k_basis_fp / k_basis_scale), -127, 127).to(torch.int8)
    k_q = torch.clamp(k_coeffs_fp.round(), -7, 7).to(torch.int8)
    k_shifted = (k_q + 8).to(torch.uint8).reshape(-1)
    if k_shifted.shape[0] % 2:
        k_shifted = torch.cat([k_shifted, torch.zeros(1, dtype=torch.uint8, device=dev)])
    packed = (k_shifted[0::2] | (k_shifted[1::2] << 4))

    v_true = torch.randn(B_e * nkv, n_comp, hd_e, device=dev, dtype=torch.float32) * 0.3
    v_rot = hadamard(v_true)
    v_qdata = vq_encode(v_rot, 4, 256)

    gc = {
        "k_coeffs": packed,
        "k_coeffs_shape": (B_e * n_comp, rank),
        "k_coeff_scale": k_coeff_scale,
        "k_basis_q": k_basis_q,
        "k_basis_scale": k_basis_scale,
        "k_mean": k_mean,
        "compress_start": 4,
        "n_compress": n_comp,
        "batch": B_e,
        "v_qdata": v_qdata,
        "sink_k": torch.zeros(B_e, nkv, 0, hd_e, device=dev, dtype=torch.float16),
        "sink_v": torch.zeros(B_e, nkv, 0, hd_e, device=dev, dtype=torch.float16),
        "win_k": torch.zeros(B_e, nkv, 0, hd_e, device=dev, dtype=torch.float16),
        "win_v": torch.zeros(B_e, nkv, 0, hd_e, device=dev, dtype=torch.float16),
    }

    # Reference
    def unpack_int4(packed, n_total, rank):
        lo = (packed & 0x0F).to(torch.float16) - 8.0
        hi = ((packed >> 4) & 0x0F).to(torch.float16) - 8.0
        return torch.stack([lo, hi], dim=1).flatten()[:n_total * rank].reshape(n_total, rank)

    unp = unpack_int4(packed, B_e * n_comp, rank)
    k_norope = ((unp * k_coeff_scale) @ (k_basis_q.to(torch.float16) * k_basis_scale).T + k_mean)
    k_norope = k_norope.view(B_e, n_comp, nkv, hd_e).permute(0, 2, 1, 3)
    positions_kv = torch.arange(4, 4 + n_comp, device=dev)
    k_rope = reapply_rope(k_norope, positions_kv, inv_freq_e.cpu())

    pos_q = 4 + n_comp + 32
    q_rope = torch.randn(B_e, n_q, 1, hd_e, device=dev, dtype=torch.float16) * 0.3

    # Eager reference attention
    k_exp = k_rope.repeat_interleave(gqa, dim=1).float()
    ref_scores = torch.einsum("bhsd,bhnd->bhsn", q_rope.float(), k_exp) / math.sqrt(hd_e)
    ref_weights = torch.softmax(ref_scores, dim=-1)
    from shard.cache import vq_decode
    v_ref_rot = vq_decode(v_qdata, dev)
    v_ref = hadamard(v_ref_rot).view(B_e, nkv, n_comp, hd_e).float()
    v_ref_exp = v_ref.repeat_interleave(gqa, dim=1)
    ref_out = torch.einsum("bhsk,bhkd->bhsd", ref_weights, v_ref_exp)

    # Shard fused path: compressed_scores → softmax → _v_weighted_sum_compressed
    fused_scores = compressed_scores(q_rope, gc, pos_q, n_q, nkv, hd_e, inv_freq_e).float()
    fused_weights = torch.softmax(fused_scores, dim=-1)
    fused_out = _v_weighted_sum_compressed(fused_weights, gc, B_e, nkv, n_comp, hd_e)

    d_scores = (fused_scores - ref_scores).abs()
    d_out = (fused_out - ref_out).abs()
    print(f"  [{'PASS' if d_scores.max() < 0.05 else 'FAIL'}] "
          f"fused_scores vs eager:  max={d_scores.max().item():.4e}")
    print(f"  [{'PASS' if d_out.max() < 0.05 else 'FAIL'}] "
          f"fused_out vs eager:      max={d_out.max().item():.4e}")
    all_ok &= (d_scores.max() < 0.05 and d_out.max() < 0.05)

    print(f"\n{'='*70}")
    print(f"OVERALL: {'ALL PASS' if all_ok else 'FAILURES DETECTED'}")


@app.local_entrypoint()
def main():
    run.remote()
