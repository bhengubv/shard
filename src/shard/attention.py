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
