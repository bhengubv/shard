"""RoPE utilities and DP bit allocation."""
import numpy as np
import torch


def rotate_half(x):
    # HuggingFace convention: [-x[..., d//2:], x[..., :d//2]]
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def rope_cos_sin(positions, inv_freq):
    if positions.dim() == 1:
        positions = positions.unsqueeze(0)
    freqs = inv_freq[None, :, None].float() @ positions[:, None, :].float()
    emb = torch.cat([freqs.transpose(1, 2)] * 2, dim=-1)
    return emb.cos(), emb.sin()


def undo_rope(keys, positions, inv_freq):
    cos, sin = rope_cos_sin(positions, inv_freq.to(keys.device))
    while cos.dim() < keys.dim():
        cos = cos.unsqueeze(-3); sin = sin.unsqueeze(-3)
    return keys * cos - rotate_half(keys) * sin


def reapply_rope(keys, positions, inv_freq):
    cos, sin = rope_cos_sin(positions, inv_freq.to(keys.device))
    while cos.dim() < keys.dim():
        cos = cos.unsqueeze(-3); sin = sin.unsqueeze(-3)
    return keys * cos + rotate_half(keys) * sin


def get_rope_inv_freq(model):
    backbone = getattr(model, "model", getattr(model, "transformer", None))
    for t in (backbone, getattr(backbone, "layers", [None])[0] if backbone else None):
        if t is None: continue
        rot = getattr(t, "rotary_emb", None) or \
              getattr(getattr(t, "self_attn", None) or object(), "rotary_emb", None)
        if rot is not None and hasattr(rot, "inv_freq"):
            return rot.inv_freq.detach().float().cpu()
    cfg = model.config
    hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    theta = getattr(cfg, "rope_theta", 10000.0)
    rp = getattr(cfg, "rope_parameters", None)
    if isinstance(rp, dict) and "rope_theta" in rp:
        theta = rp["rope_theta"]
    return 1.0 / (theta ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))


def dp_bit_allocation(variances, total_budget):
    """DP over groups of 64 components, allocating 0/2/4/6/8 bits each.
    0-bit is penalized 4x over its MSE to discourage dropping directions."""
    GROUP, OPTIONS, DROP, UNIT = 64, [0, 2, 4, 6, 8], 4.0, 16
    v = variances.float().cpu()
    d = v.shape[0]
    ng = (d + GROUP - 1) // GROUP
    gs = [min(GROUP, d - g * GROUP) for g in range(ng)]
    gv = [v[g * GROUP: g * GROUP + gs[g]].sum().item() for g in range(ng)]

    mu = total_budget // UNIT + 1
    INF = float("inf")
    dp = [[INF] * (mu + 1) for _ in range(ng + 1)]
    back = [[0] * (mu + 1) for _ in range(ng + 1)]
    dp[0] = [0.0] * (mu + 1)
    for g in range(ng):
        for b in range(mu + 1):
            for bits in OPTIONS:
                cost = 0 if bits == 0 else (gs[g] * bits + UNIT - 1) // UNIT
                err = gv[g] * DROP if bits == 0 else gv[g] / (3.0 * (1 << (2 * bits)))
                if b - cost < 0: continue
                c = dp[g][b - cost] + err
                if c < dp[g + 1][b]:
                    dp[g + 1][b] = c
                    back[g + 1][b] = bits

    best = min(range(mu + 1), key=lambda b: dp[ng][b])
    alloc = [0] * d
    b = best
    for g in range(ng - 1, -1, -1):
        bits = back[g + 1][b]
        for i in range(g * GROUP, g * GROUP + gs[g]):
            alloc[i] = bits
        if bits > 0:
            b -= (gs[g] * bits + UNIT - 1) // UNIT
    return alloc
