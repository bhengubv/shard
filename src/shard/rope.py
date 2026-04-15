"""RoPE utilities, DP bit allocation, variable-bit quantization, bit packing."""
import math
import zlib
import numpy as np
import torch
from torch import Tensor


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


def _groups(alloc):
    g, i = [], 0
    while i < len(alloc):
        bits, gs = alloc[i], 1
        while i + gs < len(alloc) and alloc[i + gs] == bits and gs < 1024:
            gs += 1
        g.append((i, gs, bits)); i += gs
    return g


def quantize_group(projected, alloc):
    n = projected.shape[0]
    groups = _groups(alloc)
    headers, payload = [], []
    for start, gs, bits in groups:
        if bits == 0: continue
        block = projected[:, start:start + gs]
        lo = block.min(dim=0).values
        rng = (block.max(dim=0).values - lo).clamp(min=1e-6)
        headers.append(np.stack([lo.cpu().numpy().astype(np.float32),
                                  rng.cpu().numpy().astype(np.float32)], axis=1).tobytes())
        levels = (1 << bits) - 1
        arr = (((block - lo) / rng).clamp(0, 1) * levels).round().long().clamp(0, levels).byte().cpu().numpy()
        if bits == 2:
            if n % 4: arr = np.pad(arr, ((0, (4 - n % 4) % 4), (0, 0)))
            for i in range(gs):
                c = arr[:, i]
                payload.append(((c[0::4] | (c[1::4] << 2) | (c[2::4] << 4) | (c[3::4] << 6)).astype(np.uint8).tobytes()))
        elif bits == 4:
            if n % 2: arr = np.pad(arr, ((0, 1), (0, 0)))
            for i in range(gs):
                c = arr[:, i]
                payload.append((((c[0::2] & 0x0F) | ((c[1::2] & 0x0F) << 4)).astype(np.uint8).tobytes()))
        else:
            for i in range(gs):
                payload.append(arr[:, i].tobytes())
    nz = [(s, g, b) for s, g, b in groups if b > 0]
    meta = np.array([len(nz), n], dtype=np.int32).tobytes()
    g_bytes = b"".join(np.array([s, g, b], dtype=np.int16).tobytes() for s, g, b in nz)
    return zlib.compress(meta + g_bytes + b"".join(headers) + b"".join(payload), level=6)


def dequantize_group(compressed, alloc, n, device):
    raw = zlib.decompress(compressed)
    recon = torch.zeros(n, len(alloc), device=device)
    ng = int(np.frombuffer(raw[:4], dtype=np.int32)[0])
    off = 8
    groups = []
    for _ in range(ng):
        s, g, b = np.frombuffer(raw[off:off + 6], dtype=np.int16)
        groups.append((int(s), int(g), int(b))); off += 6
    nz = sum(g for _, g, _ in groups)
    hdr = np.frombuffer(raw[off:off + nz * 8], dtype=np.float32).reshape(-1, 2)
    shifts = torch.from_numpy(hdr[:, 0].copy()).to(device)
    scales = torch.from_numpy(hdr[:, 1].copy()).to(device)
    p = off + nz * 8
    ci = 0
    for start, gs, bits in groups:
        levels = (1 << bits) - 1
        cols = []
        for _ in range(gs):
            if bits == 2:
                nb = (n + 3) // 4
                pk = np.frombuffer(raw[p:p + nb], dtype=np.uint8); p += nb
                cols.append(np.stack([pk & 3, (pk >> 2) & 3, (pk >> 4) & 3, (pk >> 6) & 3], axis=1).ravel()[:n])
            elif bits == 4:
                nb = (n + 1) // 2
                pk = np.frombuffer(raw[p:p + nb], dtype=np.uint8); p += nb
                cols.append(np.stack([pk & 0x0F, (pk >> 4) & 0x0F], axis=1).ravel()[:n])
            else:
                cols.append(np.frombuffer(raw[p:p + n], dtype=np.uint8).copy()); p += n
        arr = torch.from_numpy(np.stack(cols, axis=1)).to(device).float()
        s = scales[ci:ci + gs].unsqueeze(0)
        sh = shifts[ci:ci + gs].unsqueeze(0)
        recon[:, start:start + gs] = arr / levels * s + sh
        ci += gs
    return recon


def pack_nbits(data, bits):
    if bits == 8:
        return data.byte().reshape(-1), data.reshape(-1).shape[0]
    flat = data.reshape(-1).long()
    n = flat.shape[0]
    if bits == 4:
        if n % 2: flat = torch.nn.functional.pad(flat, (0, 1))
        return (flat[0::2] | (flat[1::2] << 4)).byte(), n
    if bits == 2:
        pad = (4 - n % 4) % 4
        if pad: flat = torch.nn.functional.pad(flat, (0, pad))
        return (flat[0::4] | (flat[1::4] << 2) | (flat[2::4] << 4) | (flat[3::4] << 6)).byte(), n
    if bits == 1:
        pad = (8 - n % 8) % 8
        if pad: flat = torch.nn.functional.pad(flat, (0, pad))
        out = torch.zeros(flat.shape[0] // 8, dtype=torch.uint8, device=data.device)
        for i in range(8): out |= flat[i::8].byte() << i
        return out, n
    mask = (1 << bits) - 1
    nb = (n * bits + 7) // 8
    buf = torch.zeros(nb + 1, dtype=torch.int32, device=data.device)
    pos = torch.arange(n, device=data.device) * bits
    bi = pos >> 3; off = (pos & 7).int()
    vals = (flat & mask).int()
    buf.scatter_add_(0, bi, vals << off)
    ov = (off + bits) > 8
    if ov.any():
        buf.scatter_add_(0, bi[ov] + 1, vals[ov] >> (8 - off[ov]))
    return buf[:nb].byte(), n


def unpack_nbits(packed, bits, n):
    if bits == 8:
        return packed[:n].byte()
    if bits == 4:
        return torch.stack([packed & 0x0F, (packed >> 4) & 0x0F], dim=-1).reshape(-1)[:n].byte()
    if bits == 2:
        return torch.stack([packed & 3, (packed >> 2) & 3, (packed >> 4) & 3, (packed >> 6) & 3],
                            dim=-1).reshape(-1)[:n].byte()
    if bits == 1:
        out = torch.zeros(packed.shape[0] * 8, dtype=torch.uint8, device=packed.device)
        for i in range(8): out[i::8] = (packed >> i) & 1
        return out[:n]
    mask = (1 << bits) - 1
    pos = torch.arange(n, device=packed.device) * bits
    bi = pos >> 3; off = pos & 7
    p = packed.long()
    vals = (p[bi] >> off) & mask
    ov = (off + bits) > 8
    if ov.any():
        oi = bi[ov] + 1
        valid = oi < packed.shape[0]
        if valid.any():
            hi = torch.zeros_like(oi)
            hi[valid] = p[oi[valid]]
            vals[ov] |= (hi << (8 - off[ov])) & mask
    return vals.byte()
