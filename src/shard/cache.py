"""KV cache compression utilities — hadamard rotation + k-means VQ."""
import math
import torch

from .rope import pack_nbits, unpack_nbits


def hadamard(x):
    # normalized butterfly WHT; self-inverse on last dim (power of 2)
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
    # per-channel normalize, group channels, k-means each group
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
