"""KV cache compression utilities — hadamard rotation + k-means quantization."""
import math
import torch


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
