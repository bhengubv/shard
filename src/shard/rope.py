"""RoPE utilities."""
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
