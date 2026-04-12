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
