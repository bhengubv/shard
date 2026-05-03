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
