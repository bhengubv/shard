"""Data-oblivious per-token quantizer (Lloyd-Max codebook for N(0, 1/d))."""
import math
import torch


# Acklam's inverse normal CDF
_A = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
       1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
_B = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
       6.680131188771972e+01, -1.328068155288572e+01]
_C = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
       -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
_D = [7.784695709041462e-03, 3.224671290700398e-01,
       2.445134137142996e+00, 3.754408661907416e+00]


def _inv_cdf(p):
    lo, hi = 0.02425, 1 - 0.02425
    if p < lo:
        q = math.sqrt(-2 * math.log(p))
        return (((((_C[0]*q+_C[1])*q+_C[2])*q+_C[3])*q+_C[4])*q+_C[5]) / \
               ((((_D[0]*q+_D[1])*q+_D[2])*q+_D[3])*q+1)
    if p <= hi:
        q = p - 0.5; r = q * q
        return (((((_A[0]*r+_A[1])*r+_A[2])*r+_A[3])*r+_A[4])*r+_A[5]) * q / \
               (((((_B[0]*r+_B[1])*r+_B[2])*r+_B[3])*r+_B[4])*r+1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((_C[0]*q+_C[1])*q+_C[2])*q+_C[3])*q+_C[4])*q+_C[5]) / \
            ((((_D[0]*q+_D[1])*q+_D[2])*q+_D[3])*q+1)


def lloyd_max_codebook(d, bits, n_iter=100):
    # Lloyd-Max centroids for N(0, 1/d); closed-form E[X|X∈[a,b]] update
    n = 1 << bits
    sigma = 1.0 / math.sqrt(d)
    c = [_inv_cdf((i + 0.5) / n) * sigma for i in range(n)]
    phi = lambda z: math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    Phi = lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2)))
    for _ in range(n_iter):
        bounds = [-1e9] + [(c[k - 1] + c[k]) / 2 for k in range(1, n)] + [1e9]
        nc = []
        for k in range(n):
            a, b = bounds[k], bounds[k + 1]
            m = Phi(b / sigma) - Phi(a / sigma)
            nc.append(c[k] if m < 1e-12 else (phi(a / sigma) - phi(b / sigma)) * sigma / m)
        if max(abs(x - y) for x, y in zip(nc, c)) < 1e-9: break
        c = nc
    return torch.tensor(c, dtype=torch.float32)


_codebooks = {}
def get_codebook(d, bits, device):
    k = (d, bits)
    if k not in _codebooks: _codebooks[k] = lloyd_max_codebook(d, bits)
    return _codebooks[k].to(device)
