"""Lloyd-Max optimal scalar quantizer for the Gaussian distribution.

After rotating a d-dimensional unit vector by a random orthogonal matrix,
each coordinate follows approximately N(0, 1/d) for d >= 64.
We solve the Lloyd-Max conditions to find optimal centroids.
"""

import math
from functools import lru_cache

import torch


def solve_lloyd_max(d: int, bits: int, max_iter: int = 200, tol: float = 1e-10):
    """Solve Lloyd-Max optimal quantizer for N(0, 1/d).

    Uses fully vectorized operations:
    - Analytical Gaussian CDF via torch.erf for centroid updates
    - No per-element Python loops in the hot path

    Returns:
        centroids: sorted tensor of 2^bits optimal centroids
        boundaries: sorted tensor of 2^bits - 1 boundaries
    """
    n_levels = 2 ** bits
    sigma = 1.0 / math.sqrt(d)

    # Initialize centroids uniformly in [-3.5*sigma, 3.5*sigma]
    lo, hi = -3.5 * sigma, 3.5 * sigma
    centroids = torch.linspace(lo, hi, n_levels + 2)[1:-1]  # n_levels points, excluding endpoints

    for _ in range(max_iter):
        # Step 1: boundaries = midpoints between adjacent centroids
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0

        # Step 2: update centroids via analytical Gaussian conditional mean.
        # For N(0, sigma^2) in interval [a, b]:
        #   E[X | a < X < b] = sigma * (phi(a/sigma) - phi(b/sigma))
        #                       / (Phi(b/sigma) - Phi(a/sigma))
        # where phi = standard normal PDF, Phi = CDF.
        # phi(x) = exp(-x^2/2) / sqrt(2pi), Phi(x) = 0.5*(1+erf(x/sqrt(2)))
        edges = torch.zeros(n_levels + 1)
        edges[0] = lo * 3
        edges[1:-1] = boundaries
        edges[-1] = hi * 3

        # Standardize edges
        z = edges / sigma

        # phi(z) = exp(-z^2/2) / sqrt(2*pi)
        neg_half_z2 = -0.5 * z * z
        phi_z = neg_half_z2.exp() * (1.0 / math.sqrt(2.0 * math.pi))

        # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
        cdf_z = 0.5 * (1.0 + torch.erf(z * (1.0 / math.sqrt(2.0))))

        # Conditional means: sigma * (phi(z_a) - phi(z_b)) / (Phi(z_b) - Phi(z_a))
        phi_diff = phi_z[:-1] - phi_z[1:]   # phi(z_a) - phi(z_b), shape (n_levels,)
        cdf_diff = cdf_z[1:] - cdf_z[:-1]   # Phi(z_b) - Phi(z_a), shape (n_levels,)

        # Guard against tiny intervals
        valid = cdf_diff > 1e-15
        new_centroids = torch.where(
            valid,
            sigma * phi_diff / cdf_diff.clamp(min=1e-15),
            centroids,
        )

        # Check convergence
        max_shift = (new_centroids - centroids).abs().max().item()
        centroids = new_centroids
        if max_shift < tol:
            break

    boundaries = (centroids[:-1] + centroids[1:]) / 2.0
    return centroids.clone(), boundaries.clone()


@lru_cache(maxsize=None)
def _cached_solve(d: int, bits: int):
    """Cache (d, bits) -> codebook to avoid redundant computation."""
    return solve_lloyd_max(d, bits)


class LloydMaxCodebook:
    """Precomputed Lloyd-Max codebook for a given dimension and bit-width.

    Results are cached by (d, bits) so that multiple layers with the same
    head_dim and bit-width share one codebook.
    """

    def __init__(self, d: int, bits: int):
        import time
        self.d = d
        self.bits = bits
        self.n_levels = 2 ** bits
        t0 = time.perf_counter()
        self.centroids, self.boundaries = _cached_solve(d, bits)
        dt = (time.perf_counter() - t0) * 1000
        tag = "(cached)" if dt < 0.1 else ""
        print(f"[TurboQuant] Lloyd-Max codebook: d={d}, bits={bits}, levels={self.n_levels} "
              f"{tag} {dt:.1f} ms", flush=True)

    def __repr__(self):
        return f"LloydMaxCodebook(d={self.d}, bits={self.bits}, levels={self.n_levels})"
