"""Quantization-based clustering utilities.

This module implements the Quantization-Based Clustering Algorithm (QBCA)
from Yu & Wong (2010).
"""

import numpy as np
import time
from collections import defaultdict


class QBCA:
    """Quantization-Based Clustering Algorithm (QBCA).

    Reference: Yu & Wong (2010), Pattern Recognition 43.
    """

    def __init__(self, k, threshold=0.0001, max_iter=30):
        """Initialize the QBCA instance.

        Parameters
        ----------
        k : int
            Number of clusters.
        threshold : float
            Convergence threshold on average squared seed shift.
        max_iter : int
            Maximum number of iterations.
        """
        self.k = k
        self.threshold = threshold
        self.max_iter = max_iter
        self.seeds = None
        self.bins = {}
        self.bin_bounds = {}

        # Metric tracking
        self.init_time = 0
        self.total_dist_calls = 0
        self.iterations = 0

    def quantization(self, P):
        """Step 1: The quantization process.

        This builds an m-dimensional histogram and shrinks bin bounds to the
        contained points.
        """
        n, m = P.shape
        # Exact rho from Equation 19.
        self.rho = int(np.floor(np.log(n) / np.log(m))) if m > 1 else int(
            np.sqrt(n)
        )

        p_min = P.min(axis=0)
        p_max = P.max(axis=0)
        # Size of histogram bins per dimension.
        bin_widths = (p_max - p_min) / self.rho
        bin_widths[bin_widths == 0] = 1.0

        # Quantization function.
        self.bins = defaultdict(list)
        for i, point in enumerate(P):
            # Calculate linearized index.
            xi = np.floor((point - p_min) / bin_widths).astype(int)
            xi = np.clip(xi, 0, self.rho - 1)

            # Linearize the m-dimensional histogram.
            bin_id = 0
            for dim in range(m):
                bin_id += xi[dim] * (self.rho ** (m - 1 - dim))

            self.bins[bin_id].append(i)

        # Shrinking process: tighten per-bin bounds to contained points.
        self.bin_bounds = {}
        for bid, indices in self.bins.items():
            pts = P[indices]
            self.bin_bounds[bid] = (pts.min(axis=0), pts.max(axis=0))

        return self.rho

    def _dist_min(self, seed, bid):
        """Definition 2: Minimum distance (Equation 10).

        Returns a lower bound on distance from `seed` to the bin.
        """
        b_min, b_max = self.bin_bounds[bid]
        # Find closest point in the shrunken bin.
        closest_point = np.where(
            seed < b_min, b_min, np.where(seed > b_max, b_max, seed)
        )

        return np.linalg.norm(seed - closest_point)

    def _dist_max(self, seed, bid):
        """Definition 3: Maximum distance (upper bound).

        This returns an upper bound on distance from `seed` to any point in
        the shrunken bin.
        """
        b_min, b_max = self.bin_bounds[bid]
        # Find furthest corner in the shrunken bin.
        furthest_point = np.where(seed >= (b_min + b_max) / 2, b_min, b_max)

        return np.linalg.norm(seed - furthest_point)

    def cci(self, P):
        """Step 2: Cluster center initialization.

        Selects centers from the densest histogram bins as initial seeds.
        """
        # Count points in each bin and sort by density.
        bin_counts = {bid: len(indices) for bid, indices in self.bins.items()}
        sorted_bins = sorted(
            bin_counts.keys(), key=lambda x: bin_counts[x], reverse=True
        )

        seed_list = []
        for bid in sorted_bins:
            if len(seed_list) < self.k:
                # Use the mean point of the dense bin as a seed.
                seed_list.append(P[self.bins[bid]].mean(axis=0))

        self.seeds = np.array(seed_list)

    def cca(self, P):
        """Step 3: Cluster center assignment.

        Uses per-bin distance bounds to prune seed candidates and avoid
        unnecessary point-to-seed distance computations.
        """
        new_assignments = np.zeros(P.shape[0], dtype=int)

        for bid, indices in self.bins.items():
            # Compute min of the seeds' max-distance bounds.
            d_maxes = [self._dist_max(s, bid) for s in self.seeds]
            d_star_max = min(d_maxes)

            # Prune seeds whose min-distance is larger than d_star_max.
            candidates = [
                i
                for i, s in enumerate(self.seeds)
                if self._dist_min(s, bid) <= d_star_max
            ]

            if len(candidates) == 1:
                # Direct assignment without per-point distance computation.
                new_assignments[indices] = candidates[0]
            else:
                pts = P[indices]
                for idx_in_bin, p_idx in enumerate(indices):
                    p = pts[idx_in_bin]
                    dists = []
                    for c_idx in candidates:
                        dists.append(np.linalg.norm(p - self.seeds[c_idx]))
                        self.total_dist_calls += 1

                    new_assignments[p_idx] = candidates[np.argmin(dists)]

        return new_assignments

    def get_metrics(self, P, assignments):
        """Calculate algorithm metrics.

        Returns average distortion and average number of distance
        computations per iteration.
        """
        # Average distortion (sum of squared errors divided by n).
        total_error = 0
        for h in range(self.k):
            cluster_pts = P[assignments == h]
            if len(cluster_pts) > 0:
                total_error += np.sum((cluster_pts - self.seeds[h]) ** 2)

        avg_distortion = total_error / P.shape[0]

        # Average number of distance computations per iteration.
        avg_dist_comp = (
            self.total_dist_calls / (P.shape[0] * self.iterations)
            if self.iterations > 0
            else 0
        )

        return avg_distortion, avg_dist_comp

    def fit(self, P):
        """Run the QBCA algorithm until convergence or max iterations.

        Returns the final assignments, seed centers and iteration history.
        """
        self.total_dist_calls = 0

        # Initialization stage.
        start_init = time.time()
        self.quantization(P)
        self.cci(P)
        self.init_time = time.time() - start_init

        history = []
        for t in range(self.max_iter):
            old_seeds = self.seeds.copy()
            assignments = self.cca(P)

            # Recompute centers (cluster means).
            for h in range(self.k):
                cluster_pts = P[assignments == h]
                if len(cluster_pts) > 0:
                    self.seeds[h] = cluster_pts.mean(axis=0)

            # Calculate gap delta (average squared seed shift).
            shift = np.mean(
                [np.sum((old_seeds[h] - self.seeds[h]) ** 2) for h in range(self.k)]
            )
            history.append({"iteration": t, "shift": shift})
            self.iterations = t + 1

            if shift < self.threshold:  # Termination threshold.
                break

        return assignments, self.seeds, history
