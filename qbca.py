"""Implementation of Image Segmentation using QBCA."""

import numpy as np
import time
from collections import defaultdict


class QBCA:
    """Quantization-Based Clustering Algorithm (QBCA) with Hierarchical Structure.

    Ref: Yu & Wong (2010), Pattern Recognition 43. [cite: 2].
    """

    def __init__(self, k, threshold=0.01, max_iter=50):
        """Initialize a QBCA instance with clustering parameters."""
        self.k = k
        self.threshold = threshold
        self.max_iter = max_iter
        self.seeds = None
        self.bins = {}
        self.bin_bounds = {}
        self.parents = {}

        # Metric tracking [cite: 563]
        self.init_time = 0
        self.total_dist_calls = 0
        self.iterations = 0

    def quantization(self, P):
        """Step 1: The quantization process & Hierarchical Construction [cite: 218, 483]."""
        n, m = P.shape
        # Calculate rho (Equation 19) [cite: 220]
        self.rho = int(np.floor(np.log(n) / np.log(m))) if m > 1 else int(np.sqrt(n))

        p_min, p_max = P.min(axis=0), P.max(axis=0)
        bin_widths = (p_max - p_min) / self.rho
        bin_widths[bin_widths == 0] = 1.0

        self.bins = defaultdict(list)
        self.parents = defaultdict(lambda: {'children': set(), 'bounds': [None, None]})

        # Quantization and parent mapping [cite: 227, 494]
        for i, point in enumerate(P):
            xi = np.floor((point - p_min) / bin_widths).astype(int)
            xi = np.clip(xi, 0, self.rho - 1)

            # Linearize bin index (Equation 21) [cite: 227]
            bin_id = 0
            for dim in range(m):
                bin_id += xi[dim] * (self.rho ** (m - 1 - dim))
            self.bins[bin_id].append(i)

            # Define hierarchical structure (Section 6) [cite: 483, 494]
            parent_xi = xi // 2
            parent_id = 0
            for dim in range(m):
                parent_id += parent_xi[dim] * ((self.rho // 2 + 1) ** (m - 1 - dim))
            self.parents[parent_id]['children'].add(bin_id)

        # Shrinking process for child bins (Equation 27) [cite: 476, 478]
        for bid, indices in self.bins.items():
            pts = P[indices]
            self.bin_bounds[bid] = (pts.min(axis=0), pts.max(axis=0))

        # Shrinking process for parents (Lemma 4) [cite: 522, 528]
        for _pid, data in self.parents.items():
            child_mins = [self.bin_bounds[cid][0] for cid in data['children']]
            child_maxes = [self.bin_bounds[cid][1] for cid in data['children']]
            data['bounds'] = (np.min(child_mins, axis=0), np.max(child_maxes, axis=0))

    def _get_candidates(self, seed_indices, bounds):
        """Determine seed candidates based on hierarchical bounds and distance bounds."""
        b_min, b_max = bounds

        # Minimum distance lower bound (Equation 10) [cite: 174]
        def d_min(s):
            closest = np.where(s < b_min, b_min, np.where(s > b_max, b_max, s))
            return np.linalg.norm(s - closest)

        # Maximum distance upper bound (Equation 12) [cite: 177]
        def d_max(s):
            furthest = np.where(s >= (b_min + b_max) / 2, b_min, b_max)
            return np.linalg.norm(s - furthest)

        # Find the min of max distances (Equation 15) [cite: 192]
        d_star_max = min(d_max(self.seeds[i]) for i in seed_indices)

        # Lemma 1 filter: Reject seeds further than d* [cite: 188, 780]
        return [i for i in seed_indices if d_min(self.seeds[i]) <= d_star_max]

    def cci(self, P):
        """Step 2: Cluster center initialization."""
        bin_counts = {bid: len(indices) for bid, indices in self.bins.items()}
        sorted_bins = sorted(bin_counts.keys(), key=lambda x: bin_counts[x], reverse=True)

        seed_list = []
        for bid in sorted_bins:
            if len(seed_list) < self.k:
                seed_list.append(P[self.bins[bid]].mean(axis=0))
        self.seeds = np.array(seed_list)

    def cca(self, P):
        """Step 3: Cluster center assignment (CCA) with early pruning [cite: 353, 487]."""
        new_assignments = np.zeros(P.shape[0], dtype=int)
        all_seed_indices = list(range(self.k))

        for _pid, pdata in self.parents.items():
            # Parent level pruning (Lemma 3) [cite: 488, 796]
            parent_candidates = self._get_candidates(all_seed_indices, pdata['bounds'])

            for bid in pdata['children']:
                indices = self.bins[bid]
                # Child level inheritance [cite: 488, 807]
                candidates = self._get_candidates(parent_candidates, self.bin_bounds[bid])

                if len(candidates) == 1:
                    # EARLY PRUNING: No point-level distance math needed [cite: 380]
                    new_assignments[indices] = candidates[0]
                else:
                    # Fallback to point-level assignment [cite: 380]
                    pts = P[indices]
                    for idx_in_bin, p_idx in enumerate(indices):
                        p = pts[idx_in_bin]
                        dists = [np.linalg.norm(p - self.seeds[c]) for c in candidates]
                        self.total_dist_calls += len(candidates)
                        new_assignments[p_idx] = candidates[np.argmin(dists)]

        return new_assignments

    def dunn_index(self, P, assignments):
        """Quality measure for clustering (Dunn Index)."""
        if self.k < 2:
            return 0

        # Inter-cluster distance (Equation 24) [cite: 415]
        inter_dist = float('inf')
        for i in range(self.k):
            for j in range(i + 1, self.k):
                pts_i = P[assignments == i]
                pts_j = P[assignments == j]
                if len(pts_i) == 0 or len(pts_j) == 0:
                    continue
                # Approximation: Distance between centers for performance
                dist = np.linalg.norm(self.seeds[i] - self.seeds[j])
                if dist < inter_dist:
                    inter_dist = dist

        # Intra-cluster diameter (Equation 25) [cite: 416]
        max_diam = 0
        for i in range(self.k):
            pts = P[assignments == i]
            if len(pts) == 0:
                continue
            # Approximation: Max distance from center
            diam = np.max(np.linalg.norm(pts - self.seeds[i], axis=1)) * 2
            if diam > max_diam:
                max_diam = diam

        return inter_dist / max_diam if max_diam > 0 else 0

    def fit(self, P):
        """Run QBCA until convergence."""
        self.total_dist_calls = 0
        start_init = time.time()
        self.quantization(P)
        self.cci(P)
        self.init_time = time.time() - start_init

        history = []
        for t in range(self.max_iter):
            old_seeds = self.seeds.copy()
            assignments = self.cca(P)

            # Recompute centers (Equation 8) [cite: 101]
            for h in range(self.k):
                cluster_pts = P[assignments == h]
                if len(cluster_pts) > 0:
                    self.seeds[h] = cluster_pts.mean(axis=0)

            # Calculate gap delta (Equations 17-18) [cite: 212]
            shift = np.mean([np.sum((old_seeds[h] - self.seeds[h])**2) for h in range(self.k)])
            history.append({"iteration": t, "shift": shift})
            self.iterations = t + 1
            if shift < self.threshold:
                break

        return assignments, self.seeds, history
