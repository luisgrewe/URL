"""Quantization-Based Clustering Algorithm (QBCA) for image segmentation."""

import numpy as np
import time
from collections import defaultdict


class QBCA:
    """QBCA with hierarchical bin pruning for efficient clustering."""

    def __init__(self, k, threshold=0.0001, max_iter=50):
        """Initialize QBCA hyperparameters and runtime state."""
        self.k = k
        self.threshold = threshold
        self.max_iter = max_iter
        self.seeds = None
        self.bins = {}
        self.bin_bounds = {}
        self.bin_means = {}
        self.bin_counts = {}
        self.parents = {}

        # Runtime metrics used for empirical evaluation.
        self.init_time = 0
        self.total_dist_calls = 0
        self.iterations = 0

    def quantization(self, P):
        """Quantize points into bins and construct a parent-child hierarchy."""
        n, m = P.shape
        self.rho = int(np.floor(np.log(n) / np.log(m))) if m > 1 else int(np.sqrt(n))
        self.rho = max(self.rho, 1)

        # Reinitialize caches so repeated `fit` calls do not reuse stale state.
        self.bins = {}
        self.bin_bounds = {}
        self.bin_means = {}
        self.bin_counts = {}
        self.parents = defaultdict(lambda: {'children': set(), 'bounds': [None, None]})

        p_min, p_max = P.min(axis=0), P.max(axis=0)
        bin_widths = (p_max - p_min) / self.rho
        bin_widths[bin_widths == 0] = 1.0

        # Assign each point to a grid cell in a fully vectorized manner.
        xi = np.floor((P - p_min) / bin_widths).astype(int)
        xi = np.clip(xi, 0, self.rho - 1)

        # Convert multi-dimensional bin coordinates into scalar bin identifiers.
        powers = self.rho ** np.arange(m - 1, -1, -1)
        bin_ids = np.dot(xi, powers)

        # Group points by bin identifier using sorted indices.
        sort_idx = np.argsort(bin_ids)
        sorted_bin_ids = bin_ids[sort_idx]

        # Split sorted indices at bin-identifier boundaries.
        _, unique_indices = np.unique(sorted_bin_ids, return_index=True)
        split_indices = unique_indices[1:]
        indices_per_bin = np.split(sort_idx, split_indices)
        unique_bin_ids = sorted_bin_ids[unique_indices]

        for bid, indices in zip(unique_bin_ids, indices_per_bin):
            bid_int = int(bid)
            self.bins[bid_int] = indices
            # Store per-bin sufficient statistics for downstream updates.
            pts = P[indices]
            self.bin_bounds[bid_int] = (pts.min(axis=0), pts.max(axis=0))
            self.bin_means[bid_int] = pts.mean(axis=0)
            self.bin_counts[bid_int] = len(indices)

        # Construct a coarser parent lattice for hierarchical candidate pruning.
        parent_powers = (self.rho // 2 + 1) ** np.arange(m - 1, -1, -1)

        # Decode child coordinates and map them to parent identifiers in one pass.
        xi_coords = (unique_bin_ids[:, None] // powers[None, :]) % self.rho
        parent_xi = xi_coords // 2
        parent_ids = np.dot(parent_xi, parent_powers)

        for parent_id, bid in zip(parent_ids, unique_bin_ids):
            self.parents[int(parent_id)]['children'].add(int(bid))

        # Aggregate child bounds to define each parent bounding box.
        for _pid, data in self.parents.items():
            child_mins = [self.bin_bounds[cid][0] for cid in data['children']]
            child_maxes = [self.bin_bounds[cid][1] for cid in data['children']]
            data['bounds'] = (np.min(child_mins, axis=0), np.max(child_maxes, axis=0))

    def _get_candidates(self, seed_indices, bounds):
        """Prune centroid candidates using lower and upper distance bounds."""
        if not seed_indices:
            return []

        b_min, b_max = bounds
        seeds = self.seeds[seed_indices]
        midpoint = (b_min + b_max) / 2.0

        # Lower bound: distance to the closest point within the bounding box.
        closest = np.where(seeds < b_min, b_min, np.where(seeds > b_max, b_max, seeds))
        d_min_vals = np.linalg.norm(seeds - closest, axis=1)

        # Upper bound: distance to the farthest corner implied by the midpoint rule.
        furthest = np.where(seeds >= midpoint, b_min, b_max)
        d_max_vals = np.linalg.norm(seeds - furthest, axis=1)

        # Keep distance-count accounting consistent with scalar implementations.
        self.total_dist_calls += 2 * len(seed_indices)

        # Apply the pruning criterion: keep candidates with admissible lower bounds.
        d_star_max = np.min(d_max_vals)
        valid_idx = np.flatnonzero(d_min_vals <= d_star_max)
        return [seed_indices[i] for i in valid_idx]

    def cci(self, P):
        """Initialize centroids from the most populated quantization bins."""
        # Rank bins by occupancy to prioritize high-density regions.
        sorted_bins = sorted(self.bin_counts.items(), key=lambda x: x[1], reverse=True)

        seed_list = []

        # Select up to `k` bin means as initial centroids.
        for bid, _count in sorted_bins:
            seed_list.append(self.bin_means[bid])
            if len(seed_list) == self.k:
                break

        # If non-empty bins are insufficient, sample additional points uniformly.
        # This guards against index errors when `k` exceeds occupied-bin count.
        while len(seed_list) < self.k:
            random_idx = np.random.randint(0, P.shape[0])
            seed_list.append(P[random_idx])

        self.seeds = np.array(seed_list)

    def cca(self):
        """Assign each bin to a centroid using hierarchical pruning."""
        bin_assignments = {}
        all_seed_indices = list(range(self.k))

        for _pid, pdata in self.parents.items():
            # First prune candidates at the parent level.
            parent_candidates = self._get_candidates(all_seed_indices, pdata['bounds'])

            for bid in pdata['children']:
                # Refine candidates at the child-bin level.
                candidates = self._get_candidates(parent_candidates, self.bin_bounds[bid])

                if len(candidates) == 1:
                    # Early exit when pruning identifies a unique centroid.
                    bin_assignments[bid] = candidates[0]
                else:
                    # Otherwise, assign via nearest centroid to the bin mean.
                    p = self.bin_means[bid]
                    dists = [np.linalg.norm(p - self.seeds[c]) for c in candidates]
                    self.total_dist_calls += len(candidates)
                    bin_assignments[bid] = candidates[np.argmin(dists)]

        return bin_assignments

    def dunn_index(self, P, assignments):
        """Compute the Dunn index as an internal clustering validity measure."""
        if self.k < 2:
            return 0

        # Minimum pairwise centroid separation (inter-cluster distance).
        inter_dist = float('inf')
        for i in range(self.k):
            for j in range(i + 1, self.k):
                dist = np.linalg.norm(self.seeds[i] - self.seeds[j])
                if dist < inter_dist:
                    inter_dist = dist

        # Maximum cluster diameter via bounding-box approximation.
        max_diam = 0
        for i in range(self.k):
            # Extract all points currently assigned to cluster `i`.
            pts = P[assignments == i]
            if len(pts) == 0:
                continue

            # Use box diagonal as a low-cost proxy for exact cluster diameter.
            c_min, c_max = pts.min(axis=0), pts.max(axis=0)
            diam = np.linalg.norm(c_max - c_min)
            if diam > max_diam:
                max_diam = diam

        return inter_dist / max_diam if max_diam > 0 else 0

    def fit(self, P):
        """Fit QBCA iteratively until convergence or maximum iterations."""
        self.total_dist_calls = 0
        start_init = time.time()
        self.quantization(P)
        self.cci(P)
        self.init_time = time.time() - start_init

        history = []
        for t in range(self.max_iter):
            old_seeds = self.seeds.copy()
            bin_assignments = self.cca()

            # Accumulate weighted bin statistics in one pass to avoid repeated scans.
            weight_sums = np.zeros((self.k, P.shape[1]), dtype=float)
            cluster_weights = np.zeros(self.k, dtype=float)
            for bid, cluster_id in bin_assignments.items():
                weight = self.bin_counts[bid]
                weight_sums[cluster_id] += self.bin_means[bid] * weight
                cluster_weights[cluster_id] += weight

            nonzero_clusters = cluster_weights > 0
            updated_seeds = weight_sums[nonzero_clusters] / cluster_weights[nonzero_clusters, None]
            self.seeds[nonzero_clusters] = updated_seeds

            # Monitor mean squared centroid displacement as convergence criterion.
            shift = np.mean(np.sum((old_seeds - self.seeds)**2, axis=1))

            history.append({"iteration": t, "shift": shift})
            self.iterations = t + 1
            if shift < self.threshold:
                break

        # Expand bin-level assignments back to point-level labels.
        final_assignments = np.zeros(P.shape[0], dtype=int)
        for bid, c in bin_assignments.items():
            final_assignments[self.bins[bid]] = c

        return final_assignments, self.seeds, history

    def get_metrics(self, P, assignments):
        """Compute distortion and computational-effort metrics for QBCA.

        Returns:
            avg_distortion: Mean squared distance from points to assigned centroids.
            avg_dist_comp: Mean number of distance evaluations per point per iteration.
        """
        # Distortion is the within-cluster sum of squared errors normalized by N.
        distortion = 0.0
        for h in range(self.k):
            cluster_pts = P[assignments == h]
            if len(cluster_pts) > 0:
                # Compute squared Euclidean distances to centroid `h`.
                squared_distances = np.sum((cluster_pts - self.seeds[h])**2, axis=1)
                distortion += np.sum(squared_distances)

        avg_distortion = distortion / len(P)

        # Normalize total distance evaluations by points and iterations.
        if self.iterations > 0 and len(P) > 0:
            avg_dist_comp = self.total_dist_calls / (len(P) * self.iterations)
        else:
            avg_dist_comp = 0.0

        return avg_distortion, avg_dist_comp
