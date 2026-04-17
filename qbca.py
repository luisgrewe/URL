"""Implementation of Image Segmentation using QBCA."""

import numpy as np
import time
from collections import defaultdict


class QBCA:
    """Quantization-Based Clustering Algorithm (QBCA) with Hierarchical Structure."""

    def __init__(self, k, threshold=0.0001, max_iter=50):
        """Initialize a QBCA instance with clustering parameters."""
        self.k = k
        self.threshold = threshold
        self.max_iter = max_iter
        self.seeds = None
        self.bins = {}
        self.bin_bounds = {}
        self.parents = {}

        # Metric tracking
        self.init_time = 0
        self.total_dist_calls = 0
        self.iterations = 0

    def quantization(self, P):
        """Fully Vectorized Quantization & Hierarchical Construction."""
        n, m = P.shape
        self.rho = int(np.floor(np.log(n) / np.log(m))) if m > 1 else int(np.sqrt(n))

        p_min, p_max = P.min(axis=0), P.max(axis=0)
        bin_widths = (p_max - p_min) / self.rho
        bin_widths[bin_widths == 0] = 1.0

        # Vectorized Bin Assignment
        xi = np.floor((P - p_min) / bin_widths).astype(int)
        xi = np.clip(xi, 0, self.rho - 1)

        # Vectorized Linearization
        powers = self.rho ** np.arange(m - 1, -1, -1)
        bin_ids = np.dot(xi, powers)

        # Fast Grouping using argsort
        sort_idx = np.argsort(bin_ids)
        sorted_bin_ids = bin_ids[sort_idx]

        # Find where the bin IDs change to split the array
        _, unique_indices = np.unique(sorted_bin_ids, return_index=True)
        split_indices = unique_indices[1:]
        indices_per_bin = np.split(sort_idx, split_indices)
        unique_bin_ids = sorted_bin_ids[unique_indices]

        self.bins = defaultdict(list)
        for bid, indices in zip(unique_bin_ids, indices_per_bin):
            self.bins[bid] = indices.tolist()
            # Child shrinking bounds computed instantly
            pts = P[indices]
            self.bin_bounds[bid] = (pts.min(axis=0), pts.max(axis=0))

        # Hierarchical Construction
        self.parents = defaultdict(lambda: {'children': set(), 'bounds': [None, None]})
        parent_powers = (self.rho // 2 + 1) ** np.arange(m - 1, -1, -1)

        for bid in unique_bin_ids:
            # Reconstruct xi coordinates
            temp_id = bid
            xi_coords = []
            for p in powers:
                xi_coords.append(temp_id // p)
                temp_id %= p

            parent_xi = np.array(xi_coords) // 2
            parent_id = np.dot(parent_xi, parent_powers)
            self.parents[parent_id]['children'].add(bid)

        # Shrinking process for parents
        for _pid, data in self.parents.items():
            child_mins = [self.bin_bounds[cid][0] for cid in data['children']]
            child_maxes = [self.bin_bounds[cid][1] for cid in data['children']]
            data['bounds'] = (np.min(child_mins, axis=0), np.max(child_maxes, axis=0))

    def _get_candidates(self, seed_indices, bounds):
        """Determine seed candidates based on hierarchical bounds and distance bounds."""
        b_min, b_max = bounds

        # Minimum distance lower bound
        def d_min(s):
            self.total_dist_calls += 1
            closest = np.where(s < b_min, b_min, np.where(s > b_max, b_max, s))
            return np.linalg.norm(s - closest)

        # Maximum distance upper bound
        def d_max(s):
            self.total_dist_calls += 1
            furthest = np.where(s >= (b_min + b_max) / 2, b_min, b_max)
            return np.linalg.norm(s - furthest)

        # Find the min of max distances
        d_star_max = min(d_max(self.seeds[i]) for i in seed_indices)

        # Lemma 1 filter: Reject seeds further than d*
        return [i for i in seed_indices if d_min(self.seeds[i]) <= d_star_max]

    def cci(self, P):
        """Cluster center initialization with density-based seeding."""
        # Count points in bins and sort by density
        bin_counts = {bid: len(indices) for bid, indices in self.bins.items()}
        # Use a list of (bin_id, count) to make tracking easier
        sorted_bins = sorted(bin_counts.items(), key=lambda x: x[1], reverse=True)

        seed_list = []
        chosen_bins = set()

        # Find local density peaks
        for bid, _count in sorted_bins:
            if len(seed_list) < self.k:
                seed_list.append(P[self.bins[bid]].mean(axis=0))
                chosen_bins.add(bid)

        # If the number of peak bins is less than k, select from remaining bins
        if len(seed_list) < self.k:
            remaining_bins = [bid for bid, count in sorted_bins if bid not in chosen_bins]
            for bid in remaining_bins:
                if len(seed_list) < self.k:
                    seed_list.append(P[self.bins[bid]].mean(axis=0))
                    chosen_bins.add(bid)

        # If k is still larger than non-empty bins, pad with random noise
        # This prevents the IndexError when k > number of non-empty bins
        while len(seed_list) < self.k:
            random_idx = np.random.randint(0, P.shape[0])
            seed_list.append(P[random_idx])

        self.seeds = np.array(seed_list)

    def cca(self, P):
        """Cluster center assignment with early pruning."""
        new_assignments = np.zeros(P.shape[0], dtype=int)
        all_seed_indices = list(range(self.k))

        for _pid, pdata in self.parents.items():
            # Parent level pruning
            parent_candidates = self._get_candidates(all_seed_indices, pdata['bounds'])

            for bid in pdata['children']:
                indices = self.bins[bid]
                # Child level inheritance
                candidates = self._get_candidates(parent_candidates, self.bin_bounds[bid])

                if len(candidates) == 1:
                    # EARLY PRUNING: No point-level distance math needed
                    new_assignments[indices] = candidates[0]
                else:
                    # Fallback to point-level assignment
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

        # Inter-cluster distance
        inter_dist = float('inf')
        for i in range(self.k):
            for j in range(i + 1, self.k):
                dist = np.linalg.norm(self.seeds[i] - self.seeds[j])
                if dist < inter_dist:
                    inter_dist = dist

        # Intra-cluster diameter (Fast Approximation)
        max_diam = 0
        for i in range(self.k):
            # Boolean indexing is faster
            pts = P[assignments == i]
            if len(pts) == 0:
                continue

            # Use bounding box diagonal as diameter estimate instead of testing all points
            c_min, c_max = pts.min(axis=0), pts.max(axis=0)
            diam = np.linalg.norm(c_max - c_min)
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

            for h in range(self.k):
                cluster_pts = P[assignments == h]
                if len(cluster_pts) > 0:
                    self.seeds[h] = cluster_pts.mean(axis=0)

            # Calculates the shift for all centroids instantly
            shift = np.mean(np.sum((old_seeds - self.seeds)**2, axis=1))

            history.append({"iteration": t, "shift": shift})
            self.iterations = t + 1
            if shift < self.threshold:
                break

        return assignments, self.seeds, history

    def get_metrics(self, P, assignments):
        """Calculate evaluation metrics for the QBCA clustering performance.

        Returns:
            avg_distortion: The average squared distance from points to their cluster centers.
            avg_dist_comp: The average number of distance computations per point per iteration.
        """
        # Calculate Average Distortion (phi)
        distortion = 0.0
        for h in range(self.k):
            cluster_pts = P[assignments == h]
            if len(cluster_pts) > 0:
                # Squared Euclidean distance from points to their assigned centroid
                squared_distances = np.sum((cluster_pts - self.seeds[h])**2, axis=1)
                distortion += np.sum(squared_distances)

        avg_distortion = distortion / len(P)

        # Calculate Average Distance Computations
        if self.iterations > 0 and len(P) > 0:
            avg_dist_comp = self.total_dist_calls / (len(P) * self.iterations)
        else:
            avg_dist_comp = 0.0

        return avg_distortion, avg_dist_comp
