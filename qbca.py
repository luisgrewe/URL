import numpy as np
import time
from collections import defaultdict

class QBCA:
    """
    Quantization-Based Clustering Algorithm (QBCA) 
    Reference: Yu & Wong (2010), Pattern Recognition 43
    """
    
    def __init__(self, k, threshold=0.0001, max_iter=30):
        self.k = k
        self.threshold = threshold
        self.max_iter = max_iter
        self.seeds = None
        self.bins = {} 
        self.bin_bounds = {} 
        # Metric Tracking
        self.init_time = 0
        self.total_dist_calls = 0
        self.iterations = 0

    def quantization(self, P):
        """Step 1: The Quantization Process [cite: 218]"""
        n, m = P.shape
        # Exact rho from Equation 19 [cite: 220]
        self.rho = int(np.floor(np.log(n) / np.log(m))) if m > 1 else int(np.sqrt(n))
        
        p_min = P.min(axis=0)
        p_max = P.max(axis=0)
        # Size of histogram bins per dimension (Equation 20) [cite: 221]
        bin_widths = (p_max - p_min) / self.rho 
        bin_widths[bin_widths == 0] = 1.0
        
        # Quantization function (Equation 21) 
        self.bins = defaultdict(list)
        for i, point in enumerate(P):
            # Calculate linearized index (Equation 21) [cite: 231]
            xi = np.floor((point - p_min) / bin_widths).astype(int)
            xi = np.clip(xi, 0, self.rho - 1)
            # Linearization of the m-dimensional histogram [cite: 231]
            bin_id = 0
            for l in range(m):
                bin_id += xi[l] * (self.rho ** (m - 1 - l))
            self.bins[bin_id].append(i)

        # Shrinking Process (Section 5) [cite: 428, 471]
        # Tighten the boundaries to the actual points in the bin [cite: 475]
        self.bin_bounds = {}
        for bid, indices in self.bins.items():
            pts = P[indices]
            self.bin_bounds[bid] = (pts.min(axis=0), pts.max(axis=0))
        
        return self.rho

    def _dist_min(self, seed, bid):
        """Definition 2: Minimum distance (Equation 10) [cite: 147]"""
        b_min, b_max = self.bin_bounds[bid]
        # Find closest point in the shrunken bin [cite: 174]
        closest_point = np.where(seed < b_min, b_min, np.where(seed > b_max, b_max, seed))
        return np.linalg.norm(seed - closest_point)

    def _dist_max(self, seed, bid):
        """Definition 3: Maximum distance (Equation 12) [cite: 176]"""
        b_min, b_max = self.bin_bounds[bid]
        # Find furthest corner in the shrunken bin [cite: 177, 181]
        furthest_point = np.where(seed >= (b_min + b_max) / 2, b_min, b_max)
        return np.linalg.norm(seed - furthest_point)

    def cci(self, P):
        """Step 2: Cluster Center Initialization (CCI) [cite: 257]"""
        # Count points in each bin [cite: 261]
        bin_counts = {bid: len(indices) for bid, indices in self.bins.items()}
        # Paper utilizes a Max-heap to process bins by density [cite: 261, 262]
        sorted_bins = sorted(bin_counts.keys(), key=lambda x: bin_counts[x], reverse=True)
        
        seed_list = []
        for bid in sorted_bins:
            if len(seed_list) < self.k:
                # The center of the cluster in dense bins is viewed as a seed [cite: 306]
                seed_list.append(P[self.bins[bid]].mean(axis=0))
        
        self.seeds = np.array(seed_list)

    def cca(self, P):
        """Step 3: Cluster Center Assignment (CCA) [cite: 353]"""
        new_assignments = np.zeros(P.shape[0], dtype=int)
        
        for bid, indices in self.bins.items():
            # Find min of all max distances (Definition 4) [cite: 192]
            d_maxes = [self._dist_max(s, bid) for s in self.seeds]
            d_star_max = min(d_maxes)
            
            # Lemma 1: Pruning seeds that cannot be the closest [cite: 188]
            candidates = [i for i, s in enumerate(self.seeds) 
                          if self._dist_min(s, bid) <= d_star_max]
            
            if len(candidates) == 1:
                # Direct assignment without distance computation [cite: 380, 388]
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
        """Calculates exact paper metrics [cite: 563, 566]"""
        # Average Distortion (Equation 34) [cite: 567]
        total_error = 0
        for h in range(self.k):
            cluster_pts = P[assignments == h]
            if len(cluster_pts) > 0:
                total_error += np.sum((cluster_pts - self.seeds[h])**2)
        avg_distortion = total_error / P.shape[0]
        
        # Average number of distance computations per iteration (Section 7.1) [cite: 563]
        avg_dist_comp = self.total_dist_calls / (P.shape[0] * self.iterations) if self.iterations > 0 else 0
        
        return avg_distortion, avg_dist_comp

    def fit(self, P):
        """Execution loop for QBCA [cite: 256]"""
        self.total_dist_calls = 0
        
        # Initialization Stage [cite: 247, 632]
        start_init = time.time()
        self.quantization(P)
        self.cci(P)
        self.init_time = time.time() - start_init
        
        history = []
        for t in range(self.max_iter):
            old_seeds = self.seeds.copy()
            assignments = self.cca(P)
            
            # Recompute centers (Equation 8) [cite: 100, 252]
            for h in range(self.k):
                cluster_pts = P[assignments == h]
                if len(cluster_pts) > 0:
                    self.seeds[h] = cluster_pts.mean(axis=0)
            
            # Calculate gap delta (Equations 17 & 18) [cite: 212]
            shift = np.mean([np.sum((old_seeds[h] - self.seeds[h])**2) for h in range(self.k)])
            history.append({'iteration': t, 'shift': shift})
            self.iterations = t + 1
            
            if shift < self.threshold: # Termination threshold [cite: 392]
                break
                
        return assignments, self.seeds, history