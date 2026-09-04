import math

import numpy as np
import open3d as o3d
from numba import njit


@njit
def _grow(seeds, nbr, normals, angle_deg, smooth, angle_ok, curv_ok, resid_ok, min_size, max_size):
    """
    Breadth-first region growing over precomputed neighbour lists and tests.
    nbr[i] holds point i's k nearest neighbours, itself first. angle_ok and
    resid_ok are per (point, neighbour slot); curv_ok is per point. When
    smooth is False the angle test compares against the region's first seed
    instead, so it is evaluated here from the normals.
    Returns (labels, position, n_clusters): labels are 1-based with 0 for
    unclustered, position is each point's place in the order its region grew.
    """
    n = nbr.shape[0]
    k = nbr.shape[1]
    processed = np.zeros(n, dtype=np.bool_)
    labels = np.zeros(n, dtype=np.int64)
    position = np.zeros(n, dtype=np.int64)
    queue = np.empty(n, dtype=np.int64)
    region = np.empty(n, dtype=np.int64)
    n_clusters = 0
    for seed in seeds:
        if processed[seed]:
            continue
        queue[0] = seed
        region[0] = seed
        head = 0
        tail = 1
        size = 1
        processed[seed] = True
        while head < tail:
            curr = queue[head]
            head += 1
            for slot in range(1, k):
                j = nbr[curr, slot]
                if processed[j]:
                    continue
                if smooth:
                    if not angle_ok[curr, slot]:
                        continue
                else:
                    dot = abs(normals[seed, 0] * normals[j, 0]
                              + normals[seed, 1] * normals[j, 1]
                              + normals[seed, 2] * normals[j, 2])
                    if math.acos(min(dot, 1.0)) * 180.0 / math.pi > angle_deg:
                        continue
                region[size] = j
                size += 1
                processed[j] = True
                if curv_ok[j] and resid_ok[curr, slot]:
                    queue[tail] = j
                    tail += 1
        if min_size <= size <= max_size:
            n_clusters += 1
            for r in range(size):
                labels[region[r]] = n_clusters
                position[region[r]] = r
    return labels, position, n_clusters


def _row_dot(a, b):
    """Row-wise dot product of (..., 3) arrays, summed in the order np.dot uses."""
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1] + a[..., 2] * b[..., 2]


class RegionGrowing:
    def __init__(self):
        self.pcd = None
        self.normals = None
        self.curvatures = None
        self.k_neighbors = 30
        self.TAngle = 15.0  # degrees
        self.curvatureThreshold = 0.1
        self.residualThreshold = 0.05
        self.smoothMode = True
        self.useCurvatureTest = True
        self.useResidualTest = True
        self.minClusterSize = 100
        self.maxClusterSize = 100000
        self.Clusters = []
        self.pcd_tree = None
        self.NPt = 0
        self.neighbors = None

    def SetDataThresholds(self, pcd, angle_deg=15.0, curv_thresh=0.1, resid_thresh=0.05, k=30):
        self.pcd = pcd
        self.NPt = len(pcd.points)
        self.k_neighbors = k
        self.TAngle = angle_deg
        self.curvatureThreshold = curv_thresh
        self.residualThreshold = resid_thresh
        self.pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    def _knn(self):
        """
        k nearest neighbours of every point, itself first, as an (N, k) int
        array. The queries stay on the Open3D tree so the neighbour sets and
        their order are the ones the growing has always used.
        """
        k = min(self.k_neighbors, self.NPt)
        idx = np.empty((self.NPt, k), dtype=np.int64)
        points = self.pcd.points
        search = self.pcd_tree.search_knn_vector_3d
        for i in range(self.NPt):
            found = search(points[i], k)[1]
            m = len(found)
            idx[i, :m] = found
            if m < k:
                idx[i, m:] = i
        return idx

    def compute_curvatures(self, chunk=50000):
        """
        Surface variation of every point: smallest eigenvalue of the
        neighbourhood covariance over the sum of the eigenvalues. The
        covariance is formed the way np.cov forms it so the values match.
        """
        points = np.asarray(self.pcd.points)
        if self.neighbors is None:
            self.neighbors = self._knn()
        k = self.neighbors.shape[1]
        curvatures = np.zeros(self.NPt)
        if k < 2:
            return curvatures
        for start in range(0, self.NPt, chunk):
            stop = min(start + chunk, self.NPt)
            X = points[self.neighbors[start:stop]].transpose(0, 2, 1)
            X = X - X.mean(axis=2, keepdims=True)
            cov = (X @ X.transpose(0, 2, 1)) * (1.0 / (k - 1))
            eig = np.sort(np.abs(np.linalg.eigvalsh(cov)), axis=1)
            total = eig.sum(axis=1)
            total[total == 0] = 1.0
            curvatures[start:stop] = eig[:, 0] / total
        return curvatures

    def angle_between_normals(self, n1, n2):
        dot = np.clip(np.abs(np.dot(n1, n2)), -1.0, 1.0)
        return math.acos(dot) * 180.0 / math.pi

    def point_to_plane_residual(self, pt, pt_seed, normal_seed):
        diff = pt_seed - pt
        return np.abs(np.dot(normal_seed, diff))

    def RGKnn(self, chunk=50000):
        if len(self.pcd.normals) < self.NPt:
            self.pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=self.k_neighbors))
        self.normals = np.ascontiguousarray(np.asarray(self.pcd.normals), dtype=np.float64)
        points = np.ascontiguousarray(np.asarray(self.pcd.points), dtype=np.float64)
        self.Clusters = []
        if self.NPt == 0:
            self.curvatures = np.zeros(0)
            return
        self.neighbors = self._knn()
        self.curvatures = self.compute_curvatures(chunk)
        nbr = self.neighbors
        k = nbr.shape[1]

        # The tests between a point and each of its neighbours, evaluated once
        # with the same expressions validate_point used per pair.
        angle_ok = np.ones((self.NPt, k), dtype=np.bool_)
        resid_ok = np.ones((self.NPt, k), dtype=np.bool_)
        for start in range(0, self.NPt, chunk):
            stop = min(start + chunk, self.NPt)
            nb = nbr[start:stop]
            normals = self.normals[start:stop, None, :]
            if self.smoothMode:
                dot = np.clip(np.abs(_row_dot(normals, self.normals[nb])), -1.0, 1.0)
                angle_ok[start:stop] = ~(np.arccos(dot) * 180.0 / math.pi > self.TAngle)
            if self.useResidualTest:
                diff = points[start:stop, None, :] - points[nb]
                resid_ok[start:stop] = ~(np.abs(_row_dot(normals, diff)) > self.residualThreshold)
        if self.useCurvatureTest:
            curv_ok = ~(self.curvatures > self.curvatureThreshold)
        else:
            curv_ok = np.ones(self.NPt, dtype=np.bool_)

        # Seeds in order of increasing curvature, ties by index, as before.
        seeds = np.lexsort((np.arange(self.NPt), self.curvatures))
        labels, position, n_clusters = _grow(seeds, nbr, self.normals, float(self.TAngle), self.smoothMode,
                                             angle_ok, curv_ok, resid_ok, self.minClusterSize, self.maxClusterSize)
        # One index array per cluster, points in the order the region grew.
        order = np.lexsort((position, labels))
        bounds = np.searchsorted(labels[order], np.arange(1, n_clusters + 2))
        self.Clusters = [order[bounds[c]:bounds[c + 1]] for c in range(n_clusters)]

    def ReLabeles(self):
        labels = np.zeros(self.NPt)
        for i, cluster in enumerate(self.Clusters):
            labels[cluster] = i + 1
        return labels
