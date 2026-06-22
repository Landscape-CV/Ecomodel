import numpy as np
from scipy.spatial import KDTree
from scipy.spatial import procrustes
from scipy.optimize import linear_sum_assignment
import argparse

class QSMEvaluator:
    def __init__(self):
        pass
        
    def _get_tree_instances(self, cyl_array):
        """ Group cylinders by tree_instance_id (column 8) """
        instances = {}
        for row in cyl_array:
            tid = int(row[8])
            if tid not in instances:
                instances[tid] = []
            instances[tid].append(row)
        for tid in instances:
            instances[tid] = np.array(instances[tid])
        return instances
        
    def _match_instances(self, gt_instances, pred_instances):
        """ Match predicted instances to GT instances based on centroid distance """
        gt_keys = list(gt_instances.keys())
        pred_keys = list(pred_instances.keys())
        
        if not gt_keys or not pred_keys:
            return []
            
        gt_centroids = []
        for k in gt_keys:
            cyls = gt_instances[k]
            # centroid of tree = mean of all cylinder starts
            gt_centroids.append(np.mean(cyls[:, :3], axis=0))
            
        pred_centroids = []
        for k in pred_keys:
            cyls = pred_instances[k]
            pred_centroids.append(np.mean(cyls[:, :3], axis=0))
            
        # Cost matrix: distance between centroids
        from scipy.spatial.distance import cdist
        cost_matrix = cdist(np.array(pred_centroids), np.array(gt_centroids))
        
        # Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matches = []
        for r, c in zip(row_ind, col_ind):
            # Only accept match if distance is reasonable (e.g. < 5 meters)
            if cost_matrix[r, c] < 5.0:
                matches.append((pred_keys[r], gt_keys[c]))
                
        return matches

    def _sample_points_on_cylinders(self, cyl_array, num_samples=5):
        """ Sample points along the centerlines for distance metrics """
        points = []
        radii = []
        for row in cyl_array:
            start = row[:3]
            axis = row[4:7]
            length = row[7]
            r = row[3]
            for t in np.linspace(0, 1, num_samples):
                p = start + t * length * axis
                points.append(p)
                radii.append(r)
        return np.array(points), np.array(radii)

    def evaluate_tree(self, gt_cyls, pred_cyls):
        """ Evaluate metrics for a single matched tree """
        metrics = {}
        
        # 1. Volume Error
        # Volume = pi * r^2 * h
        gt_vol = np.sum(np.pi * (gt_cyls[:, 3]**2) * gt_cyls[:, 7])
        pred_vol = np.sum(np.pi * (pred_cyls[:, 3]**2) * pred_cyls[:, 7])
        metrics['Volume_Error_Ratio'] = abs(pred_vol - gt_vol) / (gt_vol + 1e-6)
        
        # 2. Procrustes Shape Analysis
        # Extract centroids of cylinders as shape nodes
        gt_nodes = gt_cyls[:, :3]
        pred_nodes = pred_cyls[:, :3]
        
        # Procrustes requires equal number of points. We can sample or interpolate.
        # For simplicity, we just take N points by interpolating along the main trunk or random sampling.
        # A proper shape matching requires equal size point clouds.
        # We will sample 100 points from both
        def sample_n(pts, n=100):
            if len(pts) >= n:
                idx = np.random.choice(len(pts), n, replace=False)
                return pts[idx]
            else:
                idx = np.random.choice(len(pts), n, replace=True)
                return pts[idx]
                
        gt_shape = sample_n(gt_nodes)
        pred_shape = sample_n(pred_nodes)
        
        # Sort by Z to roughly align them topologically
        gt_shape = gt_shape[np.argsort(gt_shape[:, 2])]
        pred_shape = pred_shape[np.argsort(pred_shape[:, 2])]
        
        try:
            mtx1, mtx2, disparity = procrustes(gt_shape, pred_shape)
            metrics['Procrustes_Disparity'] = disparity
        except Exception as e:
            metrics['Procrustes_Disparity'] = float('nan')
            
        # 3 & 4 & 5. Centerline Spatial Deviation, Radius RMSE, Completeness, Precision
        gt_pts, gt_rads = self._sample_points_on_cylinders(gt_cyls)
        pred_pts, pred_rads = self._sample_points_on_cylinders(pred_cyls)
        
        if len(gt_pts) == 0 or len(pred_pts) == 0:
            return metrics
            
        gt_tree = KDTree(gt_pts)
        pred_tree = KDTree(pred_pts)
        
        # Pred -> GT (Precision & Spatial Deviation)
        dist_pred_to_gt, idx_pred_to_gt = gt_tree.query(pred_pts)
        metrics['Centerline_Mean_Dist_Pred_to_GT'] = np.mean(dist_pred_to_gt)
        metrics['Precision_Threshold_0.1m'] = np.mean(dist_pred_to_gt < 0.1)
        
        # GT -> Pred (Completeness & Radius RMSE)
        dist_gt_to_pred, idx_gt_to_pred = pred_tree.query(gt_pts)
        metrics['Centerline_Mean_Dist_GT_to_Pred'] = np.mean(dist_gt_to_pred)
        metrics['Completeness_Threshold_0.1m'] = np.mean(dist_gt_to_pred < 0.1)
        
        # Radius RMSE for matched points (distance < 0.2m)
        matched = dist_gt_to_pred < 0.2
        if np.any(matched):
            matched_pred_rads = pred_rads[idx_gt_to_pred[matched]]
            matched_gt_rads = gt_rads[matched]
            metrics['Radius_RMSE'] = np.sqrt(np.mean((matched_gt_rads - matched_pred_rads)**2))
        else:
            metrics['Radius_RMSE'] = float('nan')
            
        return metrics

    def run_evaluation(self, gt_txt_path, pred_txt_path):
        gt_array = np.loadtxt(gt_txt_path)
        pred_array = np.loadtxt(pred_txt_path)
        
        if len(gt_array.shape) == 1:
            gt_array = gt_array.reshape(1, -1)
        if len(pred_array.shape) == 1:
            pred_array = pred_array.reshape(1, -1)
            
        gt_instances = self._get_tree_instances(gt_array)
        pred_instances = self._get_tree_instances(pred_array)
        
        matches = self._match_instances(gt_instances, pred_instances)
        print(f"Found {len(matches)} matching tree instances between GT and Prediction.")
        
        all_metrics = []
        for pred_id, gt_id in matches:
            gt_cyls = gt_instances[gt_id]
            pred_cyls = pred_instances[pred_id]
            
            tree_metrics = self.evaluate_tree(gt_cyls, pred_cyls)
            tree_metrics['GT_Tree_ID'] = gt_id
            tree_metrics['Pred_Tree_ID'] = pred_id
            all_metrics.append(tree_metrics)
            
        # Print summary
        if not all_metrics:
            print("No metrics to display.")
            return
            
        print("\n=== Evaluation Summary ===")
        keys = [k for k in all_metrics[0].keys() if "ID" not in k]
        for k in keys:
            vals = [m[k] for m in all_metrics if not np.isnan(m[k])]
            if vals:
                print(f"{k}: {np.mean(vals):.4f}")
            else:
                print(f"{k}: N/A")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_path", type=str, required=True, help="Path to GT cylinders .txt")
    parser.add_argument("--pred_path", type=str, required=True, help="Path to Predicted cylinders .txt")
    args = parser.parse_args()
    
    evaluator = QSMEvaluator()
    evaluator.run_evaluation(args.gt_path, args.pred_path)
