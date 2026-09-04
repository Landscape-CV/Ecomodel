"""
Helpers that belong to the Ecomodel tile pipeline rather than to TreeQSM:
terrain rasterisation and subtraction, and the segment bend splitting used
when trees are separated.
"""
import numpy as np
import open3d as o3d
from numba import jit
from scipy.interpolate import griddata


def check_for_bends(segment_cloud,num_test_regions = 5,threshold = .2):
    """
    Check for bends in a point cloud segment by analyzing the angles between segments.
    
    Args:
        segment_cloud: Point cloud data as a numpy array of shape (N, 3).
        num_test_regions: Number of regions to test for bends.
        
    Returns:
        Boolean indicating if bends were detected.
    """
    segsize = len(segment_cloud)//num_test_regions
    if segsize <20:
        return False
    
    
    

    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(segment_cloud)
    try:
        obb = pcd.get_oriented_bounding_box()
    except:
        return False


    center = obb.center


    bend = True
    prev_dims = np.sort(obb.extent)
    for i in range(num_test_regions):
        
        # next_seg = segment_cloud[i*segsize:(i+1)*segsize]
        # pcd.points = o3d.utility.Vector3dVector(next_seg)
        next_seg = segment_cloud[:(i+1)*segsize]
        pcd.points = o3d.utility.Vector3dVector(next_seg)
        try:
            obb = pcd.get_oriented_bounding_box()
        except:
            pass

        # try:
        #     obb = pcd.get_oriented_bounding_box()

        # except:
        #     return False
        max_bound = obb.get_max_bound()
        min_bound = obb.get_min_bound()
        bound_range = max_bound - min_bound
        if np.all(center < max_bound-bound_range*.2) and np.all(center > min_bound+bound_range*.2):
            bend = False

        dims = np.sort(obb.extent)
        # if i>0:
        both_dims = np.array([dims,prev_dims])
        min_dims = np.min(both_dims,axis=0)
        max_dims = np.max(both_dims,axis=0)
        if min_dims[0]*2<max_dims[0] or (min_dims[0]<max_dims[0] and min_dims[1]*2<max_dims[1]):
            return True
            
        # prev_dims = dims
        

    return bend
    

def split_segments(segment_cloud, num_test_regions = 5, angle_threshold = 60):
    """
    Find bends in a point cloud based on the distance between points.
    
    Args:
        cloud: Point cloud data as a numpy array of shape (N, 3).
        num_test_regions: Number of regions to test for bends.
        
    Returns:
        Array indicating if point is in new segment
    """

    
    segs = np.zeros(len(segment_cloud),dtype=int)

    
    if not check_for_bends(segment_cloud,num_test_regions):
        return segs

    segsize = len(segment_cloud)//num_test_regions
    initial_seg = segment_cloud[:segsize]
    last_seg = segment_cloud[1*segsize:2*segsize]
    a = np.mean(initial_seg, axis=0)
    b = np.mean(last_seg, axis=0)
    initial_vec = (b-a)/(np.linalg.vector_norm(b-a))
    last_vec = initial_vec

    # full_pcd = o3d.geometry.PointCloud()
    # full_pcd.points = o3d.utility.Vector3dVector(segment_cloud)


    for i in range(2,num_test_regions):
        
        next_seg = segment_cloud[i*segsize:(i+1)*segsize]

        
        if len(next_seg) < 10: 
            break
        new_vec1 = (next_seg[-1]-next_seg[0])#/(np.linalg.vector_norm(c-b))
        
        
        denom = np.linalg.norm(last_vec) * np.linalg.norm(new_vec1)
        if denom < 1e-9:
            last_vec = new_vec1
            last_seg = next_seg
            continue
        angle = np.rad2deg(np.arccos(np.clip(np.dot(last_vec, new_vec1) / denom, -1.0, 1.0)))
        if angle > angle_threshold:
            segs[i*segsize:] = 1
            return segs
            
        last_vec = new_vec1
        last_seg = next_seg

    return segs


def get_axis(point_cloud):

    point_cloud = np.random.permutation(point_cloud)[:15]
    # mcd = FastMCD()
    # try:
    #     covariance = mcd.calculate_covariance(point_cloud)
    # except:
    # try:
    #     mcd = DetMCD()
    #     covariance = mcd.calculate_covariance(point_cloud)
    # except Exception as e:
    #     print("Failed to find covariance")
    #     raise e
            

    
    # mean = mcd.location_
    U, S, Vt = np.linalg.svd(point_cloud, full_matrices=False)
    first_pc = Vt[0, :] 
    return first_pc


@jit(nopython=True, parallel=True)
def rasterize_cloud(point_cloud, resolution=0.1):
    """
    Rasterize a point cloud into a 2D grid based on the highest z-value in each cell.
    
    Args:
        point_cloud: Point cloud data as a numpy array of shape (N, 3).
        resolution: Size of each grid cell.
        
    Returns:
        2D numpy array representing the rasterized grid.
    """
    x_min, x_max = np.min(point_cloud[:, 0]), np.max(point_cloud[:, 0])
    y_min, y_max = np.min(point_cloud[:, 1]), np.max(point_cloud[:, 1])
    
    x_bins = np.arange(x_min, x_max + resolution, resolution)
    y_bins = np.arange(y_min, y_max + resolution, resolution)
    
    raster_grid = np.full((len(x_bins)-1, len(y_bins)-1), np.nan,dtype =np.float64)
    
    x_indices = np.digitize(point_cloud[:, 0], x_bins) - 1
    y_indices = np.digitize(point_cloud[:, 1], y_bins) - 1
    
    for i in range(len(point_cloud)):
        x_idx = x_indices[i]
        y_idx = y_indices[i]
        if 0 <= x_idx < raster_grid.shape[0] and 0 <= y_idx < raster_grid.shape[1]:
            raster_grid[x_idx, y_idx] = max(raster_grid[x_idx, y_idx], point_cloud[i, 2]) if not np.isnan(raster_grid[x_idx, y_idx]) else point_cloud[i, 2]
    
    # raster_grid[raster_grid == -np.inf] = np.nan
    
    return raster_grid

def fill_raster_gaps(raster_grid):
    """
    Fill gaps (NaN values) in a raster grid using nearest neighbor interpolation.
    
    Args:
        raster_grid: 2D numpy array representing the rasterized grid with NaN values.
        
    Returns:
        2D numpy array with NaN values filled.
    """
    x = np.arange(raster_grid.shape[1])
    y = np.arange(raster_grid.shape[0])
    xx, yy = np.meshgrid(x, y)
    
    valid_mask = ~np.isnan(raster_grid)
    filled_grid = griddata(
        (xx[valid_mask], yy[valid_mask]),
        raster_grid[valid_mask],
        (xx, yy),
        method='nearest'
    )
    
    return filled_grid


def subtract_terrain(point_cloud, terrain_raster, grid_size=0.1):
    """
    Subtract terrain height from point cloud to get normalized heights.

    Args:
        point_cloud:    Point cloud as (N, 3) numpy array.
        terrain_raster: 2-D numpy array of terrain heights, shape (nx, ny).
        grid_size:      float representing the

    Returns:
        (N, 3) array with the same XY coordinates and terrain-subtracted Z.
    """
    x_min, x_max = np.min(point_cloud[:, 0]), np.max(point_cloud[:, 0])
    y_min, y_max = np.min(point_cloud[:, 1]), np.max(point_cloud[:, 1])

    nx, ny = terrain_raster.shape

    # Build one set of bin edges per axis using that axis's raster dimension.
    # np.digitize returns values in [0, len(bins)], so the maximum possible
    # index equals len(bins) = nx-1 (or ny-1), which is always a valid index.
    x_bins = np.linspace(x_min, x_max, nx - 1)
    y_bins = np.linspace(y_min, y_max, ny - 1)

    x_indices = np.digitize(point_cloud[:, 0], x_bins)
    y_indices = np.digitize(point_cloud[:, 1], y_bins)

    # Clip to valid range — floating-point edge cases can push a boundary
    # point one index beyond the raster extent.
    x_indices = np.clip(x_indices, 0, nx - 1)
    y_indices = np.clip(y_indices, 0, ny - 1)

    normalized_heights = point_cloud[:, 2] - terrain_raster[x_indices, y_indices]

    return np.column_stack((point_cloud[:, :2], normalized_heights))
