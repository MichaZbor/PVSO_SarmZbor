import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import copy  # To avoid modifying original point clouds


# --- Function 1: Remove Planar Points using RANSAC ---
# --- MODIFIED FUNCTION ---
def remove_planar_points_ransac(pcd,
                                distance_threshold=0.01,
                                ransac_n=3,
                                num_iterations=1000,
                                verbose=True):
    """
    Segments the largest plane using RANSAC and returns the points *not* belonging to that plane (outliers).

    Args:
        pcd (open3d.geometry.PointCloud): The input point cloud.
        distance_threshold (float): Max distance a point can be from the plane
                                     model to be considered an inlier. Adjust
                                     based on your point cloud's scale/noise.
        ransac_n (int): Number of points randomly sampled to estimate a plane.
        num_iterations (int): Number of iterations RANSAC will run.
        verbose (bool): If True, prints the number of points removed (inliers).

    Returns:
        tuple: A tuple containing:
            - open3d.geometry.PointCloud: Point cloud *without* the largest plane
                                          (original outlier points).
            - open3d.geometry.PointCloud: Point cloud containing *only* the points
                                          belonging to the largest plane (original
                                          inlier points).
            - list: Indices of the plane points (inliers) in the original pcd.
            - numpy.ndarray: The equation of the detected plane [a, b, c, d].
                             Returns None if no plane is found or pcd is empty.
    """
    if not pcd.has_points():
        print("Warning: Input point cloud is empty.")
        # Return empty point clouds and None for plane/indices
        return o3d.geometry.PointCloud(), o3d.geometry.PointCloud(), [], None

    try:
        plane_model, inlier_indices = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )

        if not inlier_indices:
            print("Warning: RANSAC did not find any plane inliers.")
            # Return original pcd as kept points and empty removed points
            kept_points_cloud = copy.deepcopy(pcd)
            removed_plane_cloud = o3d.geometry.PointCloud()
            return kept_points_cloud, removed_plane_cloud, [], None

        # [a, b, c, d] plane equation such that ax + by + cz + d = 0
        if verbose:
            print(f"RANSAC found plane: {plane_model}")
            print(f"Identified {len(inlier_indices)} planar points (inliers) to be removed.")

        # Create point clouds for the points kept (outliers) and removed (inliers)
        # ---MODIFICATION: kept_points_cloud is now the outliers---
        kept_points_cloud = pcd.select_by_index(inlier_indices, invert=True)
        # ---MODIFICATION: removed_plane_cloud is now the inliers---
        removed_plane_cloud = pcd.select_by_index(inlier_indices)

        # Ensure colors are preserved if they exist
        # select_by_index preserves colors if the original pcd has them.

        return kept_points_cloud, removed_plane_cloud, inlier_indices, plane_model

    except Exception as e:
        print(f"An error occurred during RANSAC plane segmentation: {e}")
        # Return original pcd as kept points and empty removed points in case of error
        kept_points_cloud = copy.deepcopy(pcd)
        removed_plane_cloud = o3d.geometry.PointCloud()
        return kept_points_cloud, removed_plane_cloud, [], None


# --- Function 2: Cluster Point Cloud (Unchanged from previous version) ---
def cluster_point_cloud(pcd, method='dbscan', **kwargs):
    """
    Clusters the point cloud using the specified method (DBSCAN or K-Means).

    Args:
        pcd (open3d.geometry.PointCloud): The input point cloud.
        method (str): Clustering algorithm ('dbscan' or 'kmeans').
        **kwargs: Algorithm-specific parameters.
            For 'dbscan':
                eps (float): Density parameter (max dist between points).
                min_points (int): Minimum number of points to form a cluster.
                print_progress (bool): Whether to print progress.
            For 'kmeans':
                n_clusters (int): The number of clusters to form.
                kmeans_kwargs (dict): Additional arguments for sklearn.cluster.KMeans.

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: An array of integer labels for each point.
                             Noise points are labeled -1 in DBSCAN.
            - open3d.geometry.PointCloud: A *copy* of the input point cloud
                                          colored by cluster labels. Returns None
                                          if clustering fails or pcd is empty.
    """
    if not pcd.has_points():
        print("Warning: Input point cloud for clustering is empty.")
        return np.array([]), None

    labels = np.array([])
    colored_pcd = None

    try:
        if method.lower() == 'dbscan':
            eps = kwargs.get('eps', 0.05)
            min_points = kwargs.get('min_points', 10)
            print_progress = kwargs.get('print_progress', False)

            print(f"Running DBSCAN with eps={eps}, min_points={min_points}")
            labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=print_progress))

        elif method.lower() == 'kmeans':
            n_clusters = kwargs.get('n_clusters', 5)
            kmeans_kwargs = kwargs.get('kmeans_kwargs', {})
            if 'n_init' not in kmeans_kwargs:
                kmeans_kwargs['n_init'] = 'auto'

            print(f"Running K-Means with n_clusters={n_clusters}")
            points_np = np.asarray(pcd.points)
            kmeans = KMeans(n_clusters=n_clusters, **kmeans_kwargs)
            labels = kmeans.fit_predict(points_np)

        else:
            print(f"Error: Clustering method '{method}' not supported. Use 'dbscan' or 'kmeans'.")
            return np.array([]), None

        # --- Visualization ---
        max_label = labels.max()
        if max_label < 0 and method.lower() == 'dbscan':
            print("Warning: DBSCAN only found noise points (label -1).")
            colors = np.zeros((len(labels), 3))
        elif max_label < 0:
            print("Warning: K-Means returned unexpected negative labels.")
            colors = np.zeros((len(labels), 3))
        else:
            print(f"Clustering found {max_label + 1} clusters (plus noise for DBSCAN).")
            cmap = plt.get_cmap("tab20")
            unique_labels = np.unique(labels)
            num_unique_labels = len(unique_labels[unique_labels != -1])

            colors = np.zeros((len(labels), 3))
            # Handle case where num_unique_labels could be 0 if only noise is found
            if num_unique_labels > 0:
                color_map = {label: cmap(i / num_unique_labels)[:3]
                             for i, label in enumerate(unique_labels[unique_labels != -1])}
            else:
                color_map = {}  # No non-noise labels

            for i, label in enumerate(labels):
                if label == -1:
                    colors[i] = [0.5, 0.5, 0.5]  # Gray for noise
                else:
                    colors[i] = color_map.get(label, [0.0, 0.0, 0.0])  # Use mapped color or black

        colored_pcd = copy.deepcopy(pcd)
        colored_pcd.colors = o3d.utility.Vector3dVector(colors)

        return labels, colored_pcd

    except Exception as e:
        print(f"An error occurred during clustering with method '{method}': {e}")
        return np.array([]), None


if __name__ == "__main__":
    # --- Configuration ---
    pcd_file_path = "Clouds/output3611.pcd"
    try:
        original_pcd = o3d.io.read_point_cloud(pcd_file_path)
        if not original_pcd.has_points():
            raise ValueError("Loaded point cloud is empty.")
        print(f"Successfully loaded point cloud from {pcd_file_path}")

    except Exception as e:
        print(f"Error loading PCD file '{pcd_file_path}': {e}")
        raise e

    print("Original Point Cloud (with colors):")
    o3d.visualization.draw_geometries([original_pcd], window_name="Original Point Cloud")

    # --- 1. Remove planar points using RANSAC ---
    print("\n--- Running RANSAC to Remove Largest Plane ---")

    # Adjust distance_threshold based on your data's scale and noise level
    points_kept, points_removed, _, _ = remove_planar_points_ransac(
        original_pcd,
        distance_threshold=0.05,  # Example threshold, adjust!
        ransac_n=4,
        num_iterations=1000
    )

    print("\nVisualizing RANSAC Results (Removed Plane in Red):")
    geometries_ransac = []

    if points_kept.has_points():
        # --- MODIFICATION: Keep original colors ---
        # No need to call paint_uniform_color on points_kept
        print(f"Number of points kept (non-planar): {len(points_kept.points)}")
        geometries_ransac.append(points_kept)
    else:
        print("No non-planar points kept after RANSAC.")

    if geometries_ransac:
        # --- MODIFICATION: Updated window title ---
        o3d.visualization.draw_geometries(geometries_ransac, window_name="Kept Points")
    else:
        print("Nothing to visualize after RANSAC.")

    target_pcd_for_clustering = points_kept

    if not target_pcd_for_clustering.has_points():
        print("\nSkipping clustering because no non-planar points were kept after RANSAC.")
    else:
        print("\n--- Running Clustering Algorithms on Kept (Non-Planar) Points ---")

        # --- Clustering Method 1: DBSCAN ---
        print("\nRunning DBSCAN...")
        dbscan_labels, dbscan_colored_pcd = cluster_point_cloud(
            target_pcd_for_clustering,
            method='dbscan',
            eps=0.15,  # Example value, may need adjustment
            min_points=5,  # Example value, may need adjustment
            print_progress=True
        )

        if dbscan_colored_pcd:
            print("Visualizing DBSCAN results on kept points (Noise is Gray)...")
            o3d.visualization.draw_geometries([dbscan_colored_pcd], window_name="DBSCAN on Kept Points")
        else:
            print("DBSCAN clustering failed or produced no result to visualize.")

        # --- Clustering Method 2: K-Means ---
        print("\nRunning K-Means...")
        kmeans_labels, kmeans_colored_pcd = cluster_point_cloud(
            target_pcd_for_clustering,
            method='kmeans',
            n_clusters=2  # Example: Try to find 2 clusters in the non-planar points
        )

        if kmeans_colored_pcd:
            print("Visualizing K-Means results on kept points...")
            o3d.visualization.draw_geometries([kmeans_colored_pcd], window_name="K-Means on Kept Points")
        else:
            print("K-Means clustering failed or produced no result to visualize.")