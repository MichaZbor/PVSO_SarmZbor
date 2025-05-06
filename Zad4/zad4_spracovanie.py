from collections import Counter

import open3d as o3d
import numpy as np
from sklearn.cluster import KMeans
import copy


# Remove Planar Points using RANSAC
def remove_planar_points_ransac(pcd, distance_threshold=0.01, ransac_n=3, num_iterations=1000):
    """
    Segments the largest plane using RANSAC and returns the points *not* belonging to that plane (outliers).

    :parameter
      - pcd (open3d.geometry.PointCloud): The input point cloud.
      - distance_threshold (float): Max distance a point can be from the plane model to be considered an inlier. Adjust based on your point cloud's scale/noise.
      - ransac_n (int): Number of points randomly sampled to estimate a plane.
      - num_iterations (int): Number of iterations RANSAC will run.

    :returns:
      - open3d.geometry.PointCloud: Point cloud *without* the largest plane (original outlier points). Preserves original colors.
      - open3d.geometry.PointCloud: Point cloud containing *only* the points belonging to the largest plane (original inlier points). Preserves original colors.
    """

    try:
        plane_modelle, inlier_indices = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )

        if not inlier_indices:
            print("RANSAC did not find any plane inliers in this iteration.")
            kept_points_cloud = copy.deepcopy(pcd)
            removed_plane_cloud = o3d.geometry.PointCloud()
            return kept_points_cloud, removed_plane_cloud

        if plane_modelle is not None:
            print(f"RANSAC found plane: {plane_modelle}")
        print(f"Identified {len(inlier_indices)} planar points (inliers) to be removed.")

        kept_points_cloud = pcd.select_by_index(inlier_indices, invert=True)
        removed_plane_cloud = pcd.select_by_index(inlier_indices)

        return kept_points_cloud, removed_plane_cloud
    except Exception as e:
        print(f"An unexpected error occurred during RANSAC plane segmentation: {e}")
        kept_points_cloud = copy.deepcopy(pcd)
        removed_plane_cloud = o3d.geometry.PointCloud()
        return kept_points_cloud, removed_plane_cloud


def clean_point_cloud_for_clustering(pcd):
    """
    Removes non-finite points from a point cloud.

    :param: pcd (open3d.geometry.PointCloud): The input point cloud.
    :return: open3d.geometry.PointCloud: A new point cloud with only finite points. Returns None if the input is empty or has no finite points.
    """
    if not pcd or not pcd.has_points():
        print("Warning: Input point cloud for cleaning is empty or invalid.")
        return None

    points_np = np.asarray(pcd.points)
    finite_mask = np.all(np.isfinite(points_np), axis=1)
    num_original_points = len(points_np)
    num_finite_points = np.sum(finite_mask)

    if num_finite_points == 0:
        print("Warning: Point cloud contains no finite points after cleaning.")
        return None

    if num_finite_points < num_original_points:
        print(f"Warning: Removing {num_original_points - num_finite_points} non-finite points.")

    finite_indices = np.where(finite_mask)[0]
    cleaned_pcd = pcd.select_by_index(finite_indices)

    print(f"Cleaning resulted in {len(cleaned_pcd.points)} finite points.")
    return cleaned_pcd


def color_point_cloud_by_labels(pcd, labels):
    """
    Colors a point cloud based on cluster labels.

    :parameter
      - pcd (open3d.geometry.PointCloud): The input point cloud (should correspond to the labels).
      - labels (numpy.ndarray): An array of integer labels for each point in pcd. Noise points (-1) will be colored gray.

    :return: open3d.geometry.PointCloud: A *copy* of the input point cloud colored by cluster labels. Returns None if coloring fails or inputs are invalid.
    """
    if not pcd or not pcd.has_points() or labels is None or labels.size == 0:
        print("Warning: Cannot color point cloud. Invalid inputs provided.")
        return None

    if len(labels) != len(pcd.points):
        print(f"Error: Mismatch between number of labels ({len(labels)}) and points ({len(pcd.points)}). Cannot color.")
        return None

    colored_pcd = copy.deepcopy(pcd)
    max_label = labels.max()

    if max_label < 0:
        print(f"Warning: All labels are negative ({max_label}). Coloring all points gray.")
        colors = np.full((len(labels), 3), [0.5, 0.5, 0.5])
    else:
        # Generate colors using a simple scheme based on label index
        unique_non_noise_labels = np.unique(labels[labels != -1])
        num_unique_clusters = len(unique_non_noise_labels)
        colors = np.zeros((len(labels), 3))

        if num_unique_clusters > 0:
            # Create a mapping from unique label to a color index (0 to num_unique_clusters-1)
            label_to_color_idx = {label: i for i, label in enumerate(unique_non_noise_labels)}
            color_palette = [[(i * 100 % 255) / 255.0, (i * 150 % 255) / 255.0, (i * 200 % 255) / 255.0]
                             for i in range(
                    num_unique_clusters)]  # simple color palette generator based on index for max 51 colors
            color_palette = np.array(color_palette)

            for i, label in enumerate(labels):
                if label == -1:
                    colors[i] = [0.5, 0.5, 0.5]  # Gray for noise
                else:
                    color_idx = label_to_color_idx.get(label)
                    if color_idx is not None and color_idx < len(color_palette):
                        colors[i] = color_palette[color_idx]
                    else:
                        # Fallback for unexpected labels
                        colors[i] = [0.0, 0.0, 0.0]  # Black

        else:
            print("Warning: No non-noise clusters found, all points (if any) are noise or have unexpected labels")
            colors = np.full((len(labels), 3), [0.5, 0.5, 0.5])  # Default to gray

    try:
        colored_pcd.colors = o3d.utility.Vector3dVector(colors)
        return colored_pcd
    except Exception as e:
        print(f"Error setting colors on point cloud: {e}")
        raise e


def cluster_by_dbscan(pcd, eps=0.1, min_points=10, print_progress=False):
    """
    Clusters a point cloud using the DBSCAN algorithm.
    Handles non-finite points internally before clustering and colors the result.

    :parameter:
        - pcd (open3d.geometry.PointCloud): The input point cloud.
        - eps (float): Density parameter. CRITICAL TO TUNE.
        - min_points (int): Minimum number of points. CRITICAL TO TUNE.
        - print_progress (bool): Whether to print progress.

    :return: open3d.geometry.PointCloud: A *copy* of the cleaned point cloud colored by cluster labels.
                                          Returns None if clustering or coloring fails
    """

    cleaned_pcd = clean_point_cloud_for_clustering(pcd)

    print(f"Running DBSCAN with eps={eps}, min_points={min_points} on {len(cleaned_pcd.points)} points.")

    try:
        labels = np.array(cleaned_pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=print_progress))
        print(
            f"DBSCAN found {labels.max() + 1 if labels.max() >= 0 else 0} cluster(s) and {np.sum(labels == -1)} noise points.")

        if labels.size == 0:
            print("Warning: DBSCAN produced no labels. Cannot color point cloud.")
            return copy.deepcopy(cleaned_pcd)  # Return a copy

        return color_point_cloud_by_labels(cleaned_pcd, labels)

    except Exception as e:
        print(f"An error occurred during DBSCAN clustering: {e}")
        return None


def cluster_by_kmeans(pcd, n_clusters=3, kmeans_kwargs=None):
    """
    Performs K-Means segmentation on a point cloud.
    Handles non-finite points internally before clustering and colors the result.

    Args:
        pcd (open3d.geometry.PointCloud): The input point cloud.
        n_clusters (int): The number of clusters to form.
        kmeans_kwargs (dict): Additional arguments for sklearn.cluster.KMeans.

    :return: open3d.geometry.PointCloud: A *copy* of the cleaned point cloud colored by cluster labels. Returns None if clustering or coloring fails or no finite points exist.
    """
    cleaned_pcd = clean_point_cloud_for_clustering(pcd)

    points_for_clustering = np.asarray(cleaned_pcd.points)
    num_points = points_for_clustering.shape[0]

    if num_points < n_clusters or n_clusters == 0:
        print(f"Warning: Number of points ({num_points}) is less than n_clusters ({n_clusters}) or n_clusters is 0. Cannot run.")
        raise ValueError("Number of points is less than n_clusters or n_clusters is 0.")

    if kmeans_kwargs is None:
        kmeans_kwargs = {}

    # --- Fix for TypeError with n_init='auto' ---
    # Ensure n_init is an integer, replacing 'auto' with a numerical value if necessary
    if 'n_init' not in kmeans_kwargs or kmeans_kwargs['n_init'] == 'auto':
        # Use a default numerical value for n_init when 'auto' is specified or missing
        kmeans_kwargs['n_init'] = 10  # A common default value

    print(f"Running K-Means segmentation with n_clusters={n_clusters} on {num_points} points.")

    try:
        kmeans = KMeans(n_clusters=n_clusters, **kmeans_kwargs)
        if np.any(~np.isfinite(points_for_clustering)):
            raise ValueError("Internal Error: Non-finite values still present before K-Means fit.")
        labels = kmeans.fit_predict(points_for_clustering)
        print(f"K-Means segmentation found {n_clusters} cluster(s).")

        return color_point_cloud_by_labels(cleaned_pcd, labels)

    except Exception as e:
        print(f"An error occurred during K-Means segmentation: {e}")
        raise e


def filter_clusters_by_size(colored_pcd, min_points_per_cluster):
    """
    Filters a colored point cloud to keep only clusters (identified by unique colors)
    that have at least 'min_points_per_cluster'.

    Args:
        colored_pcd (open3d.geometry.PointCloud): The input point cloud, assumed to have
                                                  colors assigned per cluster.
        min_points_per_cluster (int): The minimum number of points a color group
                                      must have to be kept.

    Returns:
        open3d.geometry.PointCloud: A new point cloud containing only the points
                                    from clusters that meet the size requirement.
                                    Returns an empty point cloud if no clusters meet
                                    the criteria or if input is invalid.
    """

    if min_points_per_cluster <= 0:
        print("Filter_clusters: min_points_per_cluster should be positive. Returning original cloud.")
        return copy.deepcopy(colored_pcd)

    points = np.asarray(colored_pcd.points)
    colors = np.asarray(colored_pcd.colors)

    rounded_colors_tuples = [tuple(np.round(c, decimals=4)) for c in colors]
    color_counts = Counter(rounded_colors_tuples)

    print(f"Filter_clusters: Found {len(color_counts)} unique colors (clusters).")

    # Identify colors that meet the minimum point threshold
    colors_to_keep = {color for color, count in color_counts.items() if count >= min_points_per_cluster}

    if not colors_to_keep:
        print(f"Filter_clusters: No clusters found with at least {min_points_per_cluster} points.")
        return o3d.geometry.PointCloud()

    print(f"Filter_clusters: Keeping {len(colors_to_keep)} clusters that meet the size threshold of {min_points_per_cluster} points.")

    # Create a boolean mask for points to keep
    # This is more efficient than building a list of indices for very large clouds
    keep_mask = np.array([rounded_color_tuple in colors_to_keep for rounded_color_tuple in rounded_colors_tuples])

    # Select points and their original colors using the mask
    filtered_pcd = colored_pcd.select_by_index(np.where(keep_mask)[0])

    print(f"Filter_clusters: Original points: {len(points)}, Filtered points: {len(filtered_pcd.points)}")

    return filtered_pcd


def main(path: str, ransac_dist_threshold: list, ransac_num_pts: int, dbscan_eps: float, dbscan_min_pts: int,
         kmeans_n_clusters: int, min_points_per_cluster: int):
    pcd_file_path = path
    ransac_dist_threshold = ransac_dist_threshold  # RANSAC distance threshold for each iteration
    ransac_num_pts = ransac_num_pts  # RANSAC points to sample

    # Clustering Parameters for DBSCAN and K-Means (Needs tuning)
    dbscan_eps = dbscan_eps  # DBSCAN: Max distance between points for neighborhood
    dbscan_min_pts = dbscan_min_pts  # DBSCAN: Min points to form a cluster
    kmeans_n_clusters = kmeans_n_clusters  # K-Means: How many clusters to find

    # Load Point Cloud
    try:
        original_pcd = o3d.io.read_point_cloud(pcd_file_path)
        if not original_pcd.has_points():
            raise ValueError("Loaded point cloud is empty.")
        print(f"Successfully loaded point cloud from {pcd_file_path} with {len(original_pcd.points)} points.")
    except Exception as exc:
        print(f"Error loading PCD file '{pcd_file_path}': {exc}")
        raise exc

    # Rotate the point cloud by 180 degrees around the Z-axis
    R = original_pcd.get_rotation_matrix_from_axis_angle(np.array([0, 0, np.pi]))
    original_pcd.rotate(R, center=(0, 0, 0))

    print("Displaying Original Point Cloud (After Rotation):")
    o3d.visualization.draw_geometries([original_pcd], window_name="Original Point Cloud")

    # RANSAC to Remove Planar Points
    print(f"--- RANSAC Plane Removal ({len(ransac_dist_threshold)} iterations) ---")
    original_pcd, _ = original_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    current_pcd = copy.deepcopy(original_pcd)
    all_removed_planes = []

    for idx in range(len(ransac_dist_threshold)):
        print(f"RANSAC Iter {idx + 1}/{len(ransac_dist_threshold)}: Distance Threshold = {ransac_dist_threshold[idx]}")

        points_kept_iter, points_removed_iter = remove_planar_points_ransac(
            current_pcd,
            distance_threshold=ransac_dist_threshold[idx],
            ransac_n=ransac_num_pts,
            num_iterations=1000,
        )

        print(f"  Removed {len(points_removed_iter.points)} points in this iteration.")
        current_pcd = points_kept_iter

    points_kept_final, _ = current_pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
    points_kept_final, _ = points_kept_final.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.2)

    print("--- RANSAC Processing Complete ---")
    print(
        f"Number of points remaining after {len(ransac_dist_threshold)} RANSAC iterations: {len(points_kept_final.points)}")

    o3d.visualization.draw_geometries([points_kept_final], window_name=f"After RANSAC")

    # Cluster the Final Kept Points
    target_pcd_for_clustering = points_kept_final
    print("--- Running Clustering Algorithms on Final Kept Points ---")

    # DBSCAN
    print(f"Running DBSCAN (eps={dbscan_eps}, min_points={dbscan_min_pts})")
    dbscan_colored_pcd = cluster_by_dbscan(target_pcd_for_clustering, eps=dbscan_eps, min_points=dbscan_min_pts,
                                           print_progress=True)
    dbscan_colored_pcd_filtered = filter_clusters_by_size(dbscan_colored_pcd, min_points_per_cluster)
    if dbscan_colored_pcd:
        print("Visualizing DBSCAN results (Noise is Gray)")
        o3d.visualization.draw_geometries([dbscan_colored_pcd_filtered], window_name=f"DBSCAN")
    else:
        print("DBSCAN clustering failed or produced no result to visualize.")

    # K-Means
    print(f"Running K-Means (n_clusters={kmeans_n_clusters})")
    kmeans_colored_pcd = cluster_by_kmeans(target_pcd_for_clustering, n_clusters=kmeans_n_clusters)
    kmeans_colored_pcd_filtered = filter_clusters_by_size(kmeans_colored_pcd, min_points_per_cluster)

    if kmeans_colored_pcd:
        print("Visualizing K-Means results")
        o3d.visualization.draw_geometries([kmeans_colored_pcd_filtered], window_name=f"K-Means")
    else:
        print("K-Means clustering failed or produced no result to visualize.")


if __name__ == "__main__":
    # main("Clouds/output3611.pcd", [0.06, 0.08], 4, 0.035, 15, 2, 50)
    main("Clouds/Pl_Major_4M_downsampled.ply", [0.06,], 4, 0.35, 15, 20, 50)
