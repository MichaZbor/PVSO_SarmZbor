import open3d as o3d


def display(path):
    pcd = o3d.io.read_point_cloud(path)
    o3d.visualization.draw_geometries([pcd], window_name="Displayed Point Cloud")

def main():
    ply_path = "Clouds/Pl_Major_4M.ply"
    save_path = "Clouds/Pl_Major_4M_downsampled.ply"

    # Load the original point cloud
    pcd = o3d.io.read_point_cloud(ply_path)

    # Estimate a voxel size that reduces the point count ~10x
    original_points = len(pcd.points)
    print(f"Original point count: {original_points}")

    # Estimate a good voxel size
    # Try a few values and pick one that results in ~10x reduction
    voxel_size = 0.05  # You can tweak this to get closer to 10x reduction
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)

    downsampled_points = len(pcd_down.points)
    print(f"Downsampled point count: {downsampled_points} (reduction ratio: {original_points / downsampled_points:.2f}x)")

    # Save the downsampled point cloud
    o3d.io.write_point_cloud(save_path, pcd_down)
    print(f"Saved downsampled point cloud to {save_path}")

    # Visualize
    o3d.visualization.draw_geometries([pcd_down], window_name="Downsampled Point Cloud")


if __name__ == "__main__":
    # main()
    display("Clouds/output3611.pcd")
