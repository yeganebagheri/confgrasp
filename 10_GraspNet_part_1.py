import os
from PIL import Image
import numpy as np
import scipy.io as scio
import open3d as o3d

# Define your data directory path
data_dir = 'doc/example_data'

# Constants
DEPTH_SCALE = 1000.0  # Depending on meta.mat, factor_depth is often 1000
DEPTH_TRUNC = 4.0     # Maximum depth in meters

# 1. Load RGB color image (normalized to [0, 1])
color_rgb = np.array(Image.open(os.path.join(data_dir, 'color.png')), dtype=np.float32)

# 2. Load Depth image
depth_mm = np.array(Image.open(os.path.join(data_dir, 'depth.png')))

# 3. Load Workspace Mask
workspace_mask = np.array(Image.open(os.path.join(data_dir, 'workspace_mask.png')))

# 4. Load Metadata (.mat file containing camera intrinsics and depth factor)
meta = scio.loadmat(os.path.join(data_dir, 'meta.mat'))
intrinsic_matrix = meta['intrinsic_matrix']
factor_depth = meta['factor_depth']f

print(f"factor_depth: {factor_depth}, factor_depth shape: {factor_depth.shape}")

print("Data loaded successfully!")
print(f"Color shape: {color_rgb.shape}, Depth shape: {depth_mm.shape}")



# step 1- Build a point cloud using your own projection function from week 2, not Open3D’s helper.

# Extract intrinsics correctly from the numpy 3x3 matrix
fx = intrinsic_matrix[0, 0]
fy = intrinsic_matrix[1, 1]
cx = intrinsic_matrix[0, 2]
cy = intrinsic_matrix[1, 2]
print(f"intrinsic_matrix: {intrinsic_matrix}, intrinsic_matrix shape: {intrinsic_matrix.shape}")

# --- Step 2: Apply workspace_mask.png to keep only the bin region ---
mask = workspace_mask > 0


# Apply the mask to zero out the background in depth and color images
masked_color = color_rgb.copy()
masked_color[~mask] = 0  # Background pixels become black


masked_depth = depth_mm.copy()
masked_depth[~mask] = 0  # Background depth becomes 0


print("Workspace mask applied successfully!")
print(f"Mask shape: {mask.shape}, Number of valid bin pixels: {np.sum(mask)}")


def depth_to_pointcloud(depth_mm, color_rgb, fx, fy, cx, cy,
                        depth_scale=1000.0, depth_trunc=4.0):
    """Convert a depth image into an (N,3) array of 3D points in metres.

    depth_mm  : (H,W) array of distances in millimetres, 0 = no reading
    color_rgb : (H,W,3) array of uint8 colours
    returns   : points (N,3) float64, colors (N,3) float64 in 0..1
    """
    height, width = depth_mm.shape

    # Build two grids holding the pixel coordinates of every pixel.
    # v = row index (vertical), u = column index (horizontal).
    # np.indices gives both at once, each with shape (H,W).
    v_grid, u_grid = np.indices((height, width))

    # Convert depth to metres. Z is simply the depth.
    Z = depth_mm.astype(np.float64) / depth_scale

    # The two lines you learned last week, applied to every pixel at once.
    X = (u_grid - cx) * Z / fx
    Y = (v_grid - cy) * Z / fy

    # Stack into (H,W,3), then flatten to a long list of points (H*W, 3).
    points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
    colors = color_rgb.reshape(-1, 3).astype(np.float64) / 255.0

    # Drop the pixels the sensor could not measure, and anything too far.
    # Note Z is computed for them too - the maths does not fail, it just
    # produces a meaningless point sitting at the camera origin.
    keep = (Z.reshape(-1) > 0) & (Z.reshape(-1) < depth_trunc)

    return points[keep], colors[keep]

# Generate points using the masked depth and color
my_points, my_colors = depth_to_pointcloud(
    masked_depth, masked_color, fx, fy, cx, cy, DEPTH_SCALE, DEPTH_TRUNC
)


print(f"pixels in image  outside the mask   : {depth_mm.size:,}")
print(f"points you produced : {len(my_points):,}")

#pixels outside the mask
print(f"discarded outside the mask: {depth_mm.size - len(my_points):,}"
      f"  (no depth reading, or beyond {DEPTH_TRUNC} m)")

#pixels inside the mask with no depth
print(f"discarded inside the mask with no depth: {masked_depth.size - len(my_points):,}"
      f"  (no depth reading, or beyond {DEPTH_TRUNC} m)")


# --- Step 3: Visualize using Open3D ---
# Create an Open3D PointCloud object
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(my_points)
pcd.colors = o3d.utility.Vector3dVector(my_colors)

# Visualize the filtered point cloud (only the bin contents)
print("Opening Open3D visualizer window...")
o3d.visualization.draw_geometries([pcd], window_name="GraspNet Bin Point Cloud", width=1024, height=768)