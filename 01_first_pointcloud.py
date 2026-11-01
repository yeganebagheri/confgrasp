"""
ConfGrasp — Week 1, Script 1
Depth image -> 3D point cloud, and the basic operations you will use
every single week of this project.

Run:  python 01_first_pointcloud.py

Four windows open in sequence. Close each one (press Q) to continue.
"""

import numpy as np
import open3d as o3d


def banner(text: str) -> None:
    print("\n" + "=" * 62)
    print(text)
    print("=" * 62)


# ----------------------------------------------------------------------
# STEP 1 — Load an RGB-D image
# ----------------------------------------------------------------------
# An RGB-D image is two pictures of the same scene, pixel-aligned:
#   - colour: what the camera sees (3 channels, 0-255)
#   - depth : how far away each pixel is, in millimetres
# This is exactly what an OAK-D or RealSense gives you. Here we use a
# sample that Open3D downloads once and caches, so you need no hardware.

banner("STEP 1 — Loading an RGB-D image")

dataset = o3d.data.SampleRedwoodRGBDImages()
color_raw = o3d.io.read_image(dataset.color_paths[0])
depth_raw = o3d.io.read_image(dataset.depth_paths[0])

color_np = np.asarray(color_raw)
depth_np = np.asarray(depth_raw)

print(f"colour image : shape {color_np.shape}, dtype {color_np.dtype}")
print(f"depth image  : shape {depth_np.shape}, dtype {depth_np.dtype}")

valid = depth_np[depth_np > 0]
print(f"depth range  : {valid.min()} to {valid.max()} (millimetres)")
print(f"invalid px   : {(depth_np == 0).sum()} pixels have no depth reading")
print("\nNOTE: zero means 'the sensor could not measure this pixel'.")
print("Shiny, transparent and very dark surfaces produce these holes.")
print("Handling them is a real part of the job, not an edge case.")

rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_raw,
    depth_raw,
    depth_scale=1000.0,   # depth values are mm, we want metres
    depth_trunc=4.0,      # ignore anything beyond 4 metres
    convert_rgb_to_intensity=False,
)


# ----------------------------------------------------------------------
# STEP 2 — Camera intrinsics: the maths that turns pixels into 3D
# ----------------------------------------------------------------------
# A depth camera gives you (u, v, d): pixel column, pixel row, distance.
# To get a 3D point (X, Y, Z) in metres you need four numbers that
# describe the lens:
#
#   fx, fy = focal length in pixels
#   cx, cy = the optical centre (roughly the image centre)
#
#   Z = d
#   X = (u - cx) * Z / fx
#   Y = (v - cy) * Z / fy
#
# That is the whole pinhole camera model. Learn it now; every 3D vision
# task you will ever do rests on it.

banner("STEP 2 — Camera intrinsics")

intrinsic = o3d.camera.PinholeCameraIntrinsic(
    o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
)
fx, fy = intrinsic.get_focal_length()
cx, cy = intrinsic.get_principal_point()

print(f"image size : {intrinsic.width} x {intrinsic.height}")
print(f"fx, fy     : {fx:.2f}, {fy:.2f}")
print(f"cx, cy     : {cx:.2f}, {cy:.2f}")

# Prove the formula to yourself on one pixel.
u, v = 320, 240
d_mm = depth_np[v, u]
if d_mm > 0:
    Z = d_mm / 1000.0
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    print(f"\nPixel ({u},{v}) with depth {d_mm} mm becomes the 3D point:")
    print(f"  X = {X:+.4f} m   Y = {Y:+.4f} m   Z = {Z:+.4f} m")
    print("\nThat single calculation, run on every pixel, IS a point cloud.")


# ----------------------------------------------------------------------
# STEP 3 — Build and view the point cloud
# ----------------------------------------------------------------------

banner("STEP 3 — Building the point cloud  (close the window to continue)")

pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

# Open3D's camera convention has Y and Z pointing the opposite way to
# how we like to view it, so flip it the right way up.
pcd.transform([[1, 0, 0, 0],
               [0, -1, 0, 0],
               [0, 0, -1, 0],
               [0, 0, 0, 1]])

points = np.asarray(pcd.points)
print(f"points        : {len(points):,}")
print(f"bounding box  : {pcd.get_max_bound() - pcd.get_min_bound()} (metres)")
print("\nRotate with the left mouse button. Scroll to zoom. Press Q to close.")

o3d.visualization.draw_geometries(
    [pcd], window_name="1. Raw point cloud", width=1024, height=768
)


# ----------------------------------------------------------------------
# STEP 4 — Downsampling: why nobody processes raw point clouds
# ----------------------------------------------------------------------
# A single frame is ~300,000 points. A neural network cannot take that
# directly and stay inside a latency budget. Voxel downsampling divides
# space into small cubes and keeps one point per cube — far fewer points,
# same shape.
#
# Fizyr processes a frame in under 20 ms. This is one of the reasons it
# is possible.

banner("STEP 4 — Downsampling  (close the window to continue)")

for voxel in (0.01, 0.02, 0.05):
    small = pcd.voxel_down_sample(voxel_size=voxel)
    pct = 100 * len(small.points) / len(pcd.points)
    print(f"voxel {voxel * 100:>4.1f} cm  ->  {len(small.points):>7,} points  ({pct:5.1f}% kept)")

pcd_down = pcd.voxel_down_sample(voxel_size=0.02)

o3d.visualization.draw_geometries(
    [pcd_down], window_name="2. Downsampled (2 cm voxels)", width=1024, height=768
)


# ----------------------------------------------------------------------
# STEP 5 — Surface normals: which way each surface faces
# ----------------------------------------------------------------------
# A normal is an arrow perpendicular to the surface at each point.
# This matters enormously for grasping: a gripper wants to approach
# ALONG the normal and close across the object. Grasp-detection networks
# consume normals as input, and grasp orientation is defined relative
# to them.

banner("STEP 5 — Surface normals  (close the window to finish)")

pcd_down.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
)
pcd_down.orient_normals_towards_camera_location(camera_location=[0.0, 0.0, 0.0])

normals = np.asarray(pcd_down.normals)
print(f"computed {len(normals):,} normals")
print(f"first normal : {normals[0].round(3)}  (a unit vector)")
print(f"lengths all 1: {np.allclose(np.linalg.norm(normals, axis=1), 1.0)}")
print("\nPress N in the window to toggle the normal arrows on and off.")

o3d.visualization.draw_geometries(
    [pcd_down],
    window_name="3. With normals (press N to show arrows)",
    width=1024,
    height=768,
    point_show_normal=True,
)


# ----------------------------------------------------------------------
# Save the result so week 2 can load it
# ----------------------------------------------------------------------

o3d.io.write_point_cloud("output_week1.ply", pcd_down)

banner("DONE")
print("Saved: output_week1.ply")
print("""
Before moving on, make sure you can answer these in your own words:

  1. What is the difference between a depth image and a point cloud?
  2. What do fx, fy, cx, cy mean, and what happens if they are wrong?
  3. Why downsample before processing?
  4. Why does a grasping system care about surface normals?

If any answer is shaky, re-read the comments above and re-run the script.
Understanding beats speed here — everything later is built on this.
""")
