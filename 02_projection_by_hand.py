"""
ConfGrasp — Week 2, Script 2
Write the camera projection yourself, then prove it is correct.

Last week Open3D built the point cloud for you with one function call.
This week you build it with numpy, and verify your version matches
Open3D's to within floating-point error.

Why bother? Because in month 5 you will be debugging a grasp that lands
two centimetres off, and the bug will be in a transform. You cannot debug
a black box. After this script, it is not a black box.

Run:  python 02_projection_by_hand.py
"""

import numpy as np
import open3d as o3d


def banner(text: str) -> None:
    print("\n" + "=" * 64)
    print(text)
    print("=" * 64)


# ----------------------------------------------------------------------
# Load the same data as last week
# ----------------------------------------------------------------------

dataset = o3d.data.SampleRedwoodRGBDImages()
color_raw = o3d.io.read_image(dataset.color_paths[0])
depth_raw = o3d.io.read_image(dataset.depth_paths[0])

color_np = np.asarray(color_raw)
depth_np = np.asarray(depth_raw)

intrinsic = o3d.camera.PinholeCameraIntrinsic(
    o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
)
fx, fy = intrinsic.get_focal_length()
cx, cy = intrinsic.get_principal_point()

DEPTH_SCALE = 1000.0   # millimetres -> metres
DEPTH_TRUNC = 4.0      # ignore beyond 4 m (must match Open3D for the test)


# ----------------------------------------------------------------------
# PART 1 — The projection, written out by hand
# ----------------------------------------------------------------------
# Last week you did this for ONE pixel with a for-loop's worth of thinking.
# Doing it 300,000 times with a Python loop would take seconds. Doing it
# with numpy arrays takes milliseconds, because numpy runs the loop in C.
#
# This is called vectorisation, and it is the difference between a demo
# and something that runs at camera framerate.

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


banner("PART 1 — Your own projection")

my_points, my_colors = depth_to_pointcloud(
    depth_np, color_np, fx, fy, cx, cy, DEPTH_SCALE, DEPTH_TRUNC
)

print(f"pixels in image     : {depth_np.size:,}")
print(f"points you produced : {len(my_points):,}")
print(f"discarded           : {depth_np.size - len(my_points):,}"
      f"  (no depth reading, or beyond {DEPTH_TRUNC} m)")


# ----------------------------------------------------------------------
# PART 2 — Prove it is correct
# ----------------------------------------------------------------------
# This is the habit that makes you different from most engineers. You do
# not assume your implementation is right. You check it against a known
# good one, with a number, not a glance at a picture.

banner("PART 2 — Verifying against Open3D")

rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_raw, depth_raw,
    depth_scale=DEPTH_SCALE,
    depth_trunc=DEPTH_TRUNC,
    convert_rgb_to_intensity=False,
)
reference = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
ref_points = np.asarray(reference.points)

print(f"Open3D points : {len(ref_points):,}")
print(f"your points   : {len(my_points):,}")

if len(ref_points) == len(my_points):
    # Maximum disagreement on any single coordinate of any single point.
    max_error = np.abs(my_points - ref_points).max()
    print(f"\nlargest difference on any coordinate: {max_error:.3e} metres")

    if max_error < 1e-9:
        print("PASS - identical to floating-point precision.")
        print("You now understand exactly what that library call does.")
    else:
        print("Close but not identical - check depth_scale and depth_trunc.")
else:
    print("\nDifferent point counts. Your 'keep' mask does not match")
    print("Open3D's filtering. Check the depth_trunc comparison.")


# ----------------------------------------------------------------------
# PART 3 — The reverse: 3D back to pixels
# ----------------------------------------------------------------------
# Going backwards matters more than it sounds. When your network predicts
# a grasp at some 3D position, you will want to draw it on the camera
# image to check it is where you think. That means projecting 3D -> 2D.
#
# Rearrange the same equations:
#     u = X * fx / Z + cx
#     v = Y * fy / Z + cy

def project_to_image(points, fx, fy, cx, cy):
    """Project (N,3) 3D points back onto the image plane. Returns (N,2)."""
    X, Y, Z = points[:, 0], points[:, 1], points[:, 2]
    u = X * fx / Z + cx
    v = Y * fy / Z + cy
    return np.stack((u, v), axis=-1)


banner("PART 3 — Projecting back to pixels (round trip test)")

uv = project_to_image(my_points, fx, fy, cx, cy)

print(f"first 3 points in 3D  :\n{my_points[:3].round(4)}")
print(f"\nprojected back to pixels:\n{uv[:3].round(2)}")
print("\nThese should land on whole-number pixel coordinates, because")
print("that is where they came from. A round trip that does not return")
print("you to the start means a sign error or a swapped fx/fy.")

rounding_error = np.abs(uv - np.round(uv)).max()
print(f"\nlargest distance from a whole pixel: {rounding_error:.3e}")


# ----------------------------------------------------------------------
# PART 4 — Transformation matrices, properly this time
# ----------------------------------------------------------------------
# Last week you used a 4x4 matrix to flip the cloud the right way up and
# I told you not to worry about it. Now worry about it, because in month
# 5 you will need one of these to describe where your camera sits
# relative to your robot's gripper. That relationship is called hand-eye
# calibration, and getting it wrong is the classic reason a robot misses.
#
# A 4x4 transform looks like this:
#
#     [ R11 R12 R13 | tx ]     R = 3x3 rotation
#     [ R21 R22 R23 | ty ]     t = 3x1 translation (movement)
#     [ R31 R32 R33 | tz ]
#     [  0   0   0  |  1 ]     bottom row is always this
#
# Why 4x4 and not 3x3? Because a 3x3 matrix can rotate but cannot move.
# Adding the fourth row and column - called homogeneous coordinates -
# lets one matrix do both, so you can chain them together by multiplying.

def make_transform(rotation=np.eye(3), translation=np.zeros(3)):
    """Assemble a 4x4 transform from a 3x3 rotation and a 3-vector move."""
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T


def apply_transform(points, T):
    """Apply a 4x4 transform to (N,3) points."""
    n = points.shape[0]
    # Add a column of ones -> (N,4). This is the homogeneous trick.
    homogeneous = np.hstack((points, np.ones((n, 1))))
    # Transform every point. The .T are because the matrix expects
    # points as columns, but we store them as rows.
    moved = (T @ homogeneous.T).T
    return moved[:, :3]      # drop the ones column


def rotation_z(degrees):
    """Rotation matrix about the Z axis (the camera's forward direction)."""
    t = np.radians(degrees)
    return np.array([
        [np.cos(t), -np.sin(t), 0],
        [np.sin(t),  np.cos(t), 0],
        [0,          0,         1],
    ])

def rotation_x(degrees):
    """Rotation matrix about the x axis (the camera's forward direction)."""
    t = np.radians(degrees)
    return np.array([
        [1, 0, 0],
        [0,  np.cos(t), -np.sin(t)],
        [0,          np.sin(t),         np.cos(t)],
    ])


banner("PART 4 — Transforms")

# The flip from last week, built explicitly.
flip = make_transform(rotation=np.diag([1.0, -1.0, -1.0]))
print("The 'flip upright' transform from week 1:")
print(flip)
print("\nThe diagonal 1,-1,-1 means: keep X, reverse Y, reverse Z.")

# Move half a metre to the right.
shift = make_transform(translation=np.array([0.5, 0.0, 0.0]))
shifted = apply_transform(my_points, shift)
print(f"\nmoving +0.5 m in X:")
print(f"  mean X before : {my_points[:, 0].mean():+.4f} m")
print(f"  mean X after  : {shifted[:, 0].mean():+.4f} m")

# Rotate 30 degrees about Z.
spin = make_transform(rotation=rotation_z(5))
spun = apply_transform(my_points, spin)
print(f"\nrotating 30 deg about Z:")
print(f"  point 0 before : {my_points[0].round(4)}")
print(f"  point 0 after  : {spun[0].round(4)}")
print(f"  distance from origin is unchanged: "
      f"{np.linalg.norm(my_points[0]):.6f} -> {np.linalg.norm(spun[0]):.6f}")
print("\nRotation never changes distances. If yours does, the matrix")
print("is not a valid rotation - a useful check later on.")

# Transforms compose by multiplication, and ORDER MATTERS.
rotate_then_move = shift @ spin
move_then_rotate = spin @ shift
difference = np.abs(rotate_then_move - move_then_rotate).max()
print(f"\nshift@spin vs spin@shift differ by: {difference:.4f}")
print("Order matters. Rotating then moving is not the same as moving")
print("then rotating. This causes real bugs in robot code.")


# ----------------------------------------------------------------------
# PART 5 — What wrong intrinsics look like
# ----------------------------------------------------------------------
# I said last week that wrong lens numbers wreck everything. Here is the
# proof, so you recognise the symptom when you see it on real hardware.

banner("PART 5 — Wrong intrinsics  (close each window to continue)")

def to_o3d(points, colors):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    pc.colors = o3d.utility.Vector3dVector(colors)
    pc.transform(flip)
    return pc

correct = to_o3d(my_points, my_colors)

bad_points, bad_colors = depth_to_pointcloud(
    depth_np, color_np, fx * 1.4, fy, cx, cy, DEPTH_SCALE, DEPTH_TRUNC
)
wrong_fx = to_o3d(bad_points, bad_colors)

print("Window 1: correct intrinsics.")
o3d.visualization.draw_geometries(
    [correct], window_name="Correct", width=1024, height=768)

print("Window 2: fx too large by 40%. The scene is squashed horizontally.")
print("On a real camera this would make your gripper miss sideways,")
print("by more the further the object is from the image centre.")
o3d.visualization.draw_geometries(
    [wrong_fx], window_name="fx wrong by 40%", width=1024, height=768)


# ----------------------------------------------------------------------
# show the rotation

# Show the rotation.
original = to_o3d(my_points, my_colors)
rotated  = to_o3d(spun, my_colors)

# Tint the rotated one red so you can tell them apart.
rotated.paint_uniform_color([1.0, 0.3, 0.3])

o3d.visualization.draw_geometries(
    [original, rotated],
    window_name="Original (colour) vs rotated (red)",
    width=1024, height=768,
)

# ----------------------------------------------------------------------


np.save("points_week2.npy", my_points)

banner("DONE  —  now do the exercises")
print("""
Saved: points_week2.npy

EXERCISES. Do these yourself, in a new file. They are the actual
learning; reading the code above is not enough.

  1. Write rotation_x(degrees) and rotation_y(degrees), matching the
     style of rotation_z above. Check each one leaves distances
     unchanged.

  2. Build a transform that rotates 90 degrees about X AND moves
     0.3 m in Z, in a single matrix. Apply it. Verify the result
     equals applying the two separately in the right order.

  3. Write invert_transform(T) that undoes a transform. Hint: the
     inverse of a rotation is its transpose, and the translation must
     be rotated too. Verify that applying T then its inverse returns
     the original points to within 1e-10.
     (Do not use np.linalg.inv - work out why the structured version
     is faster and more numerically stable.)

  4. Take 10 random points, project them to pixels with
     project_to_image, then convert back to 3D using the depth you
     started with. Confirm you recover the original points.

  5. Deliberately swap cx and cy in depth_to_pointcloud and look at
     the result. Describe in one sentence what it does to the cloud.

DONE WHEN: your PART 2 test prints PASS, and exercise 3 verifies to
1e-10. Commit both scripts and your exercise file.

Next week: real grasp data. You will download a slice of GraspNet-1Billion
and look at what a 6-DoF grasp actually is - a position, an orientation,
a gripper width, and a quality score.
""")