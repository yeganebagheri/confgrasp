"""
ConfGrasp — Week 8
Your first real scan.

Everything so far has been simulated: a clean mesh, synthetic noise,
synthetic holes. This week the data comes from an actual depth sensor
pointed at an actual object on your actual desk.

Two new skills, both of which every real system needs:

  PLANE REMOVAL   - a real scan is mostly table. You must find and
                    delete the supporting surface before anything else.
  CLUSTERING      - what remains is several objects plus junk. You must
                    separate them into individual things.

No scan yet? The script builds a synthetic desk scene instead, so you
can read the code today and swap in your scan tomorrow.

HOW TO MAKE A SCAN (iPhone Pro / iPad Pro)
  1. Install Record3D from the App Store.
  2. Put one object on a plain table. Good light. No clutter.
  3. Record a short clip, moving slowly around the object.
  4. Export as .ply (point cloud).
  5. Copy it next to this script as  scan.ply

Run:  python 09_real_scan.py
"""

import os
import numpy as np
import open3d as o3d

from grasp_lib import find_grasps, MAX_WIDTH, MIN_WIDTH

SCAN_FILE = "scan.ply"


def banner(text):
    print("\n" + "=" * 66)
    print(text)
    print("=" * 66)


# ----------------------------------------------------------------------
# PART 1 — Load the scan (or fake one)
# ----------------------------------------------------------------------

def synthetic_desk():
    """A fallback scene: a table with three objects on it, plus noise."""
    rng = np.random.default_rng(0)
    pieces = []

    table = o3d.geometry.TriangleMesh.create_box(0.6, 0.6, 0.01)
    table.translate((-0.3, -0.3, -0.01))
    pieces.append(table.sample_points_uniformly(20000))

    # Sizes matter. The gripper opens to MAX_WIDTH (5 cm), so every
    # object's NARROWEST dimension must be comfortably under that - and
    # clustering tends to inflate measured size by a few millimetres,
    # because leftover table points cling to each object's base.
    #
    # The first version of this scene used 4.4-5 cm objects and every
    # one came out 2-5 mm too wide. A real bin-picking cell hits this
    # constantly: the commonest reason a pick is impossible is simply
    # that the gripper does not fit.
    for mesh, where in (
        (o3d.geometry.TriangleMesh.create_cylinder(0.014, 0.09), (0.08, 0.05)),
        (o3d.geometry.TriangleMesh.create_box(0.035, 0.022, 0.08), (-0.07, 0.0)),
        (o3d.geometry.TriangleMesh.create_sphere(0.016), (0.0, -0.10)),
    ):
        mesh.translate((where[0], where[1], 0.045))
        pieces.append(mesh.sample_points_uniformly(4000))

    cloud = pieces[0]
    for p in pieces[1:]:
        cloud += p

    # Sensor-like noise and a sprinkle of stray points.
    pts = np.asarray(cloud.points)
    pts += rng.normal(0, 0.0015, pts.shape)
    strays = rng.uniform([-0.3, -0.3, 0.0], [0.3, 0.3, 0.2], (300, 3))
    cloud.points = o3d.utility.Vector3dVector(np.vstack([pts, strays]))
    return cloud


banner("PART 1 — Loading")

if os.path.exists(SCAN_FILE):
    cloud = o3d.io.read_point_cloud(SCAN_FILE)
    print(f"  loaded {SCAN_FILE}")
    using_real = True
else:
    cloud = synthetic_desk()
    print(f"  no {SCAN_FILE} found - using a synthetic desk scene instead")
    print("  (scan something with Record3D and drop it here to use real data)")
    using_real = False

extent = cloud.get_max_bound() - cloud.get_min_bound()
print(f"  points : {len(cloud.points):,}")
print(f"  extent : {extent.round(3)} metres")

if using_real and extent.max() > 5:
    print("\n  WARNING: this scene is over 5 m across. Record3D usually")
    print("  exports in metres - if yours is in millimetres, scale it:")
    print("      cloud.scale(0.001, center=(0,0,0))")

o3d.visualization.draw_geometries([cloud], window_name="1. Raw scan",
                                  width=1000, height=750)


# ----------------------------------------------------------------------
# PART 2 — Downsample
# ----------------------------------------------------------------------
# A phone scan can be a million points, most of them redundant. Voxel
# downsampling keeps one point per small cube - far fewer points, same
# shape, and every later step runs faster.

banner("PART 2 — Downsampling")

VOXEL = 0.003            # 3 mm
before = len(cloud.points)
cloud = cloud.voxel_down_sample(VOXEL)
print(f"  {before:,} -> {len(cloud.points):,} points at {VOXEL * 1000:.0f} mm")


# ----------------------------------------------------------------------
# PART 3 — Find and remove the table
# ----------------------------------------------------------------------
# Most of a real scan is the surface the object sits on. It is useless
# for grasping and it ruins clustering, because it connects everything
# into one blob.
#
# RANSAC finds it: pick 3 random points, fit a plane through them, count
# how many other points lie on that plane. Repeat thousands of times and
# keep the plane with the most supporters. Robust to clutter, because
# wrong guesses get few votes.

banner("PART 3 — Removing the support surface")

plane, inliers = cloud.segment_plane(distance_threshold=0.006,
                                     ransac_n=3,
                                     num_iterations=2000)
a, b, c, d = plane
print(f"  plane equation : {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")
print(f"  table points   : {len(inliers):,} "
      f"({100 * len(inliers) / len(cloud.points):.0f}% of the scan)")
print(f"  plane normal   : {np.array([a, b, c]).round(3)}  "
      f"(should point roughly up or down)")

table = cloud.select_by_index(inliers)
objects = cloud.select_by_index(inliers, invert=True)

table.paint_uniform_color([0.85, 0.85, 0.85])
objects.paint_uniform_color([0.2, 0.6, 0.9])

print(f"  remaining      : {len(objects.points):,} points\n")
print("  Grey = the surface we are discarding. Blue = everything else.")

o3d.visualization.draw_geometries([table, objects],
                                  window_name="2. Table (grey) vs objects (blue)",
                                  width=1000, height=750)


# ----------------------------------------------------------------------
# PART 4 — Separate into individual objects
# ----------------------------------------------------------------------
# DBSCAN groups points that are close together. Points with no close
# neighbours are labelled noise and thrown away - which conveniently
# removes the stray points every depth sensor produces.

banner("PART 4 — Clustering into objects")

labels = np.array(objects.cluster_dbscan(eps=0.015, min_points=30))
n_clusters = labels.max() + 1
n_noise = int((labels < 0).sum())

print(f"  clusters found : {n_clusters}")
print(f"  discarded as noise : {n_noise:,} points")

if n_clusters == 0:
    raise SystemExit(
        "\nNo clusters. Try a larger eps (points may be further apart\n"
        "than 15 mm) or a smaller min_points.")

palette = [[0.9, 0.3, 0.2], [0.2, 0.7, 0.3], [0.3, 0.4, 0.9],
           [0.9, 0.7, 0.1], [0.7, 0.3, 0.8], [0.2, 0.8, 0.8]]

shown, clusters = [], []
print(f"\n  gripper opens {100 * MIN_WIDTH:.1f} - {100 * MAX_WIDTH:.1f} cm\n")
print(f"  {'id':>3} {'points':>8} {'size (cm)':>22} {'narrowest':>10}  verdict")
for k in range(n_clusters):
    part = objects.select_by_index(np.where(labels == k)[0])
    size = part.get_max_bound() - part.get_min_bound()
    narrow = size.min()

    # Can the gripper span its narrowest dimension? Say WHY if not -
    # "no" alone sends you hunting through the code for a reason that
    # is right here in the number.
    if narrow >= MAX_WIDTH:
        verdict = f"too wide by {100 * (narrow - MAX_WIDTH):.1f} cm"
    elif narrow <= MIN_WIDTH:
        verdict = f"too thin by {100 * (MIN_WIDTH - narrow):.1f} cm"
    else:
        verdict = "GRASPABLE"

    print(f"  {k:>3} {len(part.points):>8,}  "
          f"{str((100 * size).round(1)):>22} {100 * narrow:>9.1f}  {verdict}")

    part.paint_uniform_color(palette[k % len(palette)])
    shown.append(part)
    clusters.append(part)

print("""
  "Graspable" here only asks whether the narrowest side fits between
  the fingers. A real system also checks reachability and collisions.

  Note that clustering inflates measured size: leftover table points
  cling to each object's base and get swept into its cluster, adding a
  few millimetres. On a borderline object that is the difference
  between a pick and a refusal.
""")

o3d.visualization.draw_geometries(shown, window_name="3. Separated objects",
                                  width=1000, height=750)


# ----------------------------------------------------------------------
# PART 5 — Grasp the biggest graspable cluster
# ----------------------------------------------------------------------

banner("PART 5 — Finding grasps on real data")

candidates = [c for c in clusters
              if MIN_WIDTH < (c.get_max_bound() - c.get_min_bound()).min()
              < MAX_WIDTH]
if not candidates:
    widths = sorted(100 * (c.get_max_bound() - c.get_min_bound()).min()
                    for c in clusters)
    raise SystemExit(
        f"\nNothing in this scene fits the gripper.\n\n"
        f"  narrowest dimensions found : {[round(w, 1) for w in widths]} cm\n"
        f"  gripper range              : "
        f"{100 * MIN_WIDTH:.1f} - {100 * MAX_WIDTH:.1f} cm\n\n"
        "This is a real failure mode, not a bug - in a bin-picking cell\n"
        "the commonest reason a pick is impossible is that the gripper\n"
        "does not fit. But if it happens on every object at once, the\n"
        "scene is wrongly scaled. Shrink the objects in synthetic_desk(),\n"
        "or raise MAX_WIDTH in grasp_lib.py to match a bigger gripper.")

target = max(candidates, key=lambda c: len(c.points))
print(f"  chosen cluster : {len(target.points):,} points")

target.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.012, max_nn=30))
target.normalize_normals()

# The sensor was roughly above the scene, so normals face upward-ish.
# This is the only orientation information a single view gives you.
target.orient_normals_towards_camera_location(
    target.get_center() + np.array([0, 0, 0.5]))

grasps = find_grasps(target, n_samples=300000, seed=0)
grasps.sort(key=lambda g: g.score, reverse=True)
print(f"  grasps found   : {len(grasps):,}")

if not grasps:
    raise SystemExit(
        "\nNo grasps. On a single-view scan this happens when no two\n"
        "opposite-facing surfaces are both visible - see week 4. Try a\n"
        "thinner object, or merge scans from several angles.")

print(f"  best score     : {grasps[0].score:.3f}")

target.paint_uniform_color([0.6, 0.6, 0.6])
shown = [target] + [g.mesh() for g in grasps[:15]]
o3d.visualization.draw_geometries(shown, window_name="4. Grasps on real data",
                                  width=1000, height=750)


# ----------------------------------------------------------------------

o3d.io.write_point_cloud("week8_object.ply", target)

banner("DONE")
print(f"""
Saved: week8_object.ply

WHAT IS NEW HERE

  RANSAC plane fitting - the standard way to find a table. Guess a
  plane from 3 random points, count supporters, repeat, keep the best.
  Robust to clutter because wrong guesses get few votes.

  DBSCAN clustering - groups nearby points into objects and labels
  isolated points as noise, which also cleans up sensor strays.

  Both appear in essentially every industrial picking pipeline. Fizyr,
  Smart Robotics and Viroteq all do some version of this before any
  neural network runs.

WHAT YOU CANNOT DO HERE

  You have no ground truth. On the bunny you could check every grasp
  against the complete model. On a real scan there is no answer key -
  the only way to know if a grasp works is to try it with a robot.

  That is exactly why GraspNet matters: 190 real cluttered scenes with
  labels from physics simulation. Real data AND an answer key.

EXERCISES

  1. Scan the same object from three angles and merge the scans
     (look up ICP registration). Do grasps appear that a single view
     could not find? Week 4 predicts yes.

  2. Scan something shiny or transparent - a glass, a metal spoon.
     Count the holes. This is the unsolved problem Fizyr's customers
     actually have.

  3. Vary eps in DBSCAN from 0.005 to 0.05. Watch objects merge and
     split. There is no universally right value, which is why learned
     segmentation replaced it.

  4. Put two objects touching each other. Does DBSCAN separate them?
     This failure is why the field moved to learned instance
     segmentation.

NEXT: GraspNet, if the download has finished. Real sensor data with
real labels, 88 objects, 190 scenes.
""")