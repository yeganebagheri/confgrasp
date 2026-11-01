"""
ConfGrasp — Week 3
What a 6-DoF grasp actually is, and how to find one.

Everything you learned about transforms gets used here. A grasp IS a
transform: it says where the gripper goes and which way it is turned.

Run:  python 04_what_is_a_grasp.py

Several windows open in turn. Close each one to continue.
"""

import numpy as np
import open3d as o3d


def banner(text):
    print("\n" + "=" * 64)
    print(text)
    print("=" * 64)


def make_transform(rotation=np.eye(3), translation=np.zeros(3)):
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T


# ----------------------------------------------------------------------
# PART 1 — Build a gripper
# ----------------------------------------------------------------------
# A two-finger gripper. We define it ONCE, sitting at the origin facing
# a fixed direction, and then move it around with transforms.
#
# The convention (the same one GraspNet uses):
#     +X = the APPROACH direction - the way the gripper moves in
#     +Y = the CLOSING direction  - the way the fingers squeeze
#     +Z = whatever is left over  (X cross Y)
#
# The "grasp point" sits between the fingertips, at the origin.

FINGER_LENGTH = 0.04      # 4 cm fingers
FINGER_THICK = 0.006
WRIST_LENGTH = 0.03


def gripper_mesh(width, color=(0.1, 0.4, 0.9)):
    """A gripper opened to `width` metres, at the origin, facing +X."""
    parts = []

    for side in (+1, -1):          # the two fingers
        finger = o3d.geometry.TriangleMesh.create_box(
            FINGER_LENGTH, FINGER_THICK, FINGER_THICK * 3
        )
        finger.translate((-FINGER_LENGTH,
                          side * width / 2 - FINGER_THICK / 2,
                          -FINGER_THICK * 1.5))
        parts.append(finger)

    # the bar joining them
    base = o3d.geometry.TriangleMesh.create_box(
        FINGER_THICK, width + FINGER_THICK, FINGER_THICK * 3
    )
    base.translate((-FINGER_LENGTH - FINGER_THICK,
                    -width / 2 - FINGER_THICK / 2,
                    -FINGER_THICK * 1.5))
    parts.append(base)

    # the wrist, pointing back the way the robot arm would come from
    wrist = o3d.geometry.TriangleMesh.create_box(
        WRIST_LENGTH, FINGER_THICK * 2, FINGER_THICK * 3
    )
    wrist.translate((-FINGER_LENGTH - FINGER_THICK - WRIST_LENGTH,
                     -FINGER_THICK,
                     -FINGER_THICK * 1.5))
    parts.append(wrist)

    gripper = parts[0]
    for p in parts[1:]:
        gripper += p
    gripper.paint_uniform_color(color)
    gripper.compute_vertex_normals()
    return gripper


banner("PART 1 — A gripper, at the origin  (close window to continue)")
print("""
  RED arrow   = X = the direction the gripper approaches from
  GREEN arrow = Y = the direction the fingers close
  BLUE arrow  = Z

  The gripper grabs whatever sits at the origin, between the fingertips.
""")

o3d.visualization.draw_geometries(
    [gripper_mesh(0.0002),
     o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)],
    window_name="A gripper at the origin", width=900, height=700
)


# ----------------------------------------------------------------------
# PART 2 — A grasp is a transform plus a width
# ----------------------------------------------------------------------
# Here is the whole data structure. It is smaller than people expect:
#
#     position    3 numbers   where the gripper goes
#     rotation    3x3 matrix  which way it is turned
#     width       1 number    how far open the fingers are
#     score       1 number    how good this grasp is (0 to 1)
#
# Position + rotation = 6 numbers of freedom. Hence "6-DoF grasp".

class Grasp:
    def __init__(self, position, rotation, width, score=1.0):
        self.position = np.asarray(position, dtype=float)
        self.rotation = np.asarray(rotation, dtype=float)
        self.width = float(width)
        self.score = float(score)

    @property
    def transform(self):
        """The grasp AS a 4x4 transform - exactly what you built last week."""
        return make_transform(self.rotation, self.position)

    @property
    def approach(self):
        return self.rotation[:, 0]      # first column = where X points now

    @property
    def closing(self):
        return self.rotation[:, 1]      # second column = where Y points now

    def mesh(self, color=(0.1, 0.4, 0.9)):
        g = gripper_mesh(self.width, color)
        g.transform(self.transform)     # <- the whole point of week 2
        return g

    def __repr__(self):
        return (f"Grasp(pos={self.position.round(3)}, "
                f"width={self.width:.3f}, score={self.score:.2f})")


# ----------------------------------------------------------------------
# PART 3 — Grasping a real object
# ----------------------------------------------------------------------

banner("PART 3 — Loading an object  (close window to continue)")

bunny = o3d.data.BunnyMesh()
mesh = o3d.io.read_triangle_mesh(bunny.path)
mesh.compute_vertex_normals()

# Sample points off the surface - this is what a camera would give you.
cloud = mesh.sample_points_poisson_disk(number_of_points=4000)
cloud.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
cloud.normalize_normals()
cloud.paint_uniform_color([0.7, 0.7, 0.7])

pts = np.asarray(cloud.points)
nrm = np.asarray(cloud.normals)

print(f"object   : {len(pts):,} points")
print(f"size     : {(cloud.get_max_bound() - cloud.get_min_bound()).round(3)} m")

o3d.visualization.draw_geometries(
    [cloud], window_name="The object", width=900, height=700)


# ----------------------------------------------------------------------
# PART 4 — Finding grasps: the antipodal idea
# ----------------------------------------------------------------------
# This is the classical method, and understanding it is what makes the
# neural-network version make sense later.
#
# Two fingers hold an object securely when they press on two points
# whose surfaces FACE EACH OTHER. Think of pinching a book: one finger
# on the front cover, one on the back, surfaces pointing at each other.
#
# Bad: both fingers on the same flat face - they would slide off.
#
# So: pick two points. Check their normals point roughly towards each
# other AND roughly along the line joining them. That pair is called
# ANTIPODAL, and it is a grasp candidate.

MAX_WIDTH = 0.05          # the gripper cannot open wider than this
MIN_WIDTH = 0.008
ANGLE_TOL = np.radians(25)  # how much misalignment we tolerate


def find_antipodal_grasps(points, normals, n_samples=20000, rng=None):
    rng = rng or np.random.default_rng(0)
    grasps = []

    a_idx = rng.integers(0, len(points), n_samples)
    b_idx = rng.integers(0, len(points), n_samples)

    for i, j in zip(a_idx, b_idx):
        if i == j:
            continue

        pa, pb = points[i], points[j]
        na, nb = normals[i], normals[j]

        # The line joining the two contact points. The fingers close
        # along this line, so its length is the gripper width.
        between = pb - pa
        width = np.linalg.norm(between)
        if not (MIN_WIDTH < width < MAX_WIDTH):
            continue
        direction = between / width

        # Test 1: does A's surface face towards B?
        #   The normal points OUT of the surface, so for the finger to
        #   press inwards it must come in against the normal.
        angle_a = np.arccos(np.clip(np.dot(na, direction), -1, 1))
        angle_b = np.arccos(np.clip(np.dot(nb, -direction), -1, 1))

        # Both normals should be roughly ANTI-parallel to the closing
        # line (pointing outwards, away from each other).
        if abs(angle_a - np.pi) > ANGLE_TOL:
            continue
        if abs(angle_b - np.pi) > ANGLE_TOL:
            continue

        # Build the grasp frame.
        #   Y = closing direction (the line between the contacts)
        #   X = approach - must be perpendicular to Y. We pick the
        #       direction that comes from above, which is what a robot
        #       over a table would do.
        y_axis = direction
        preferred = np.array([0.0, -1.0, 0.0])          # "from above"
        x_axis = preferred - np.dot(preferred, y_axis) * y_axis
        norm = np.linalg.norm(x_axis)
        if norm < 1e-6:
            continue
        x_axis /= norm
        z_axis = np.cross(x_axis, y_axis)

        rotation = np.column_stack((x_axis, y_axis, z_axis))

        # A simple quality score: how close to perfectly antipodal, and
        # how centred the width is in the gripper's range. Real systems
        # use far more than this - and later, YOUR network will replace
        # this score entirely.
        align = (abs(angle_a - np.pi) + abs(angle_b - np.pi)) / 2
        score = float(np.clip(1.0 - align / ANGLE_TOL, 0, 1))

        grasps.append(Grasp((pa + pb) / 2, rotation, width, score))

    return grasps


banner("PART 4 — Searching for grasps")

grasps = find_antipodal_grasps(pts, nrm)
grasps.sort(key=lambda g: g.score, reverse=True)

print(f"candidates found : {len(grasps):,}")
if grasps:
    print(f"best score       : {grasps[0].score:.3f}")
    print(f"best grasp       : {grasps[0]}")
    print(f"\nIts approach direction : {grasps[0].approach.round(3)}")
    print(f"Its closing direction  : {grasps[0].closing.round(3)}")
    print(f"These are perpendicular: "
          f"{np.dot(grasps[0].approach, grasps[0].closing):.2e}")


# ----------------------------------------------------------------------
# PART 5 — Look at them
# ----------------------------------------------------------------------

banner("PART 5 — The best grasp  (close window to continue)")

best = grasps[0]
contact_a = best.position - best.closing * best.width / 2
contact_b = best.position + best.closing * best.width / 2

dots = o3d.geometry.PointCloud()
dots.points = o3d.utility.Vector3dVector([contact_a, contact_b])
dots.paint_uniform_color([1.0, 0.0, 0.0])

o3d.visualization.draw_geometries(
    [cloud, best.mesh(), dots],
    window_name="Best grasp (red dots = where the fingers touch)",
    width=900, height=700)


banner("PART 5b — The top 25  (close window to continue)")
print("""
  Notice they cluster on the narrow parts - ears, legs, the neck.
  Wide flat regions give no antipodal pairs the gripper can span.
  That is the geometry deciding, with no learning involved at all.
""")

shown = [cloud]
for k, g in enumerate(grasps[:25]):
    t = k / 25
    shown.append(g.mesh(color=(0.1 + 0.8 * t, 0.4, 0.9 - 0.7 * t)))

o3d.visualization.draw_geometries(
    shown, window_name="Top 25 grasp candidates", width=900, height=700)


# ----------------------------------------------------------------------

banner("DONE")
print("""
WHAT YOU NOW KNOW

  A 6-DoF grasp is a position, a rotation, a width and a score.
  The rotation is just a frame: X approaches, Y closes.
  Placing a gripper in space is a 4x4 transform - week 2's work.
  Antipodal pairs are the classical way to find grasp candidates.

WHY THIS MATTERS FOR YOUR PROJECT

  Everything above is pure geometry. No learning. It works on a clean,
  complete 3D model of a single object.

  Reality is harder: a real camera sees ONE SIDE of a cluttered pile,
  with holes where the depth sensor failed. The back of every object is
  invisible. Antipodal search on that data produces many grasps that
  look fine geometrically and fail physically.

  That gap is why learned methods exist - and it is exactly why a
  CONFIDENCE estimate matters. Your project's whole argument is that
  the system should know which of these candidates it can trust.

EXERCISES

  1. Change ANGLE_TOL to 10 degrees, then 45. How does the number of
     candidates change? Which gives better-looking grasps?

  2. Change MAX_WIDTH to 0.02. Where do the grasps move to, and why?

  3. The approach direction is hard-coded as "from above". Change
     `preferred` to come from the side. What changes?

  4. Write a filter that rejects any grasp whose gripper mesh would
     pass through the object. (Hint: sample points along the finger
     bodies and check their distance to the cloud with
     o3d.geometry.KDTreeFlann.) This is called a collision check and
     every real system has one.

  5. Save the top 100 grasps to a .npy file. You will want a grasp
     format of your own before the real data arrives.

Commit it. Next week: the real thing - a slice of GraspNet-1Billion,
where the grasps are labelled by physics simulation rather than by a
geometric rule, and the point clouds have all the holes and clutter
of a real camera.
""")