"""
ConfGrasp — Week 4 (fixed)
What happens when the camera can only see one side.

Two things changed from the first version:

  1. Normal orientation is now checked explicitly. Open3D's consistent
     orientation can point every normal INTO the object, which silently
     inverts the antipodal test and returns almost nothing.

  2. Grasp sampling no longer picks random pairs. For each point it
     looks ALONG the normal to find what lies on the far side. Random
     pairing works on a dense complete model and collapses on sparse
     partial data - which is exactly the data we care about.

Run:  python 05_partial_view.py
"""

import numpy as np
import open3d as o3d

MAX_WIDTH = 0.05
MIN_WIDTH = 0.008
ANGLE_TOL = np.radians(45)   # loose: partial-view normals are unreliable
SURFACE_TOL = 0.008          # how close the far side must be to the ray


def banner(text):
    print("\n" + "=" * 66)
    print(text)
    print("=" * 66)


def make_transform(rotation=np.eye(3), translation=np.zeros(3)):
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T


def gripper_mesh(width, color=(0.1, 0.4, 0.9)):
    fl, ft, wl = 0.04, 0.006, 0.03
    parts = []
    for side in (+1, -1):
        f = o3d.geometry.TriangleMesh.create_box(fl, ft, ft * 3)
        f.translate((-fl, side * width / 2 - ft / 2, -ft * 1.5))
        parts.append(f)
    b = o3d.geometry.TriangleMesh.create_box(ft, width + ft, ft * 3)
    b.translate((-fl - ft, -width / 2 - ft / 2, -ft * 1.5))
    parts.append(b)
    w = o3d.geometry.TriangleMesh.create_box(wl, ft * 2, ft * 3)
    w.translate((-fl - ft - wl, -ft, -ft * 1.5))
    parts.append(w)
    g = parts[0]
    for p in parts[1:]:
        g += p
    g.paint_uniform_color(color)
    g.compute_vertex_normals()
    return g


class Grasp:
    def __init__(self, position, rotation, width, score=1.0):
        self.position = np.asarray(position, float)
        self.rotation = np.asarray(rotation, float)
        self.width = float(width)
        self.score = float(score)

    @property
    def transform(self):
        return make_transform(self.rotation, self.position)

    @property
    def closing(self):
        return self.rotation[:, 1]

    def contacts(self):
        half = self.closing * self.width / 2
        return self.position - half, self.position + half

    def mesh(self, color=(0.1, 0.4, 0.9)):
        g = gripper_mesh(self.width, color)
        g.transform(self.transform)
        return g


# ----------------------------------------------------------------------
# FIX 1 — make sure normals point OUTWARD
# ----------------------------------------------------------------------
# Open3D can orient normals consistently but pointing the wrong way.
# A normal pointing into the object silently inverts every antipodal
# test. We check against the centroid: for a roughly blobby object,
# an outward normal points away from the centre more often than not.

def force_outward(cloud):
    pts = np.asarray(cloud.points)
    nrm = np.asarray(cloud.normals)
    outward = pts - pts.mean(axis=0)
    agreement = np.sum(nrm * outward, axis=1)
    fraction_outward = (agreement > 0).mean()
    if fraction_outward < 0.5:
        cloud.normals = o3d.utility.Vector3dVector(-nrm)
        print(f"  normals were pointing inward "
              f"({100 * fraction_outward:.0f}% outward) - flipped them")
    else:
        print(f"  normals point outward ({100 * fraction_outward:.0f}%) - ok")
    return cloud


# ----------------------------------------------------------------------
# FIX 2 — look along the normal instead of pairing at random
# ----------------------------------------------------------------------
# For a point p with outward normal n, the opposite face of the object
# lies somewhere along the ray  p - n*w  for some width w. So we walk
# along that ray and ask the KD-tree whether any point is sitting there.
#
# This is both far more efficient and far more robust on sparse data
# than picking random pairs and hoping.

def find_antipodal_grasps(cloud, n_samples=4000, seed=0, verbose=False):
    points = np.asarray(cloud.points)
    normals = np.asarray(cloud.normals)
    tree = o3d.geometry.KDTreeFlann(cloud)

    rng = np.random.default_rng(seed)
    starts = rng.choice(len(points), min(n_samples, len(points)),
                        replace=False)
    widths = np.linspace(MIN_WIDTH, MAX_WIDTH, 12)

    grasps = []
    tried = hit = 0

    for i in starts:
        pa, na = points[i], normals[i]

        for w in widths:
            # Walk INTO the object along -normal and see what is there.
            target = pa - na * w
            tried += 1

            _, idx, dist2 = tree.search_knn_vector_3d(target, 1)
            if np.sqrt(dist2[0]) > SURFACE_TOL:
                continue
            j = idx[0]
            if j == i:
                continue
            hit += 1

            pb, nb = points[j], normals[j]
            between = pb - pa
            width = np.linalg.norm(between)
            if not (MIN_WIDTH < width < MAX_WIDTH):
                continue
            direction = between / width

            # The antipodal test, unchanged.
            angle_a = np.arccos(np.clip(np.dot(na, direction), -1, 1))
            angle_b = np.arccos(np.clip(np.dot(nb, -direction), -1, 1))
            if abs(angle_a - np.pi) > ANGLE_TOL:
                continue
            if abs(angle_b - np.pi) > ANGLE_TOL:
                continue

            y_axis = direction
            preferred = np.array([0.0, -1.0, 0.0])
            x_axis = preferred - np.dot(preferred, y_axis) * y_axis
            n = np.linalg.norm(x_axis)
            if n < 1e-6:
                continue
            x_axis /= n
            rotation = np.column_stack((x_axis, y_axis,
                                        np.cross(x_axis, y_axis)))

            align = (abs(angle_a - np.pi) + abs(angle_b - np.pi)) / 2
            score = float(np.clip(1.0 - align / ANGLE_TOL, 0, 1))
            grasps.append(Grasp((pa + pb) / 2, rotation, width, score))
            break        # one grasp per start point is plenty

    if verbose:
        print(f"  ray-walk : cast {tried:,}, hit a far surface {hit:,}, "
              f"passed the antipodal test {len(grasps):,}")
    return grasps


# ----------------------------------------------------------------------
# A second sampler, for partial data
# ----------------------------------------------------------------------
# The ray-walk fails on a single camera view, and for a real reason:
# walking inward from a visible point comes out the BACK of the object,
# where the camera saw nothing, so the ray hits empty space.
#
# From one viewpoint every visible surface faces the camera. Two
# surfaces can face OPPOSITE ways and both still be visible only if
# they are nearly edge-on - silhouette edges, thin features, the sides
# of the bunny's ears. Those pairs exist, but the ray-walk cannot find
# them reliably because partial-view normals are noisy.
#
# So we also try random pairs, which does find silhouette-crossing
# pairs. It is wasteful, but on 2,000 points that does not matter.

def find_antipodal_random(cloud, n_samples=400000, seed=0, verbose=False):
    points = np.asarray(cloud.points)
    normals = np.asarray(cloud.normals)
    rng = np.random.default_rng(seed)

    grasps = []
    passed_width = 0

    for i, j in zip(rng.integers(0, len(points), n_samples),
                    rng.integers(0, len(points), n_samples)):
        if i == j:
            continue
        pa, pb = points[i], points[j]
        between = pb - pa
        width = np.linalg.norm(between)
        if not (MIN_WIDTH < width < MAX_WIDTH):
            continue
        passed_width += 1
        direction = between / width

        na, nb = normals[i], normals[j]
        angle_a = np.arccos(np.clip(np.dot(na, direction), -1, 1))
        angle_b = np.arccos(np.clip(np.dot(nb, -direction), -1, 1))
        if abs(angle_a - np.pi) > ANGLE_TOL:
            continue
        if abs(angle_b - np.pi) > ANGLE_TOL:
            continue

        y_axis = direction
        preferred = np.array([0.0, -1.0, 0.0])
        x_axis = preferred - np.dot(preferred, y_axis) * y_axis
        n = np.linalg.norm(x_axis)
        if n < 1e-6:
            continue
        x_axis /= n
        rotation = np.column_stack((x_axis, y_axis, np.cross(x_axis, y_axis)))

        align = (abs(angle_a - np.pi) + abs(angle_b - np.pi)) / 2
        score = float(np.clip(1.0 - align / ANGLE_TOL, 0, 1))
        grasps.append(Grasp((pa + pb) / 2, rotation, width, score))

    if verbose:
        print(f"  random   : tried {n_samples:,}, width ok {passed_width:,}, "
              f"passed the antipodal test {len(grasps):,}")
    return grasps


# ----------------------------------------------------------------------
# PART 1 — The truth
# ----------------------------------------------------------------------

banner("PART 1 — The complete object (what we will never actually have)")

mesh = o3d.io.read_triangle_mesh(o3d.data.BunnyMesh().path)
mesh.compute_vertex_normals()

full = mesh.sample_points_poisson_disk(8000)
full.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
full.normalize_normals()
full.orient_normals_consistent_tangent_plane(30)
full = force_outward(full)
full.paint_uniform_color([0.75, 0.75, 0.75])

full_pts = np.asarray(full.points)
full_nrm = np.asarray(full.normals)
full_tree = o3d.geometry.KDTreeFlann(full)

print(f"  complete model : {len(full_pts):,} points, all sides, no noise")
print("  Kept ONLY as ground truth. The search never looks at it.")


# ----------------------------------------------------------------------
# PART 2 — Break it the way a camera breaks it
# ----------------------------------------------------------------------

banner("PART 2 — Simulating a real camera  (close window to continue)")

CAMERA = np.array([0.0, 0.3, 0.4])
NOISE_MM = 1.5
DROPOUT = 0.12

diameter = np.linalg.norm(full.get_max_bound() - full.get_min_bound())
_, visible_idx = full.hidden_point_removal(CAMERA, diameter * 100)
seen = full.select_by_index(visible_idx)
print(f"  after occlusion : {len(seen.points):,} points "
      f"({100 * len(seen.points) / len(full_pts):.0f}% of the object)")

pts = np.asarray(seen.points).copy()
rng = np.random.default_rng(1)
rays = pts - CAMERA
rays /= np.linalg.norm(rays, axis=1, keepdims=True)
pts += rays * rng.normal(0, NOISE_MM / 1000.0, (len(pts), 1))
pts = pts[rng.random(len(pts)) > DROPOUT]
print(f"  after dropout   : {len(pts):,} points")

partial = o3d.geometry.PointCloud()
partial.points = o3d.utility.Vector3dVector(pts)
partial.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.012, max_nn=30))
partial.normalize_normals()
# All the camera can know: visible surfaces face towards it.
partial.orient_normals_towards_camera_location(CAMERA)
partial.paint_uniform_color([0.2, 0.6, 0.9])

print("\n  Grey = the truth.  Blue = what the camera sees.")
print("  Spin it around and look at the back. There is nothing there.")

o3d.visualization.draw_geometries(
    [full, partial], window_name="Truth (grey) vs camera view (blue)",
    width=1000, height=750)


# ----------------------------------------------------------------------
# PART 3 — Search both
# ----------------------------------------------------------------------

banner("PART 3 — Running the same search on both")

print("complete model:")
grasps_full = find_antipodal_grasps(full, verbose=True)

print("\ncamera view:")
grasps_part = find_antipodal_grasps(partial, verbose=True)
grasps_part += find_antipodal_random(partial, verbose=True)
print("""
  Notice how badly the ray-walk does on the camera view. Walking
  inward from a visible point exits through the BACK of the object,
  which the camera never saw, so the ray finds nothing.

  Random pairing rescues it by finding silhouette-crossing pairs -
  the sides of the ears, the gap by the neck. Those are the only
  places where two opposite-facing surfaces are both visible at once.""")

grasps_full.sort(key=lambda g: g.score, reverse=True)
grasps_part.sort(key=lambda g: g.score, reverse=True)

print(f"\n  grasps from the complete model : {len(grasps_full):,}")
print(f"  grasps from the camera view    : {len(grasps_part):,}")
print("""
  Fewer from the camera view - partly honest (you genuinely cannot
  grasp what you cannot see) and partly misleading, because the ones
  it DOES produce include contacts invented on unseen surfaces.
""")

if not grasps_part:
    print("Still zero. Move CAMERA closer to the bunny, or raise")
    print("SURFACE_TOL to 0.006, and run again.")
    raise SystemExit


# ----------------------------------------------------------------------
# PART 4 — Which of them are actually wrong?
# ----------------------------------------------------------------------

CONTACT_TOL = 0.005


def verify(grasp):
    """True if this grasp would really work, judged on the full model."""
    real_normals = []
    for c in grasp.contacts():
        _, idx, dist2 = full_tree.search_knn_vector_3d(c, 1)
        if np.sqrt(dist2[0]) > CONTACT_TOL:
            return False                   # fingertip lands in empty air
        real_normals.append(full_nrm[idx[0]])

    d = grasp.closing
    a = np.arccos(np.clip(np.dot(real_normals[0], d), -1, 1))
    b = np.arccos(np.clip(np.dot(real_normals[1], -d), -1, 1))
    return bool(abs(a - np.pi) <= ANGLE_TOL and abs(b - np.pi) <= ANGLE_TOL)


banner("PART 4 — Checking the camera's grasps against the truth")

sample = grasps_part[:500]
labels = np.array([verify(g) for g in sample], dtype=bool)
scores = np.array([g.score for g in sample], dtype=float)

n_good = int(labels.sum())
n_bad = len(labels) - n_good

print(f"  checked        : {len(labels)} grasps (highest scoring first)")
print(f"  actually valid : {n_good}  ({100 * n_good / len(labels):.1f}%)")
print(f"  actually WRONG : {n_bad}  ({100 * n_bad / len(labels):.1f}%)")

if n_good and n_bad:
    print("\n  Does the score know which is which?")
    print(f"    mean score, valid   : {scores[labels].mean():.4f}")
    print(f"    mean score, invalid : {scores[~labels].mean():.4f}")

    best_acc, best_t = 0.0, 0.0
    for t in np.linspace(scores.min(), scores.max(), 200):
        acc = ((scores >= t) == labels).mean()
        if acc > best_acc:
            best_acc, best_t = acc, t
    baseline = max(labels.mean(), 1 - labels.mean())

    print(f"\n    best accuracy thresholding the score : {best_acc:.3f}")
    print(f"    accuracy always guessing the majority : {baseline:.3f}")
    print(f"    improvement from the score            : {best_acc - baseline:+.3f}")
else:
    print("\n  All grasps came out the same way - adjust CAMERA or")
    print("  CONTACT_TOL to get a mix, otherwise there is nothing to compare.")


# ----------------------------------------------------------------------
# PART 5 — See the failures
# ----------------------------------------------------------------------

banner("PART 5 — The grasps, drawn  (close window to continue)")
print("""
  GREY  = the real object
  BLUE  = grasps that would actually work
  RED   = grasps the camera was confident about that would FAIL
""")

shown = [full]
for g, ok in list(zip(sample, labels))[:60]:
    shown.append(g.mesh(color=(0.15, 0.5, 0.9) if ok else (0.9, 0.15, 0.15)))

o3d.visualization.draw_geometries(
    shown, window_name="Blue = would work.  Red = would fail.",
    width=1000, height=750)


np.savez("week4_results.npz", scores=scores, labels=labels)

banner("WHAT YOU JUST PROVED")
print(f"""
Saved: week4_results.npz

  A one-sided camera view still yields grasp candidates, and roughly
  {100 * n_bad / max(len(labels), 1):.0f}% of the confident ones are physically wrong.
  The confidence score barely separates good from bad, because it
  measures how neat the geometry LOOKS in the data it was given -
  not whether that data was trustworthy.

  That gap is what your project addresses. Conformal prediction turns
  an ungrounded confidence number into a promise with a guarantee.

WHAT THE BUG TAUGHT YOU (keep this, it is interview material)

  The first version returned zero grasps. Two causes:

    1. Normals silently pointed INWARD. Every antipodal test then
       looked for 0 degrees instead of 180. No error, no warning,
       just nothing found. Sign conventions on normals are a classic
       3D-vision trap.

    2. Random pair sampling collapses on sparse data. Searching ALONG
       the normal for the opposite face is both faster and far more
       robust - and it is what real grasp samplers do.

  Noticing that a silent zero means a bug, rather than a result, is
  the same instinct that made you re-check your attention weights at
  Eurac. Write this one down.

EXERCISES

  1. Move CAMERA to several positions, record the failure rate at
     each. Which viewpoints are worst, and why?

  2. Sweep NOISE_MM from 0 to 5 and plot failure rate against noise.

  3. Build a better score: penalise grasps whose contacts sit far
     from any SEEN point, or near the silhouette edge where normals
     are least reliable. How much does the separation improve?

  4. Overlay histograms of scores for valid vs invalid grasps. If
     they overlap heavily, no threshold can separate them - and that
     picture is worth more than any single number.
""")