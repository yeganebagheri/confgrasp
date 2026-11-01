"""
ConfGrasp — shared library.

Everything from weeks 1-4, collected in one place so later scripts can
import it instead of copy-pasting. Refactoring into a module like this
is itself worth doing: it forces you to decide what is reusable.

    from grasp_lib import Grasp, make_scene, find_grasps, verify
"""

import numpy as np
import open3d as o3d

MAX_WIDTH = 0.05
MIN_WIDTH = 0.008
ANGLE_TOL = np.radians(45)
CONTACT_TOL = 0.005


# ----------------------------------------------------------------------
# Transforms and the gripper
# ----------------------------------------------------------------------

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
    def approach(self):
        return self.rotation[:, 0]

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

    def __repr__(self):
        return (f"Grasp(pos={self.position.round(3)}, "
                f"width={self.width:.3f}, score={self.score:.2f})")


# ----------------------------------------------------------------------
# Scene construction
# ----------------------------------------------------------------------

def force_outward(cloud, quiet=True):
    pts = np.asarray(cloud.points)
    nrm = np.asarray(cloud.normals)
    agreement = np.sum(nrm * (pts - pts.mean(axis=0)), axis=1)
    if (agreement > 0).mean() < 0.5:
        cloud.normals = o3d.utility.Vector3dVector(-nrm)
        if not quiet:
            print("  flipped normals outward")
    return cloud


def ground_truth(n_points=8000, seed=42):
    """The complete object. Seeded, so the experiment is reproducible.

    Note: older Open3D builds do not accept a `seed` argument on
    sample_points_poisson_disk, so we seed globally instead. Check that
    repeated runs give identical point counts - if they do not, the
    sampler is ignoring this and your numbers are not reproducible.
    """
    o3d.utility.random.seed(seed)
    np.random.seed(seed)
    mesh = o3d.io.read_triangle_mesh(o3d.data.BunnyMesh().path)
    mesh.compute_vertex_normals()
    cloud = mesh.sample_points_poisson_disk(n_points)
    cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01,
                                                          max_nn=30))
    cloud.normalize_normals()
    cloud.orient_normals_consistent_tangent_plane(30)
    return force_outward(cloud)


# Per-object scale, in metres (longest side).
#
# These are hand-tuned, and that is the point: what matters for
# grasping is the NARROW dimension, not the longest one. A sphere
# scaled to 15 cm is a 15 cm ball - nothing a 5 cm gripper can hold.
# A torus scaled to 15 cm happens to have a 4 cm tube, so every point
# around it is graspable and it floods the dataset.
#
# Curating a dataset so that every object is actually solvable, and
# no single object dominates, is real work. Benchmarks like GraspNet
# do this deliberately with 88 objects chosen for graspability.
OBJECT_SCALES = {
    "bunny": 0.15,        # ears and legs are the graspable parts
    "armadillo": 0.15,
    "knot": 0.12,         # thin tube throughout
    "box": 0.15,          # graspable across its thickness
    "sphere": 0.045,      # must fit entirely inside the gripper
    "cylinder": 0.09,     # diameter ends up ~4.5 cm
    "cone": 0.10,         # graspable near the tip, not the base
    "torus": 0.15,        # tube ends up ~4 cm
    "icosahedron": 0.045,
}


def load_object(name, n_points=8000, target_size=None, seed=42):
    """A complete point cloud of one object, scaled to gripper size.

    target_size defaults to a per-object value from OBJECT_SCALES.
    """
    o3d.utility.random.seed(seed)
    np.random.seed(seed)

    builders = {
        "bunny": lambda: o3d.io.read_triangle_mesh(o3d.data.BunnyMesh().path),
        "armadillo": lambda: o3d.io.read_triangle_mesh(
            o3d.data.ArmadilloMesh().path),
        "knot": lambda: o3d.io.read_triangle_mesh(o3d.data.KnotMesh().path),
        "box": lambda: o3d.geometry.TriangleMesh.create_box(1.0, 0.6, 0.25),
        "sphere": lambda: o3d.geometry.TriangleMesh.create_sphere(0.5, 30),
        "cylinder": lambda: o3d.geometry.TriangleMesh.create_cylinder(
            0.25, 1.0, 30, 10),
        "cone": lambda: o3d.geometry.TriangleMesh.create_cone(0.4, 1.0, 30, 10),
        "torus": lambda: o3d.geometry.TriangleMesh.create_torus(
            0.5, 0.18, 40, 20),
        "icosahedron": lambda: o3d.geometry.TriangleMesh.create_icosahedron(0.5),
    }
    if name not in builders:
        raise ValueError(f"unknown object {name!r}; "
                         f"choose from {sorted(builders)}")

    if target_size is None:
        target_size = OBJECT_SCALES.get(name, 0.12)

    mesh = builders[name]()
    mesh.compute_vertex_normals()

    # Scale so the longest side is target_size, and centre on the origin.
    extent = mesh.get_max_bound() - mesh.get_min_bound()
    mesh.scale(target_size / extent.max(), center=mesh.get_center())
    mesh.translate(-mesh.get_center())

    cloud = mesh.sample_points_poisson_disk(n_points)
    cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01,
                                                          max_nn=30))
    cloud.normalize_normals()
    cloud.orient_normals_consistent_tangent_plane(30)
    return force_outward(cloud)


def camera_view(full, camera, noise_mm=1.5, dropout=0.12, seed=1):
    """What a depth camera at `camera` would actually see."""
    rng = np.random.default_rng(seed)
    diameter = np.linalg.norm(full.get_max_bound() - full.get_min_bound())
    _, visible = full.hidden_point_removal(camera, diameter * 100)
    pts = np.asarray(full.select_by_index(visible).points).copy()

    rays = pts - camera
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    pts += rays * rng.normal(0, noise_mm / 1000.0, (len(pts), 1))
    pts = pts[rng.random(len(pts)) > dropout]

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts)
    cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.012,
                                                          max_nn=30))
    cloud.normalize_normals()
    cloud.orient_normals_towards_camera_location(camera)
    return cloud


# ----------------------------------------------------------------------
# Grasp sampling
# ----------------------------------------------------------------------

def _build(pa, pb, na, nb):
    """Try to build a grasp from two contact points. None if invalid."""
    between = pb - pa
    width = np.linalg.norm(between)
    if not (MIN_WIDTH < width < MAX_WIDTH):
        return None
    direction = between / width

    angle_a = np.arccos(np.clip(np.dot(na, direction), -1, 1))
    angle_b = np.arccos(np.clip(np.dot(nb, -direction), -1, 1))
    if abs(angle_a - np.pi) > ANGLE_TOL or abs(angle_b - np.pi) > ANGLE_TOL:
        return None

    y_axis = direction
    preferred = np.array([0.0, -1.0, 0.0])
    x_axis = preferred - np.dot(preferred, y_axis) * y_axis
    n = np.linalg.norm(x_axis)
    if n < 1e-6:
        return None
    x_axis /= n
    rotation = np.column_stack((x_axis, y_axis, np.cross(x_axis, y_axis)))

    align = (abs(angle_a - np.pi) + abs(angle_b - np.pi)) / 2
    score = float(np.clip(1.0 - align / ANGLE_TOL, 0, 1))
    return Grasp((pa + pb) / 2, rotation, width, score)


def find_grasps(cloud, n_samples=400000, seed=0):
    """Random-pair antipodal sampling. Works on partial data."""
    points = np.asarray(cloud.points)
    normals = np.asarray(cloud.normals)
    rng = np.random.default_rng(seed)

    grasps = []
    for i, j in zip(rng.integers(0, len(points), n_samples),
                    rng.integers(0, len(points), n_samples)):
        if i == j:
            continue
        g = _build(points[i], points[j], normals[i], normals[j])
        if g is not None:
            grasps.append(g)
    return grasps


# ----------------------------------------------------------------------
# Ground-truth verification
# ----------------------------------------------------------------------

def make_verifier(full):
    """Returns a function that marks a grasp valid or invalid."""
    tree = o3d.geometry.KDTreeFlann(full)
    normals = np.asarray(full.normals)

    def verify(grasp):
        real = []
        for c in grasp.contacts():
            _, idx, dist2 = tree.search_knn_vector_3d(c, 1)
            if np.sqrt(dist2[0]) > CONTACT_TOL:
                return False
            real.append(normals[idx[0]])
        d = grasp.closing
        a = np.arccos(np.clip(np.dot(real[0], d), -1, 1))
        b = np.arccos(np.clip(np.dot(real[1], -d), -1, 1))
        return bool(abs(a - np.pi) <= ANGLE_TOL and abs(b - np.pi) <= ANGLE_TOL)

    return verify