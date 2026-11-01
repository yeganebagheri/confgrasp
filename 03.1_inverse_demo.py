"""
Why -T is not the inverse of T.

Four windows. Each shows the GREY box (where we started) and a
coloured box (where we ended up after trying to undo a transform).

If an inverse is correct, the coloured box lands exactly on the grey
one and you only see grey.

Run:  python inverse_demo.py
"""

import numpy as np
import open3d as o3d


# ----------------------------------------------------------------------
# The building blocks
# ----------------------------------------------------------------------

def rotation_x(degrees):
    t = np.radians(degrees)
    return np.array([[1, 0, 0],
                     [0, np.cos(t), -np.sin(t)],
                     [0, np.sin(t),  np.cos(t)]])


def make_transform(rotation=np.eye(3), translation=np.zeros(3)):
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T


def invert_transform(T):
    """The CORRECT inverse."""
    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.T                 # a rotation's inverse is its transpose
    t_inv = -R_inv @ t          # the move must be rotated too
    return make_transform(rotation=R_inv, translation=t_inv)


def naive_inverse(T):
    """The WRONG inverse - just flip the sign of the translation."""
    R = T[:3, :3]
    t = T[:3, 3]
    return make_transform(rotation=R.T, translation=-t)   # missing the rotation of t


def make_box(color):
    box = o3d.geometry.TriangleMesh.create_box(0.6, 0.3, 0.05)
    box.translate((-0.3, -0.15, -0.025))
    box.paint_uniform_color(color)
    box.compute_vertex_normals()
    return box


def show(title, *transforms_and_colors):
    """Grey box = the original. Each extra box = a box put through transforms."""
    things = [make_box([0.75, 0.75, 0.75]),
              o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)]

    for chain, color in transforms_and_colors:
        box = make_box(color)
        for T in chain:                      # apply each transform in order
            box.transform(T)
        things.append(box)

    print(f"\n  {title}")
    print("  (drag to look around, close the window to continue)")
    o3d.visualization.draw_geometries(things, window_name=title,
                                      width=900, height=700)


# ----------------------------------------------------------------------
# The transform we will try to undo: rotate 50 deg about X, then move
# ----------------------------------------------------------------------

T = make_transform(rotation=rotation_x(50),
                   translation=np.array([0.5, 0.3, 0.0]))

RED    = [0.9, 0.2, 0.2]
ORANGE = [1.0, 0.6, 0.1]
GREEN  = [0.2, 0.8, 0.3]

print("""
=================================================================
We rotate the box 50 degrees about X and move it, then try three
ways of putting it back. The grey box shows where it STARTED.

A correct inverse puts the coloured box exactly on top of the grey
one - so you will see only grey.
=================================================================
""")

# 1. Just the transform, nothing undone.
show("1. After T  (rotated and moved away from grey)",
     ([T], RED))

# 2. Using -T, which is what you tried.
show("2. Undone with  -T   -- WRONG, it flies off somewhere else",
     ([T, -T], RED))

# 3. Using the naive inverse: transpose the rotation, negate t.
show("3. Undone with naive inverse  -- WRONG, close but offset",
     ([T, naive_inverse(T)], ORANGE))

# 4. The correct inverse.
show("4. Undone with the correct inverse  -- RIGHT, lands on grey",
     ([T, invert_transform(T)], GREEN))


# ----------------------------------------------------------------------
# The same thing as numbers
# ----------------------------------------------------------------------

p = np.array([[1.0, 2.0, 3.0]])


def apply_transform(points, T):
    n = points.shape[0]
    homogeneous = np.hstack((points, np.ones((n, 1))))
    return (T @ homogeneous.T).T[:, :3]


moved = apply_transform(p, T)

print("""
=================================================================
THE SAME THING AS NUMBERS
=================================================================""")
print(f"  started at            : {p[0].round(4)}")
print(f"  after T               : {moved[0].round(4)}")
print()
for name, inv in (("-T                  ", -T),
                  ("naive inverse       ", naive_inverse(T)),
                  ("correct inverse     ", invert_transform(T))):
    back = apply_transform(moved, inv)
    err = np.abs(p - back).max()
    verdict = "CORRECT" if err < 1e-10 else "WRONG"
    print(f"  undone with {name}: {back[0].round(4)}   error {err:.2e}   {verdict}")

print("""
WHY THE NAIVE ONE FAILS

  Undoing a transform means undoing the rotation first. But the
  original move happened BEFORE that un-rotation, so the move itself
  has to be rotated into the new frame too.

  That is the whole meaning of   -R.T @ t   instead of   -t .

  Window 3 shows the result: the orientation is right, the position
  is not. That offset is exactly the kind of bug that makes a robot
  gripper close two centimetres away from the object.
=================================================================
""")