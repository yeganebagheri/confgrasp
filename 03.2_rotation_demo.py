"""
A simple, readable demonstration of what rotation does.

No point clouds, no room. Just a box and three arrows.

Run:  python rotation_demo.py

Three windows open in turn. Close each one to see the next.
Drag with the left mouse button to look around.
"""

import numpy as np
import open3d as o3d


def rotation_x(degrees):
    t = np.radians(degrees)
    return np.array([[1, 0, 0],
                     [0, np.cos(t), -np.sin(t)],
                     [0, np.sin(t),  np.cos(t)]])


def rotation_y(degrees):
    t = np.radians(degrees)
    return np.array([[ np.cos(t), 0, np.sin(t)],
                     [ 0,         1, 0],
                     [-np.sin(t), 0, np.cos(t)]])


def rotation_z(degrees):
    t = np.radians(degrees)
    return np.array([[np.cos(t), -np.sin(t), 0],
                     [np.sin(t),  np.cos(t), 0],
                     [0,          0,         1]])


def make_box(color):
    """A flat, wide box - easy to see which way it is facing."""
    box = o3d.geometry.TriangleMesh.create_box(
        width=0.6, height=0.3, depth=0.05
    )
    box.translate((-0.3, -0.15, -0.025))   # centre it on the origin
    box.paint_uniform_color(color)
    box.compute_vertex_normals()
    return box


def show(title, rotation=None):
    """Grey box = original. Red box = rotated. Arrows = the axes."""
    original = make_box([0.75, 0.75, 0.75])

    # The arrows: RED = X axis, GREEN = Y axis, BLUE = Z axis
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

    things = [original, axes]

    if rotation is not None:
        rotated = make_box([0.9, 0.2, 0.2])
        rotated.rotate(rotation, center=(0, 0, 0))
        things.append(rotated)

    print(f"\n  {title}")
    print("  (drag with the left mouse button, close the window to continue)")

    o3d.visualization.draw_geometries(
        things, window_name=title, width=900, height=700
    )


print("""
=================================================================
WHAT THE ARROWS MEAN
    RED arrow   = the X axis
    GREEN arrow = the Y axis
    BLUE arrow  = the Z axis

Rotating "about X" means spinning around the RED arrow,
like a chicken on a rotisserie skewer.
=================================================================
""")

show("1. The box, not rotated yet")

show("2. Rotated 45 degrees about X (around the RED arrow)",
     rotation_x(45))

show("3. Rotated 45 degrees about Y (around the GREEN arrow)",
     rotation_y(45))

show("4. Rotated 45 degrees about Z (around the BLUE arrow)",
     rotation_z(45))

print("""
=================================================================
WHAT YOU SHOULD HAVE SEEN

  about X (red arrow)   -> the box tipped forwards/backwards,
                           like nodding your head yes

  about Y (green arrow) -> the box turned left/right,
                           like shaking your head no

  about Z (blue arrow)  -> the box spun flat,
                           like a steering wheel

The box is the SAME SIZE in all four windows. If it looked bigger
or smaller, that is perspective: the part tipped towards you looks
bigger, the part tipped away looks smaller. Rotation never resizes
anything.

NOW EXPERIMENT. Change the 45 to other numbers and re-run:
    10   - barely moved
    90   - exactly on its side
    180  - completely flipped over
    360  - back where it started
=================================================================
""")
