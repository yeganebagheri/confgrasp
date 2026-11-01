import numpy as np
from importlib import import_module

#m = import_module("02_projection_by_hand")


# Exercise 1

def apply_transform(points, T):
    """Apply a 4x4 transform to (N,3) points."""
    n = points.shape[0]
    # Add a column of ones -> (N,4). This is the homogeneous trick.
    homogeneous = np.hstack((points, np.ones((n, 1))))
    # Transform every point. The .T are because the matrix expects
    # points as columns, but we store them as rows.
    moved = (T @ homogeneous.T).T
    return moved[:, :3]      # drop the ones column

def make_transform(rotation=np.eye(3), translation=np.zeros(3)):
    """Assemble a 4x4 transform from a 3x3 rotation and a 3-vector move."""
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T

def rotation_x(degrees):
    """Rotation matrix about the x axis (the camera's forward direction)."""
    t = np.radians(degrees)
    return np.array([
        [1, 0, 0],
        [0,  np.cos(t), -np.sin(t)],
        [0,          np.sin(t),         np.cos(t)],
    ])

p = np.array([[1.0, 2.0, 3.0]])
#rotated = apply_transform(p, make_transform(rotation=rotation_x(45)))
#print(np.linalg.norm(p), np.linalg.norm(rotated))   # must be equal

# Exercise 3 — undoing a transform (the important one)


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


# Move half a metre to the right.
shift = make_transform(translation=np.array([0.5, 0.0, 0.0]))
moved = apply_transform(p, shift)
back = apply_transform(moved, -shift)
print(np.abs(p - back).max())   # should be about 1e-16, definitely under 1e-10