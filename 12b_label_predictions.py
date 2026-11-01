"""
ConfGrasp - Week 12, part B
Label every grasp the network predicted as CORRECT or WRONG,
using the GraspNet answer key you built in week 11.

Rule (a simplification of GraspNet's physics-based evaluation):
  a prediction is correct if some valid answer-key grasp is
    - closer than 1 cm in position, AND
    - within 30 degrees in rotation
  (a parallel gripper turned 180 degrees around its approach axis
   is the same grasp, so that flip counts as a match)

Output: week12_calib.npz with `scores` and `correct` -> week 13 calibration.

Run:  python 12b_label_predictions.py
"""

import os
import numpy as np
import scipy.io as scio
from scipy.spatial import cKDTree
from graspnetAPI import GraspGroup
from graspnetAPI.utils.utils import generate_views
from graspnetAPI.utils.rotation import batch_viewpoint_params_to_matrix

ROOT_DIR = r'C:/Users/yebagheri/Desktop/Robot project/Robot/graspnet'
PRED_FILE = r'C:/Users/yebagheri/Desktop/Robot/pred_scene100_0000_medium.npy'
SCENE_ID, FRAME_ID = 100, 0

DIST_TOL = 0.01              # 1 cm
ROT_TOL = np.radians(30)     # 30 degrees
FRIC_MAX = 0.2               # valid label: 0 < friction <= 0.2


# ----------------------------------------------------------------------
# 1. The network's predictions (from Colab)
# ----------------------------------------------------------------------
gg = GraspGroup().from_npy(PRED_FILE)
pred_t = gg.translations          # (M, 3)  camera frame
pred_R = gg.rotation_matrices     # (M, 3, 3)
pred_s = gg.scores                # (M,)    network's own score, higher = better
print(f"predictions loaded : {len(gg)}")


# ----------------------------------------------------------------------
# 2. The answer key (your week 11 code, stored compactly)
# ----------------------------------------------------------------------
# Instead of building a rotation for every valid label grasp (millions),
# we keep only their indices, and build rotations later just for the
# few labels that are near each prediction.

scene_dir = os.path.join(ROOT_DIR, 'scenes', f'scene_{SCENE_ID:04d}', 'kinect')
meta = scio.loadmat(os.path.join(scene_dir, 'meta', f'{FRAME_ID:04d}.mat'))
cls_indexes = meta['cls_indexes'].flatten()
poses = meta['poses']                                   # (3, 4, n_objects)
scene_coll = np.load(os.path.join(ROOT_DIR, 'collision_label',
                                  f'scene_{SCENE_ID:04d}', 'collision_labels.npz'))
views = generate_views(300)

objects = []
for obj_idx, cls_id in enumerate(cls_indexes):
    data = np.load(os.path.join(ROOT_DIR, 'grasp_label', f'{cls_id - 1:03d}_labels.npz'))
    scores = data['scores']
    valid = ((scores > 0) & (scores <= FRIC_MAX)
             & (data['collision'] == 0)
             & (scene_coll[f'arr_{obj_idx}'] == 0))

    n_idx, v_idx, a_idx, d_idx = np.where(valid)        # n_idx comes out sorted
    angles = data['offsets'][n_idx, v_idx, a_idx, d_idx, 0]

    R_pose = poses[:, :3, obj_idx]
    t_pose = poses[:, 3, obj_idx]
    pts_cam = data['points'] @ R_pose.T + t_pose        # seed points -> camera frame

    objects.append(dict(tree=cKDTree(pts_cam), n_idx=n_idx, v_idx=v_idx,
                        angles=angles, R_pose=R_pose))
    print(f"  object {cls_id - 1:3d}: {len(n_idx):>10,} valid label grasps")


# ----------------------------------------------------------------------
# 3. Compare each prediction with nearby labels
# ----------------------------------------------------------------------
FLIP = np.diag([1.0, -1.0, -1.0])     # 180 deg turn around the approach (x) axis


def rotation_error(R_labels, R_pred):
    """Smallest angle between R_pred (or its flip) and each label rotation."""
    best = None
    for Rp in (R_pred, R_pred @ FLIP):
        trace = np.einsum('kij,ij->k', R_labels, Rp)     # = trace(R_label^T @ Rp)
        err = np.arccos(np.clip((trace - 1) / 2, -1, 1))
        best = err if best is None else np.minimum(best, err)
    return best


best_err = np.full(len(gg), np.pi)       # best rotation error found (radians)
has_near = np.zeros(len(gg), bool)       # any valid label within 1 cm at all?

for m in range(len(gg)):
    for ob in objects:
        near_pts = ob['tree'].query_ball_point(pred_t[m], DIST_TOL)
        if not near_pts:
            continue
        # all valid label grasps that sit on those nearby seed points
        lo = np.searchsorted(ob['n_idx'], near_pts, 'left')
        hi = np.searchsorted(ob['n_idx'], near_pts, 'right')
        sel = np.concatenate([np.arange(a, b) for a, b in zip(lo, hi)])
        if len(sel) == 0:
            continue
        has_near[m] = True

        R_local = batch_viewpoint_params_to_matrix(-views[ob['v_idx'][sel]],
                                                   ob['angles'][sel])
        R_labels = ob['R_pose'] @ R_local                # object -> camera frame
        best_err[m] = min(best_err[m], rotation_error(R_labels, pred_R[m]).min())

correct = best_err < ROT_TOL


# ----------------------------------------------------------------------
# 4. Results
# ----------------------------------------------------------------------
print(f"\n  predictions with a valid label within 1 cm : {has_near.sum()} / {len(gg)}")
print(f"  correct (1 cm AND 30 deg)                  : {correct.sum()} / {len(gg)}"
      f"  ({100 * correct.mean():.1f}%)")

# 5 bins with the same number of predictions each (we only have ~100)
edges = np.quantile(pred_s, np.linspace(0, 1, 6))
bin_id = np.clip(np.searchsorted(edges, pred_s, 'right') - 1, 0, 4)

print(f"\n  {'score range':<17} {'count':>6} {'% correct':>10}")
for b in range(5):
    sel = bin_id == b
    if sel.sum() == 0:
        continue
    print(f"  {edges[b]:.3f} - {edges[b + 1]:.3f}   {sel.sum():>6} "
          f"{100 * correct[sel].mean():>9.1f}%")

np.savez('week12_calib.npz', scores=pred_s, correct=correct,
         rot_err_deg=np.degrees(best_err), has_near=has_near)
print("\nSaved: week12_calib.npz")