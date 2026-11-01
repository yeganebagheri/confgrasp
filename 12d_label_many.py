"""
ConfGrasp - Week 12, step D (laptop)
Label the predictions from MANY frames as correct / wrong, and save
everything into one week12_calib.npz for calibration (week 13).

Same rule as 12b: correct = a valid label grasp within 1 cm AND 30 deg
(180-degree flip around the approach axis counts as the same grasp).

Before running: unzip preds_pack.zip into a folder called  preds
next to this script, so you have  preds/s0100_f0000.npy, ...

Run:  python 12d_label_many.py
"""

import os
import re
import glob
import numpy as np
import scipy.io as scio
from scipy.spatial import cKDTree
from graspnetAPI import GraspGroup
from graspnetAPI.utils.utils import generate_views
from graspnetAPI.utils.rotation import batch_viewpoint_params_to_matrix

ROOT_DIR = r'C:/Users/yebagheri/Desktop/Robot project/Robot/graspnet'
PRED_DIR = 'preds'

DIST_TOL = 0.01
ROT_TOL = np.radians(30)
FRIC_MAX = 0.2
FLIP = np.diag([1.0, -1.0, -1.0])
views = generate_views(300)


# ----------------------------------------------------------------------
# Answer key, part 1: per SCENE (labels + scene collisions, object frame)
# ----------------------------------------------------------------------
_scene_cache = {}


def scene_labels(scene_id):
    """Valid label grasps for every object in the scene, in OBJECT frame.

    Collisions are stored per scene (the objects don't move between
    frames), so this is computed once per scene and reused for all frames.
    """
    if scene_id in _scene_cache:
        return _scene_cache[scene_id]

    kinect = os.path.join(ROOT_DIR, 'scenes', f'scene_{scene_id:04d}', 'kinect')
    meta0 = scio.loadmat(os.path.join(kinect, 'meta', '0000.mat'))
    cls_order = meta0['cls_indexes'].flatten()      # order used by collision file
    coll = np.load(os.path.join(ROOT_DIR, 'collision_label',
                                f'scene_{scene_id:04d}', 'collision_labels.npz'))

    labels = {}
    for i, cls_id in enumerate(cls_order):
        data = np.load(os.path.join(ROOT_DIR, 'grasp_label', f'{cls_id - 1:03d}_labels.npz'))
        s = data['scores']
        valid = ((s > 0) & (s <= FRIC_MAX) & (data['collision'] == 0)
                 & (coll[f'arr_{i}'] == 0))
        n_idx, v_idx, a_idx, d_idx = np.where(valid)
        labels[int(cls_id)] = dict(
            points=data['points'],
            n_idx=n_idx, v_idx=v_idx,
            angles=data['offsets'][n_idx, v_idx, a_idx, d_idx, 0])
    _scene_cache[scene_id] = labels
    print(f"  scene {scene_id}: labels loaded for {len(labels)} objects")
    return labels


# ----------------------------------------------------------------------
# Answer key, part 2: per FRAME (move labels into this camera view)
# ----------------------------------------------------------------------
def frame_objects(scene_id, frame_id):
    labels = scene_labels(scene_id)
    meta = scio.loadmat(os.path.join(ROOT_DIR, 'scenes', f'scene_{scene_id:04d}',
                                     'kinect', 'meta', f'{frame_id:04d}.mat'))
    objs = []
    for j, cls_id in enumerate(meta['cls_indexes'].flatten()):
        lab = labels[int(cls_id)]
        R_pose = meta['poses'][:, :3, j]
        t_pose = meta['poses'][:, 3, j]
        objs.append(dict(tree=cKDTree(lab['points'] @ R_pose.T + t_pose),
                         R_pose=R_pose, **lab))
    return objs


def rotation_error(R_labels, R_pred):
    best = None
    for Rp in (R_pred, R_pred @ FLIP):
        tr = np.einsum('kij,ij->k', R_labels, Rp)
        err = np.arccos(np.clip((tr - 1) / 2, -1, 1))
        best = err if best is None else np.minimum(best, err)
    return best


def label_frame(pred_file, scene_id, frame_id):
    gg = GraspGroup().from_npy(pred_file)
    if len(gg) == 0:
        return np.zeros(0), np.zeros(0, bool), np.zeros(0)
    objs = frame_objects(scene_id, frame_id)
    best_err = np.full(len(gg), np.pi)
    for m in range(len(gg)):
        t, R = gg.translations[m], gg.rotation_matrices[m]
        for ob in objs:
            near = ob['tree'].query_ball_point(t, DIST_TOL)
            if not near:
                continue
            lo = np.searchsorted(ob['n_idx'], near, 'left')
            hi = np.searchsorted(ob['n_idx'], near, 'right')
            sel = np.concatenate([np.arange(a, b) for a, b in zip(lo, hi)])
            if len(sel) == 0:
                continue
            R_labels = ob['R_pose'] @ batch_viewpoint_params_to_matrix(
                -views[ob['v_idx'][sel]], ob['angles'][sel])
            best_err[m] = min(best_err[m], rotation_error(R_labels, R).min())
    return gg.scores, best_err < ROT_TOL, best_err


# ----------------------------------------------------------------------
# Run over every prediction file
# ----------------------------------------------------------------------
pred_files = sorted(glob.glob(os.path.join(PRED_DIR, 's*_f*.npy')))
print(f"prediction files: {len(pred_files)}")

all_s, all_c, all_e, all_scene, all_frame = [], [], [], [], []
print(f"\n  {'frame':<13} {'grasps':>7} {'% correct':>10}")
for pf in pred_files:
    m = re.search(r's(\d+)_f(\d+)\.npy$', pf)
    scene_id, frame_id = int(m.group(1)), int(m.group(2))
    s, c, e = label_frame(pf, scene_id, frame_id)
    all_s.append(s); all_c.append(c); all_e.append(e)
    all_scene.append(np.full(len(s), scene_id)); all_frame.append(np.full(len(s), frame_id))
    pct = f"{100 * c.mean():9.1f}%" if len(c) else "      -"
    print(f"  s{scene_id:04d}_f{frame_id:04d}  {len(s):>7} {pct:>10}")

scores = np.concatenate(all_s)
correct = np.concatenate(all_c)
rot_err = np.concatenate(all_e)
scene = np.concatenate(all_scene)
frame = np.concatenate(all_frame)


# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------
print(f"\nTOTAL: {len(scores):,} grasps, {100 * correct.mean():.1f}% correct")

print(f"\n  per scene:")
for sid in np.unique(scene):
    sel = scene == sid
    print(f"    scene {sid}: {sel.sum():>6,} grasps  {100 * correct[sel].mean():5.1f}% correct")

edges = np.quantile(scores, np.linspace(0, 1, 11))          # 10 equal-count bins
bin_id = np.clip(np.searchsorted(edges, scores, 'right') - 1, 0, 9)
print(f"\n  {'score range':<17} {'count':>6} {'% correct':>10}")
for b in range(10):
    sel = bin_id == b
    if sel.sum():
        print(f"  {edges[b]:.3f} - {edges[b + 1]:.3f}   {sel.sum():>6} "
              f"{100 * correct[sel].mean():>9.1f}%")

np.savez('week12_calib.npz', scores=scores, correct=correct,
         rot_err_deg=np.degrees(rot_err), scene=scene, frame=frame)
print("\nSaved: week12_calib.npz  (scores, correct, rot_err_deg, scene, frame)")