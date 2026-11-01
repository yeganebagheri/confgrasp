import os
import numpy as np
from PIL import Image
import scipy.io as scio
import open3d as o3d
from graspnetAPI import GraspNet
from graspnetAPI.utils.utils import generate_views
from graspnetAPI.utils.rotation import viewpoint_params_to_matrix
from grasp_lib import Grasp

# Root directory pointing to your downloaded graspnet folder
ROOT_DIR = r'C:/Users/yebagheri/Desktop/Robot/graspnet'

target_scene_id = 100
target_frame_id = 0

# ==========================================
# 1. Load Point Cloud & Scene Meta for Scene 100
# ==========================================
scene_folder = os.path.join(ROOT_DIR, 'scenes', f'scene_{target_scene_id:04d}', 'kinect')
color_path = os.path.join(scene_folder, 'rgb', f'{target_frame_id:04d}.png')
depth_path = os.path.join(scene_folder, 'depth', f'{target_frame_id:04d}.png')
meta_path  = os.path.join(scene_folder, 'meta', f'{target_frame_id:04d}.mat')

color_rgb = np.array(Image.open(color_path), dtype=np.float32) / 255.0
depth_mm = np.array(Image.open(depth_path))
meta = scio.loadmat(meta_path)

intrinsic_matrix = meta['intrinsic_matrix']
factor_depth = float(meta['factor_depth'].item()) 
fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]

def depth_to_pointcloud(depth_arr, color_arr, fx, fy, cx, cy, depth_scale=1000.0, depth_trunc=1.2):
    height, width = depth_arr.shape
    v_grid, u_grid = np.indices((height, width))
    Z = depth_arr.astype(np.float64) / depth_scale
    X = (u_grid - cx) * Z / fx
    Y = (v_grid - cy) * Z / fy
    points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
    colors = color_arr.reshape(-1, 3).astype(np.float64)
    keep = (Z.reshape(-1) > 0.2) & (Z.reshape(-1) < depth_trunc)
    return points[keep], colors[keep]

my_points, my_colors = depth_to_pointcloud(
    depth_mm, color_rgb, fx, fy, cx, cy, 
    depth_scale=factor_depth, depth_trunc=1.2
)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(my_points)
pcd.colors = o3d.utility.Vector3dVector(my_colors)

# ==========================================
# 2. Load Scene-Level Collision Labels Safely
# ==========================================
scene_collision_path = os.path.join(ROOT_DIR, 'collision_label', f'scene_{target_scene_id:04d}', 'collision_labels.npz')
print(f"Loading scene-level collision labels from {scene_collision_path}...")
scene_collision_data = np.load(scene_collision_path)
print("Available keys in scene collision file:", scene_collision_data.files)

# ==========================================
# 3. Parse Object Labels & Vectorized Filtering (Fast Array Ops)
# ==========================================
cls_indexes = meta['cls_indexes'].flatten()  # 1-indexed model categories
poses = meta['poses']                        # (3, 4, num_objects)
views = generate_views(300)                  # shape (300, 3)

all_scene_candidates = []

print("Extracting and filtering object grasps efficiently...")
for obj_idx, cls_id in enumerate(cls_indexes):
    file_id = cls_id - 1  # 0-indexed file mapping
    label_file = os.path.join(ROOT_DIR, 'grasp_label', f'{file_id:03d}_labels.npz')
    if not os.path.exists(label_file):
        continue
        
    data = np.load(label_file)
    points = data['points']         # (N, 3)
    scores = data['scores']         # (N, 300, 12, 4)
    object_collision = data['collision'] # (N, 300, 12, 4)
    offsets = data['offsets']       # (N, 300, 12, 4, 3)

    # Strict key check using exact object index key from scene collision file
    arr_key = f'arr_{obj_idx}'
    scene_collision = scene_collision_data[arr_key]

    valid_mask = (scores > 0) & (scores <= 0.2) & (object_collision == 0) & (scene_collision == 0)
    if not np.any(valid_mask):
        continue
        
    n_idx, s_idx, a_idx, t_idx = np.where(valid_mask)
    valid_scores = scores[n_idx, s_idx, a_idx, t_idx]

    # Keep only the best few per object using argsort for extreme speed
    best_local_idxs = np.argsort(valid_scores)[:10]
    
    for k in best_local_idxs:
        all_scene_candidates.append({
            'score': float(valid_scores[k]),
            'obj_idx': obj_idx,
            'n_idx': n_idx[k],
            's_idx': s_idx[k],
            'a_idx': a_idx[k],
            't_idx': t_idx[k],
            'point': points[n_idx[k]],
            'offset': offsets[n_idx[k], s_idx[k], a_idx[k], t_idx[k]]
        })

all_scene_candidates.sort(key=lambda x: x['score'])
top_20_candidates = all_scene_candidates[:20]

# ==========================================
# 4. Compute Rotations & Transformations
# ==========================================
manual_grasps = []
for item in top_20_candidates:
    obj_idx = item['obj_idx']
    s_idx = item['s_idx']
    angle_val = item['offset'][0]
    width_val = item['offset'][2]
    
    local_trans = item['point']
    approach_vec = -views[s_idx]
    local_rot = viewpoint_params_to_matrix(approach_vec, angle_val)

    pose_3x4 = poses[:, :, obj_idx]
    T_cam_obj = np.eye(4)
    T_cam_obj[:3, :4] = pose_3x4
    R_pose = T_cam_obj[:3, :3]
    t_pose = T_cam_obj[:3, 3]

    cam_trans = R_pose @ local_trans + t_pose
    cam_rot = R_pose @ local_rot

    g_manual = Grasp(
        position=cam_trans,
        rotation=cam_rot,
        width=width_val,
        score=item['score']
    )
    manual_grasps.append(g_manual)

# ==========================================
# 5. Numerical Verification Test vs Official API
# ==========================================
import graspnetAPI.graspnet as gnet

def _getObjIds(self, sceneIds=None):
    if sceneIds is None:
        return self.objIds
    if not hasattr(sceneIds, '__len__'):
        sceneIds = [sceneIds]
    obj_ids = set()
    for i in sceneIds:
        p = os.path.join(self.root, 'scenes', f'scene_{i:04d}', 'object_id_list.txt')
        if os.path.exists(p):              # skip scenes you don't have
            with open(p) as f:
                obj_ids.update(int(line.strip()) for line in f if line.strip())
    return list(obj_ids)

gnet.GraspNet.getObjIds = _getObjIds

net = GraspNet(root=ROOT_DIR, camera='kinect', split='test_seen')
api_grasps = net.loadGrasp(sceneId=target_scene_id, annId=target_frame_id, format='6d', camera='kinect', fric_coef_thresh=0.2)

api_scores = api_grasps.scores
print(f"\nAPI Grasp Scores -> Min: {api_scores.min():.3f}, Max: {api_scores.max():.3f}")

api_translations = api_grasps.translations
api_rotations = api_grasps.rotation_matrices

print("\n==============================================")
print("     NUMERICAL VERIFICATION TEST (API vs Manual)")
print("==============================================")
for i, g_man in enumerate(manual_grasps):
    man_t = g_man.position
    man_R = g_man.rotation
    
    # Compute distances to all API grasps
    distances = np.linalg.norm(api_translations - man_t, axis=1)
    
    # Filter candidates closer than 1 mm (0.001 m) to resolve multi-grasp spatial ambiguity
    close_indices = np.where(distances < 0.001)[0]
    
    if len(close_indices) > 0:
        # Among those within 1 mm, pick the one with minimum rotation error
        best_close_idx = close_indices[0]
        min_dist = distances[best_close_idx]
        best_rot_diff = float('inf')
        
        for idx in close_indices:
            api_R = api_rotations[idx]
            trace_val = np.clip(np.trace(api_R.T @ man_R), -1.0, 3.0)
            rot_diff_rad = np.arccos(np.clip((trace_val - 1.0) / 2.0, -1.0, 1.0))
            if rot_diff_rad < best_rot_diff:
                best_rot_diff = rot_diff_rad
                best_close_idx = idx
        
        final_dist = distances[best_close_idx]
        final_api_R = api_rotations[best_close_idx]
        trace_val = np.clip(np.trace(final_api_R.T @ man_R), -1.0, 3.0)
        final_rot_diff = np.arccos(np.clip((trace_val - 1.0) / 2.0, -1.0, 1.0))
    else:
        # Fallback to absolute nearest if none are under 1 mm
        nearest_idx = np.argmin(distances)
        final_dist = distances[nearest_idx]
        api_R = api_rotations[nearest_idx]
        trace_val = np.clip(np.trace(api_R.T @ man_R), -1.0, 3.0)
        final_rot_diff = np.arccos(np.clip((trace_val - 1.0) / 2.0, -1.0, 1.0))
    
    print(f"Manual Grasp {i:2d} -> Nearest API Grasp: Trans Error = {final_dist*1000:6.2f} mm | Rot Error = {np.degrees(final_rot_diff):6.2f}°")
print("==============================================\n")