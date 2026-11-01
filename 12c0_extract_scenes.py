"""
ConfGrasp - Week 12, step C0 (laptop)
Extract ONLY the files week 12 needs for scenes 101-104 from test_seen.zip,
straight into graspnet/scenes/..., without unpacking the whole 20 GB zip.

For each scene it extracts:
    kinect/rgb/XXXX.png, kinect/depth/XXXX.png, kinect/meta/XXXX.mat
    for frames 0, 25, ..., 250   (plus object_id_list.txt)

Run:  python 12c0_extract_scenes.py
Then: python 12c_pack_frames.py   (again)
"""

import os
import re
import zipfile

ROOT_DIR = r'C:/Users/yebagheri/Desktop/Robot project/Robot/graspnet'
# Where is your downloaded test_seen.zip? Change this if it is elsewhere.
ZIP_PATH = r'C:/Users/yebagheri/Desktop/Robot project/test_seen.zip'

SCENES = {101, 102, 103, 104}
FRAMES = {f'{f:04d}' for f in range(0, 256, 25)}

if not os.path.exists(ZIP_PATH):
    raise SystemExit(f"Can't find {ZIP_PATH}\n"
                     f"Search your PC for test_seen.zip and put its path in ZIP_PATH.")

# matches e.g.  .../scene_0101/kinect/rgb/0025.png   or  .../scene_0101/object_id_list.txt
pat_frame = re.compile(r'scene_(\d{4})/kinect/(rgb|depth|meta)/(\d{4})\.(png|mat)$')
pat_list = re.compile(r'scene_(\d{4})/object_id_list\.txt$')

extracted = 0
with zipfile.ZipFile(ZIP_PATH) as z:
    for info in z.infolist():
        name = info.filename.replace('\\', '/')
        m = pat_frame.search(name)
        if m:
            scene, kind, frame, ext = int(m.group(1)), m.group(2), m.group(3), m.group(4)
            if scene not in SCENES or frame not in FRAMES:
                continue
            rel = f'scenes/scene_{scene:04d}/kinect/{kind}/{frame}.{ext}'
        else:
            m = pat_list.search(name)
            if not m or int(m.group(1)) not in SCENES:
                continue
            rel = f'scenes/scene_{int(m.group(1)):04d}/object_id_list.txt'

        out = os.path.join(ROOT_DIR, rel)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with z.open(info) as src, open(out, 'wb') as dst:
            dst.write(src.read())
        extracted += 1

expected = len(SCENES) * (len(FRAMES) * 3 + 1)
print(f"extracted {extracted} files (expected {expected})")
if extracted < expected:
    print("Some files were not found in the zip - send me this output.")


# ----------------------------------------------------------------------
# Collision labels for the same scenes (from collision_label.zip)
# ----------------------------------------------------------------------
COLLISION_ZIP = r'C:/Users/yebagheri/Desktop/Robot project/collision_label.zip'

if not os.path.exists(COLLISION_ZIP):
    print(f"\nCan't find {COLLISION_ZIP} - put its real path in COLLISION_ZIP.")
else:
    pat_coll = re.compile(r'scene_(\d{4})/collision_labels\.npz$')
    n_coll = 0
    with zipfile.ZipFile(COLLISION_ZIP) as z:
        for info in z.infolist():
            m = pat_coll.search(info.filename.replace('\\', '/'))
            if not m or int(m.group(1)) not in SCENES:
                continue
            out = os.path.join(ROOT_DIR, 'collision_label',
                               f'scene_{int(m.group(1)):04d}', 'collision_labels.npz')
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with z.open(info) as src, open(out, 'wb') as dst:
                dst.write(src.read())
            n_coll += 1
    print(f"collision labels extracted: {n_coll} (expected {len(SCENES)})")