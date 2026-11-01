"""
ConfGrasp - Week 12, step C (laptop)
Pack many GraspNet frames into ONE zip, ready to upload to Google Drive.

Picks frames 0, 25, 50, ..., 250 of scenes 100-104 (55 frames) and
renames them so they don't clash:
    s0100_f0000_color.png, s0100_f0000_depth.png, s0100_f0000_meta.mat, ...

Run:  python 12c_pack_frames.py
Then: drag  frames_pack.zip  into your Drive folder  graspnet_colab
"""

import os
import zipfile

ROOT_DIR = r'C:/Users/yebagheri/Desktop/Robot project/Robot/graspnet'
SCENES = range(100, 105)          # scenes 100..104
FRAMES = range(0, 256, 25)        # frames 0, 25, ..., 250
OUT_ZIP = 'frames_pack.zip'

packed, skipped = 0, 0
with zipfile.ZipFile(OUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as z:
    for s in SCENES:
        kinect = os.path.join(ROOT_DIR, 'scenes', f'scene_{s:04d}', 'kinect')
        for f in FRAMES:
            files = {
                'color.png': os.path.join(kinect, 'rgb', f'{f:04d}.png'),
                'depth.png': os.path.join(kinect, 'depth', f'{f:04d}.png'),
                'meta.mat':  os.path.join(kinect, 'meta', f'{f:04d}.mat'),
            }
            missing = [p for p in files.values() if not os.path.exists(p)]
            if missing:
                print(f"  skip scene {s} frame {f}: missing {missing[0]}")
                skipped += 1
                continue
            for suffix, path in files.items():
                z.write(path, arcname=f's{s:04d}_f{f:04d}_{suffix}')
            packed += 1

size_mb = os.path.getsize(OUT_ZIP) / 1e6
print(f"\npacked {packed} frames, skipped {skipped}  ->  {OUT_ZIP} ({size_mb:.0f} MB)")
print("Now drag frames_pack.zip into Google Drive > graspnet_colab")