# ConfGrasp — Week 1 Setup (Windows / HP ProBook)

Goal for this week: a working Python environment, and a 3D point cloud
rendered on your screen that you can rotate with the mouse.

No GPU needed. No hardware needed. Budget: €0.

---

## Step 1 — Install Python 3.11

Open3D is reliable on Python 3.11. Newer versions sometimes lag.

Download from: https://www.python.org/downloads/release/python-3119/
Choose "Windows installer (64-bit)".

**IMPORTANT during install:** tick the box "Add python.exe to PATH" on the
first screen. If you miss it you will have path problems all week.

Verify in PowerShell:

```powershell
python --version
```

You should see `Python 3.11.9`.

---

## Step 2 — Install Git

Download: https://git-scm.com/download/win

Accept all defaults. Verify:

```powershell
git --version
```

---

## Step 3 — Create the project folder and virtual environment

A virtual environment keeps this project's packages separate from everything
else on your laptop. Always work inside it.

```powershell
mkdir C:\projects\confgrasp
cd C:\projects\confgrasp

python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell refuses with an execution-policy error, run this once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

then activate again.

When the environment is active your prompt starts with `(.venv)`.
**You must activate it every time you open a new terminal.**

---

## Step 4 — Install the packages

```powershell
python -m pip install --upgrade pip
python -m pip install open3d numpy matplotlib
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

That last line installs the CPU build of PyTorch, which is correct for your
laptop. You will use a rented GPU later; nothing about your code changes.

Verify everything imported correctly:

```powershell
python -c "import open3d, torch, numpy; print(open3d.__version__, torch.__version__)"
```

---

## Step 5 — Run the first script

Put `01_first_pointcloud.py` in `C:\projects\confgrasp` and run:

```powershell
python 01_first_pointcloud.py
```

A window opens showing a 3D point cloud of a room.

- Left mouse drag = rotate
- Scroll = zoom
- Right mouse drag = pan
- Press `Q` or close the window to continue to the next step

The script runs four steps in sequence, each opening a window. Close each one
to move on.

---

## Step 6 — Start the repo

```powershell
git init
git add .
git commit -m "Week 1: point cloud basics"
```

Create an empty repository on GitHub named `confgrasp`, then:

```powershell
git remote add origin https://github.com/YOUR-USERNAME/confgrasp.git
git branch -M main
git push -u origin main
```

Start the public repo now, not at the end. Employers look at commit history,
and a project with nine months of steady commits reads very differently from
one uploaded in a single day.

---

## What "done" looks like this week

You can answer these three questions in your own words:

1. What is a point cloud, and how is it different from a depth image?
2. What are camera intrinsics (fx, fy, cx, cy), and what goes wrong if they
   are wrong?
3. Why do we downsample a point cloud before processing it?

If you can answer those, Week 1 is finished. That is the whole goal — not
speed, understanding.

---

## Troubleshooting

**The visualization window is black or does not open.**
Update your Intel graphics driver from HP's support site. Open3D needs OpenGL
3.3. If it still fails, add `print(pcd)` after each step and check the point
counts are non-zero — the maths works even if the display does not.

**`ModuleNotFoundError: No module named 'open3d'`**
Your virtual environment is not active. Look for `(.venv)` in your prompt.

**Install is extremely slow.**
Open3D is a large package (~400 MB). This is normal. Use a wired connection
if you can.

---

## Next week (preview, do not start yet)

Week 2 is the camera model in depth: writing the projection maths yourself
with numpy, without Open3D's helper, so you understand what the helper does.
That is the foundation everything else sits on.
