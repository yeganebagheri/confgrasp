"""
ConfGrasp — Week 6
Split by OBJECT, not by viewpoint.

You spotted the flaw yourself: eight cameras pointed at one bunny does
not test generalisation. The model can memorise that one shape and
still do well on a seventh angle of it.

The question Fizyr actually cares about is different: given an object
the system has NEVER SEEN, can it still tell a good grasp from a bad
one? GraspNet is built around exactly this - its test set is divided
into seen, similar and novel objects, because object-level
generalisation is the hard part and viewpoint generalisation is not.

This script trains on five objects and tests on four different ones.
It also adds early stopping, which last week badly needed.

Run:  python 07_unseen_objects.py
"""

import numpy as np
import open3d as o3d
import torch
import torch.nn as nn

from grasp_lib import load_object, camera_view, find_grasps, make_verifier


def banner(text):
    print("\n" + "=" * 66)
    print(text)
    print("=" * 66)


torch.manual_seed(0)
np.random.seed(0)

# Organic shapes are irregular and hard; primitives are clean and
# symmetric. Mixing both means the test set is not trivially similar
# to the training set.
TRAIN_OBJECTS = ["bunny", "box", "cylinder", "torus"]
VAL_OBJECTS = ["icosahedron"]          # for early stopping only
TEST_OBJECTS = ["armadillo", "knot", "sphere", "cone"]

CAMERAS = [
    np.array([0.0, 0.30, 0.40]), np.array([0.35, 0.25, 0.25]),
    np.array([-0.35, 0.25, 0.25]), np.array([0.0, 0.45, 0.10]),
    np.array([0.25, 0.10, -0.35]), np.array([-0.25, 0.35, -0.20]),
    np.array([0.40, 0.00, 0.15]), np.array([-0.40, 0.05, -0.10]),
]


# ----------------------------------------------------------------------
# Features: how trustworthy is the DATA under this grasp?
# ----------------------------------------------------------------------
# Deliberately not about the grasp's geometry - week 4 showed that
# carries little signal. These ask whether the camera saw enough here
# to know anything at all.

def features(grasp, cloud, tree):
    normals = np.asarray(cloud.normals)
    f = [grasp.width, grasp.score]

    for c in grasp.contacts():
        _, idx, d2 = tree.search_knn_vector_3d(c, 1)
        f.append(float(np.sqrt(d2[0])))          # distance to any seen point

        k, idx_r, _ = tree.search_radius_vector_3d(c, 0.01)
        f.append(k / 50.0)                       # how much local support

        if k >= 3:                               # do local normals agree?
            local = normals[np.asarray(idx_r)]
            f.append(float(np.linalg.norm(local.mean(axis=0))))
        else:
            f.append(0.0)

    f.append(float(np.dot(grasp.approach, [0, 0, 1])))
    f.append(float(np.dot(grasp.closing, [0, 1, 0])))
    return np.array(f, dtype=np.float32)


# No single object may contribute more than this. Without a cap the
# torus alone produced 43,454 grasps against the cylinder's 480 - the
# model would have learned "torus" and nothing else.
MAX_PER_OBJECT = 2500
MIN_PER_OBJECT = 50


def collect(object_names, label):
    """Build a labelled dataset, capped so no object dominates."""
    X, y, objects = [], [], []
    rng = np.random.default_rng(0)
    print(f"\n{label}")

    for name in object_names:
        full = load_object(name)
        verify = make_verifier(full)

        obj_X, obj_y = [], []
        for ci, cam in enumerate(CAMERAS):
            cloud = camera_view(full, cam, seed=ci)
            if len(cloud.points) < 200:
                continue
            tree = o3d.geometry.KDTreeFlann(cloud)
            for g in find_grasps(cloud, seed=ci):
                obj_X.append(features(g, cloud, tree))
                obj_y.append(verify(g))

        found = len(obj_X)
        if found < MIN_PER_OBJECT:
            print(f"  {name:<12} {found:>6,} grasps   SKIPPED (too few - "
                  f"check its scale in OBJECT_SCALES)")
            continue

        # Subsample so one easy shape cannot flood the dataset.
        if found > MAX_PER_OBJECT:
            keep = rng.choice(found, MAX_PER_OBJECT, replace=False)
            obj_X = [obj_X[i] for i in keep]
            obj_y = [obj_y[i] for i in keep]

        rate = 100 * np.mean(obj_y)
        note = f"  (capped from {found:,})" if found > MAX_PER_OBJECT else ""
        print(f"  {name:<12} {len(obj_X):>6,} grasps   {rate:5.1f}% valid{note}")

        X.extend(obj_X)
        y.extend(obj_y)
        objects.extend([name] * len(obj_X))

    if not X:
        raise SystemExit(
            f"\nNo usable grasps from {object_names}.\n"
            "Every object was either too big or too small for the gripper.\n"
            "Adjust OBJECT_SCALES in grasp_lib.py - the NARROW dimension\n"
            f"needs to fall between {8} mm and {50} mm.")
    return np.stack(X), np.array(y, dtype=np.float32), np.array(objects)


banner("PART 1 — Building the dataset")

Xtr, ytr, _ = collect(TRAIN_OBJECTS, "TRAINING objects")
Xva, yva, _ = collect(VAL_OBJECTS, "VALIDATION object (for early stopping)")
Xte, yte, obj_te = collect(TEST_OBJECTS, "TEST objects (never seen)")

print(f"\n  train : {len(Xtr):>6,} grasps   {100 * ytr.mean():.1f}% valid")
print(f"  val   : {len(Xva):>6,} grasps   {100 * yva.mean():.1f}% valid")
print(f"  test  : {len(Xte):>6,} grasps   {100 * yte.mean():.1f}% valid")
print("""
  Three disjoint sets of OBJECTS. Nothing in the test set shares a
  shape with anything the model trained on. This is the honest
  version of last week's experiment.
""")


# ----------------------------------------------------------------------
# PART 2 — Train, with early stopping
# ----------------------------------------------------------------------

banner("PART 2 — Training with early stopping")

mean, std = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-8

def prep(X, y):
    return (torch.tensor((X - mean) / std),
            torch.tensor(y).unsqueeze(1))

Xtr_t, ytr_t = prep(Xtr, ytr)
Xva_t, yva_t = prep(Xva, yva)
Xte_t, yte_t = prep(Xte, yte)


class Net(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.layers(x)


model = Net(Xtr.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
pos_weight = torch.tensor([(1 - ytr.mean()) / max(ytr.mean(), 1e-6)])
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

best_val, best_state, best_epoch, patience = float("inf"), None, 0, 0
BATCH = 256

for epoch in range(200):
    model.train()
    perm = torch.randperm(len(Xtr_t))
    for start in range(0, len(Xtr_t), BATCH):
        idx = perm[start:start + BATCH]
        optimizer.zero_grad()
        loss = loss_fn(model(Xtr_t[idx]), ytr_t[idx])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_loss = loss_fn(model(Xva_t), yva_t).item()

    if val_loss < best_val - 1e-4:
        best_val, best_epoch, patience = val_loss, epoch, 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        patience += 1

    if epoch % 20 == 0:
        print(f"  epoch {epoch:>3}   train {loss.item():.4f}   "
              f"val {val_loss:.4f}")

    if patience >= 25:
        print(f"  stopped early at epoch {epoch} "
              f"(no improvement for 25 epochs)")
        break

model.load_state_dict(best_state)
print(f"\n  kept the model from epoch {best_epoch} (val loss {best_val:.4f})")
print("""
  Early stopping is the fix for last week. Instead of training a fixed
  60 epochs and hoping, we watch a held-out object and keep whichever
  epoch generalised best. Note the validation object is held out BY
  OBJECT too - otherwise early stopping would just select the epoch
  that memorised most efficiently.
""")


# ----------------------------------------------------------------------
# PART 3 — Evaluate on unseen objects
# ----------------------------------------------------------------------

banner("PART 3 — Results on objects never seen")


def auc(scores, labels):
    labels = labels.astype(bool)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    return (ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


model.eval()
with torch.no_grad():
    probs = torch.sigmoid(model(Xte_t)).numpy().ravel()

geom = Xte[:, 1]

print(f"  valid in test set : {100 * yte.mean():.1f}%\n")
print(f"  GEOMETRIC SCORE   AUC {auc(geom, yte):.3f}")
print(f"  NEURAL NETWORK    AUC {auc(probs, yte):.3f}")

print("\n  Per object, so one easy shape cannot hide a failure:")
print(f"  {'object':<12} {'n':>6} {'valid%':>8} {'geom AUC':>10} {'net AUC':>9}")
for name in TEST_OBJECTS:
    m = obj_te == name
    if m.sum() < 20:
        print(f"  {name:<12} {m.sum():>6}   (too few to score)")
        continue
    print(f"  {name:<12} {m.sum():>6} {100 * yte[m].mean():>7.1f}% "
          f"{auc(geom[m], yte[m]):>10.3f} {auc(probs[m], yte[m]):>9.3f}")


# ----------------------------------------------------------------------
# PART 4 — Calibration on unseen objects
# ----------------------------------------------------------------------

banner("PART 4 — Is the confidence honest on new objects?")

print("  predicted    actual     count")
print("  ---------    ------     -----")
for lo in np.arange(0.0, 1.0, 0.2):
    m = (probs >= lo) & (probs < lo + 0.2)
    if m.sum() < 10:
        continue
    print(f"  {lo:.1f}-{lo + 0.2:.1f}      {yte[m].mean():.2f}      {m.sum():>5}")

print("""
  Expect this to be worse than last week. Calibration that holds on
  familiar objects usually breaks on unfamiliar ones - the model has
  no idea how wrong it is about shapes it has never met.

  That is precisely the setting conformal prediction was built for:
  a guarantee that does not depend on the model being well calibrated,
  only on the calibration data being exchangeable with the test data.
""")

torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std},
           "model_week6.pt")
np.savez("week6_data.npz", Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva,
         Xte=Xte, yte=yte, obj_te=obj_te)

banner("DONE")
print("""
Saved: model_week6.pt, week6_data.npz

WHAT CHANGED FROM LAST WEEK

  Split by object instead of by viewpoint, so the test set shares no
  shape with the training set. Added a validation object and early
  stopping, so the model is no longer trained past its best point.

WHAT TO LOOK FOR

  The per-object table matters more than the overall AUC. If the
  network does well on sphere and cone (smooth, regular) and badly on
  armadillo and knot (irregular), it has learned something about easy
  shapes, not about data quality.

  If the geometric score still wins, that is a real result and worth
  reporting. It would mean the data-quality features do not transfer
  across shapes - which is itself a finding, and sharpens the case for
  calibration over better point estimates.

EXERCISES

  1. Swap which objects are in train and test. Does the conclusion
     hold, or was it an artefact of this particular split?

  2. Leave-one-object-out: train 8 times, each time holding out a
     different object, and report mean and spread of AUC. That is the
     proper way to report this with so few objects.

  3. Add the object's own statistics as features (size, curvature
     spread). Does AUC rise? If so the model is using shape, not data
     quality - which would be cheating, given the goal.

  4. Plot the calibration curves for week 5 and week 6 side by side.
     The degradation IS your argument for conformal prediction.

NEXT: GraspNet. 88 real objects, 190 cluttered scenes, labels from
physics simulation. Everything you have built so far was practice for
loading that.
""")