"""
ConfGrasp — Week 5
PyTorch, learned on your own data.

Last week you proved the geometric score is worthless: it improved
prediction accuracy by exactly +0.000. This week you try to beat it
with a small neural network, written in PyTorch.

Two things happen at once:
  - you learn PyTorch mechanics (tensors, autograd, training loops)
  - you find out whether the failures are predictable AT ALL

That second question matters more than it looks. If a model can predict
which grasps fail, then calibrated confidence is possible and your
project has something to calibrate. If nothing can predict it, the
failures are irreducible and the story changes.

Run:  python 06_pytorch.py
"""

import numpy as np
import open3d as o3d
import torch
import torch.nn as nn

from grasp_lib import ground_truth, camera_view, find_grasps, make_verifier


def banner(text):
    print("\n" + "=" * 66)
    print(text)
    print("=" * 66)


torch.manual_seed(0)
np.random.seed(0)


# ----------------------------------------------------------------------
# PART 1 — PyTorch in five minutes
# ----------------------------------------------------------------------
# You know Keras. PyTorch differs in one important way: you write the
# training loop yourself. There is no .fit(). That feels like extra work
# for ten minutes and then becomes the reason everyone prefers it - you
# can see and change every step.

banner("PART 1 — PyTorch basics")

# A tensor is a numpy array that can also remember how it was computed.
x = torch.tensor([2.0], requires_grad=True)
y = x ** 3 + 5 * x

# .backward() walks that history and computes the derivative.
y.backward()

print(f"  y = x^3 + 5x  at x=2  ->  y = {y.item():.1f}")
print(f"  dy/dx computed by hand : 3*x^2 + 5 = {3 * 2**2 + 5}")
print(f"  dy/dx computed by torch: {x.grad.item():.1f}")
print("""
  That is autograd, and it is the whole trick. You never write a
  derivative again - PyTorch records every operation and differentiates
  backwards through the record. Keras does this too; PyTorch just lets
  you see it.
""")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  running on: {device}")
if device.type == "cpu":
    print("  (no NVIDIA GPU - fine for this week, rent one in month 2)")


# ----------------------------------------------------------------------
# PART 2 — Build a real dataset
# ----------------------------------------------------------------------
# 180 grasps from one viewpoint is not a dataset. We sweep the camera
# around the object and collect thousands, each labelled by the ground
# truth. This is also more honest: a model that only sees one viewpoint
# would learn that viewpoint, not the problem.

banner("PART 2 — Generating training data")

full = ground_truth()
verify = make_verifier(full)
full_tree = o3d.geometry.KDTreeFlann(full)

CAMERAS = [
    np.array([0.0, 0.30, 0.40]), np.array([0.35, 0.25, 0.25]),
    np.array([-0.35, 0.25, 0.25]), np.array([0.0, 0.45, 0.10]),
    np.array([0.25, 0.10, -0.35]), np.array([-0.25, 0.35, -0.20]),
    np.array([0.40, 0.00, 0.15]), np.array([-0.40, 0.05, -0.10]),
]


def features(grasp, cloud, tree):
    """Describe a grasp with numbers the network can learn from.

    The interesting ones are not about the grasp's geometry - last week
    proved that carries no signal. They are about whether the DATA
    under the grasp is trustworthy: how much was actually seen near
    each fingertip, and how reliable the normals there were.
    """
    points = np.asarray(cloud.points)
    normals = np.asarray(cloud.normals)
    ca, cb = grasp.contacts()

    f = [grasp.width, grasp.score]

    for c in (ca, cb):
        # How far is the nearest point the camera ACTUALLY saw?
        _, idx, d2 = tree.search_knn_vector_3d(c, 1)
        nearest = float(np.sqrt(d2[0]))
        f.append(nearest)

        # How many seen points are nearby? Sparse support = guesswork.
        k, idx_r, _ = tree.search_radius_vector_3d(c, 0.01)
        f.append(k / 50.0)

        # How much do the local normals agree? Disagreement means the
        # surface estimate there is unreliable.
        if k >= 3:
            local = normals[np.asarray(idx_r)]
            f.append(float(np.linalg.norm(local.mean(axis=0))))
        else:
            f.append(0.0)

    # Is the approach direction pointing into the camera's blind side?
    f.append(float(np.dot(grasp.approach, [0, 0, 1])))
    f.append(float(np.dot(grasp.closing, [0, 1, 0])))
    return np.array(f, dtype=np.float32)


X, y, groups = [], [], []
for gi, cam in enumerate(CAMERAS):
    cloud = camera_view(full, cam, seed=gi)
    tree = o3d.geometry.KDTreeFlann(cloud)
    grasps = find_grasps(cloud, seed=gi)

    for g in grasps:
        X.append(features(g, cloud, tree))
        y.append(verify(g))
        groups.append(gi)
    print(f"  camera {gi}: {len(grasps):>5,} grasps, "
          f"{100 * np.mean([verify(g) for g in grasps]):.0f}% valid")

X = np.stack(X)
y = np.array(y, dtype=np.float32)
groups = np.array(groups)

print(f"\n  dataset : {X.shape[0]:,} grasps, {X.shape[1]} features each")
print(f"  valid   : {100 * y.mean():.1f}%")


# ----------------------------------------------------------------------
# PART 3 — Split by VIEWPOINT, not at random
# ----------------------------------------------------------------------
# This matters. Grasps from the same camera view are highly correlated -
# they share the same point cloud, the same holes, the same noise. Split
# them randomly and the model sees near-duplicates in both train and
# test, and your accuracy is a lie.
#
# You already know this failure mode. It is the windowing confound from
# your PV work, in a different costume.

banner("PART 3 — Splitting the data honestly")

train_mask = groups < 6         # cameras 0-5
test_mask = groups >= 6         # cameras 6-7, never seen in training

mean = X[train_mask].mean(axis=0)
std = X[train_mask].std(axis=0) + 1e-8      # scale using TRAIN stats only

Xtr = torch.tensor((X[train_mask] - mean) / std)
ytr = torch.tensor(y[train_mask]).unsqueeze(1)
Xte = torch.tensor((X[test_mask] - mean) / std)
yte = torch.tensor(y[test_mask]).unsqueeze(1)

print(f"  train : {len(Xtr):,} grasps from cameras 0-5")
print(f"  test  : {len(Xte):,} grasps from cameras 6-7 (unseen viewpoints)")
print("""
  Note the test set comes from viewpoints the model never trained on.
  That is the honest question: does it generalise to a new view, or
  has it memorised these particular point clouds?
""")


# ----------------------------------------------------------------------
# PART 4 — The network
# ----------------------------------------------------------------------

banner("PART 4 — Training")


class GraspNet(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),          # one number: the logit
        )

    def forward(self, x):
        return self.layers(x)


model = GraspNet(X.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# BCEWithLogits = sigmoid + binary cross-entropy, fused for stability.
# pos_weight corrects for the class imbalance.
pos_weight = torch.tensor([(1 - y.mean()) / y.mean()]).to(device)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

Xtr, ytr = Xtr.to(device), ytr.to(device)
Xte, yte = Xte.to(device), yte.to(device)

BATCH = 256
for epoch in range(60):
    model.train()
    perm = torch.randperm(len(Xtr), device=device)

    for start in range(0, len(Xtr), BATCH):
        idx = perm[start:start + BATCH]

        # The four lines that ARE the training loop:
        optimizer.zero_grad()              # clear last step's gradients
        loss = loss_fn(model(Xtr[idx]), ytr[idx])   # forward + loss
        loss.backward()                    # gradients, via autograd
        optimizer.step()                   # nudge the weights

    if epoch % 15 == 0 or epoch == 59:
        model.eval()
        with torch.no_grad():
            te_loss = loss_fn(model(Xte), yte).item()
        print(f"  epoch {epoch:>3}  train loss {loss.item():.4f}   "
              f"test loss {te_loss:.4f}")


# ----------------------------------------------------------------------
# PART 5 — Did it beat the useless baseline?
# ----------------------------------------------------------------------

banner("PART 5 — Evaluation on unseen viewpoints")

model.eval()
with torch.no_grad():
    probs = torch.sigmoid(model(Xte)).cpu().numpy().ravel()

truth = yte.cpu().numpy().ravel().astype(bool)
geom_score = X[test_mask][:, 1]          # the old geometric score


def auc(scores, labels):
    """Area under the ROC curve, by rank. 0.5 = useless, 1.0 = perfect."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def best_accuracy(scores, labels):
    best = 0.0
    for t in np.unique(scores):
        best = max(best, ((scores >= t) == labels).mean())
    return best


baseline = max(truth.mean(), 1 - truth.mean())

print(f"  valid grasps in the test set : {100 * truth.mean():.1f}%")
print(f"  accuracy from always guessing: {baseline:.3f}\n")

print(f"  GEOMETRIC SCORE   AUC {auc(geom_score, truth):.3f}   "
      f"best accuracy {best_accuracy(geom_score, truth):.3f}")
print(f"  NEURAL NETWORK    AUC {auc(probs, truth):.3f}   "
      f"best accuracy {best_accuracy(probs, truth):.3f}")

print(f"""
  AUC is the cleanest number here: the probability that a randomly
  chosen valid grasp scores higher than a randomly chosen invalid one.
  0.5 means the score is pure noise. 1.0 means perfect separation.
""")


# ----------------------------------------------------------------------
# PART 6 — Is the model's confidence honest?
# ----------------------------------------------------------------------
# This is the question your whole project rests on. The network outputs
# numbers it calls probabilities. When it says 0.8, do 80% of those
# grasps actually succeed?

banner("PART 6 — Calibration: does 0.8 mean 80%?")

print("  predicted    actual     count")
print("  ---------    ------     -----")
for lo in np.arange(0.0, 1.0, 0.2):
    in_bin = (probs >= lo) & (probs < lo + 0.2)
    if in_bin.sum() < 10:
        continue
    print(f"  {lo:.1f}-{lo + 0.2:.1f}      "
          f"{truth[in_bin].mean():.2f}      {in_bin.sum():>5}")

print("""
  If the two columns match, the model is calibrated. If 'predicted'
  runs higher than 'actual', it is overconfident - the usual case.

  Note that even a well-calibrated model only promises this ON AVERAGE,
  on data like the training data. It makes no promise about any single
  grasp, and no promise at all once the distribution shifts.

  Conformal prediction is what turns that soft average into a hard
  guarantee with a chosen risk level. That is month 3, and now you have
  a model whose output there is something to calibrate.
""")

torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std},
           "model_week5.pt")
np.savez("week5_data.npz", X=X, y=y, groups=groups)

banner("DONE")
print("""
Saved: model_week5.pt, week5_data.npz

WHAT YOU LEARNED

  PyTorch: tensors, autograd, nn.Module, the explicit training loop,
  BCEWithLogitsLoss, class weighting, train/eval modes, no_grad.
  You can now say you have trained a model in PyTorch, truthfully.

  And an experimental result: whether grasp failures are predictable
  from data-quality features. Whatever the AUC came out as, you
  measured it on held-out VIEWPOINTS, which is the honest split.

EXERCISES

  1. Remove the data-quality features (keep only width and the
     geometric score). How much does AUC fall? That difference is
     the value of knowing what the camera actually saw.

  2. Switch to a random train/test split instead of splitting by
     camera. Watch the accuracy jump. Then explain in one sentence
     why that number is a lie.

  3. Add more cameras to CAMERAS and see whether AUC improves with
     more viewpoints.

  4. Plot a calibration curve (predicted vs actual, from PART 6) with
     matplotlib. This figure goes in your writeup.

  5. Reimplement your Eurac autoencoder in PyTorch. That is the repo
     that removes the TensorFlow objection from your CV - and it is
     the single highest-value thing you can put on GitHub this month.

NEXT: GraspNet. By now the download should be progressing. Real
cluttered scenes, real sensor data, and grasp labels from physics
simulation rather than a geometric rule.
""")