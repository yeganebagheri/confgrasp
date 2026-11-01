"""
ConfGrasp - Week 15: figures for the write-up.

Makes two PNGs from week12_calib.npz:
  fig1_score_vs_correct.png   - does the network score mean anything?
  fig2_calibration.png        - promise vs reality, old (grasps) vs new (frames)

Run:  python 15_make_figures.py      (needs: pip install matplotlib)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")                     # save files, no window
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist

data = np.load("week12_calib.npz")
scores = data["scores"]
correct = data["correct"].astype(bool)
scene = data["scene"]
frame_id = scene * 1000 + data["frame"]
scenes = np.unique(scene)


# ----------------------------------------------------------------------
# Same functions as week 14
# ----------------------------------------------------------------------
def cp_upper(failures, n, delta=0.05):
    if n == 0 or failures >= n:
        return 1.0
    return float(beta_dist.ppf(1 - delta, failures + 1, n - failures))


def calibrate(s, y, alpha, min_n=200):
    best_t, best_n = None, -1
    for t in np.unique(s)[::-1]:
        acc = s >= t
        n = int(acc.sum())
        if n < min_n:
            continue
        if cp_upper(int((~y[acc]).sum()), n) <= alpha and n > best_n:
            best_t, best_n = float(t), n
    return best_t


def calibrate_frames(s, y, f, alpha, min_n=20):
    best_t, best_n = None, -1
    for t in np.unique(s)[::-1]:
        acc = s >= t
        n = int(acc.sum())
        if n == 0:
            continue
        n_eff = len(np.unique(f[acc]))
        if n_eff < min_n:
            continue
        fails_eff = int(round(float((~y[acc]).mean()) * n_eff))
        if cp_upper(fails_eff, n_eff) <= alpha and n > best_n:
            best_t, best_n = float(t), n
    return best_t


def loso(method, alpha):
    """Leave-one-scene-out -> (mean accepted fraction, mean failure rate, #violated)."""
    acc_list, fail_list, viol = [], [], 0
    for s in scenes:
        cal, test = scene != s, scene == s
        if method == "old":
            t = calibrate(scores[cal], correct[cal], alpha)
        else:
            t = calibrate_frames(scores[cal], correct[cal], frame_id[cal], alpha)
        if t is None:
            acc_list.append(0.0)
            continue
        a = scores[test] >= t
        acc_list.append(a.mean())
        if a.sum():
            fr = float((~correct[test][a]).mean())
            fail_list.append(fr)
            viol += fr > alpha
    fail = np.mean(fail_list) if fail_list else np.nan
    return np.mean(acc_list), fail, viol


# ----------------------------------------------------------------------
# Figure 1: score deciles vs % correct
# ----------------------------------------------------------------------
edges = np.quantile(scores, np.linspace(0, 1, 11))
bins = np.clip(np.searchsorted(edges, scores, "right") - 1, 0, 9)
pct = [100 * correct[bins == b].mean() for b in range(10)]

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(range(1, 11), pct, color="#4C72B0")
ax.axhline(100 * correct.mean(), color="gray", ls="--", lw=1,
           label=f"all grasps ({100 * correct.mean():.0f}%)")
ax.set_xlabel("network score decile (1 = lowest, 10 = highest)")
ax.set_ylabel("% correct grasps")
ax.set_title("Higher score -> more likely correct")
ax.set_xticks(range(1, 11))
ax.set_ylim(0, 100)
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig("fig1_score_vs_correct.png", dpi=150)
print("saved fig1_score_vs_correct.png")


# ----------------------------------------------------------------------
# Figure 2: promise vs reality, and the price (abstention)
# ----------------------------------------------------------------------
alphas = np.array([0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40])
res = {m: [loso(m, a) for a in alphas] for m in ("old", "new")}
style = {"old": dict(color="#C44E52", marker="o", label="count grasps (week 13)"),
         "new": dict(color="#55A868", marker="s", label="count frames (week 14)")}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
for m in ("old", "new"):
    acc = np.array([r[0] for r in res[m]])
    fail = np.array([r[1] for r in res[m]])
    ax1.plot(alphas * 100, fail * 100, **style[m])
    ax2.plot(alphas * 100, acc * 100, **style[m])

ax1.plot([10, 40], [10, 40], color="gray", ls="--", lw=1, label="promise kept (y = x)")
ax1.set_xlabel("promised max failure rate  alpha (%)")
ax1.set_ylabel("actual failure rate on held-out scene (%)")
ax1.set_title("Promise vs reality (below the line = kept)")
ax1.legend()

ax2.set_xlabel("promised max failure rate  alpha (%)")
ax2.set_ylabel("grasps accepted (%)")
ax2.set_title("The price: how much the robot still tries")
ax2.legend()

fig.tight_layout()
fig.savefig("fig2_calibration.png", dpi=150)
print("saved fig2_calibration.png")

# Small table for the README
print(f"\n| alpha | grasps: accepted / fail / violated | frames: accepted / fail / violated |")
print(f"|---|---|---|")
for i, a in enumerate(alphas):
    o, n = res["old"][i], res["new"][i]
    fmt = lambda r: (f"{100 * r[0]:.1f}% / "
                     f"{'-' if np.isnan(r[1]) else f'{100 * r[1]:.1f}%'} / {r[2]}/5")
    print(f"| {a:.2f} | {fmt(o)} | {fmt(n)} |")