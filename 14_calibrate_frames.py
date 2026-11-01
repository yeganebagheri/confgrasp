"""
ConfGrasp - Week 14
Count FRAMES, not grasps.

Week 13 treated ~4,500 calibration grasps as 4,500 independent samples.
But grasps from the same camera frame are near-copies of each other, so
the real amount of evidence is closer to the number of frames (~44).
This file compares the old method (count grasps) with the new one
(count frames) using leave-one-scene-out.

Run:  python 14_calibrate_frames.py
"""

from scipy.stats import beta as beta_dist
import numpy as np

data = np.load("week12_calib.npz")
scores = data["scores"]                   # network score per grasp (higher = better)
correct = data["correct"].astype(bool)    # True = good grasp (1 cm / 30 deg rule)
scene = data["scene"]                     # which scene (100..104)
frame = data["frame"]                     # which frame inside that scene (0, 25, ..., 250)

# A frame is only unique together with its scene: s100 f0 != s101 f0.
frame_id = scene * 1000 + frame           # e.g. 100025 = scene 100, frame 25

print(f"grasps: {len(scores):,}   correct: {100 * correct.mean():.1f}%")
print(f"scenes: {np.unique(scene)}   frames: {len(np.unique(frame_id))}")


def clopper_pearson_upper(failures, n, delta=0.05):
    """Worst-case true failure rate (95% confidence) given failures out of n."""
    if n == 0:
        return 1.0
    if failures >= n:
        return 1.0
    return float(beta_dist.ppf(1 - delta, failures + 1, n - failures))


# ----------------------------------------------------------------------
# OLD (week 13): every grasp counts as one independent sample
# ----------------------------------------------------------------------
def calibrate(cal_scores, cal_labels, alpha, delta=0.05, min_n=200):
    candidates = np.unique(cal_scores)[::-1]      # strict -> permissive
    best_t, best_n = None, -1
    for t in candidates:
        accepted = cal_scores >= t
        n = int(accepted.sum())
        if n < min_n:
            continue
        failures = int((~cal_labels[accepted]).sum())
        if clopper_pearson_upper(failures, n, delta) <= alpha and n > best_n:
            best_t, best_n = float(t), n
    return best_t


# ----------------------------------------------------------------------
# NEW (week 14): every FRAME counts as one sample
# ----------------------------------------------------------------------
def calibrate_frames(cal_scores, cal_labels, cal_frames, alpha, delta=0.05, min_n=20):
    """Same sweep as calibrate(), but the evidence is measured in frames.

    For each threshold:
      rate         = failure rate among accepted grasps (as before)
      n_eff        = number of different frames with >= 1 accepted grasp
      failures_eff = round(rate * n_eff)
      bound        = clopper_pearson_upper(failures_eff, n_eff)
    """
    candidates = np.unique(cal_scores)[::-1]      # strict -> permissive
    best_t, best_n = None, -1
    for t in candidates:
        accepted = cal_scores >= t
        n_grasps = int(accepted.sum())
        if n_grasps == 0:
            continue
        n_eff = len(np.unique(cal_frames[accepted]))
        if n_eff < min_n:
            continue
        rate = float((~cal_labels[accepted]).mean())
        failures_eff = int(round(rate * n_eff))
        if clopper_pearson_upper(failures_eff, n_eff, delta) <= alpha and n_grasps > best_n:
            best_t, best_n = float(t), n_grasps
    return best_t


def evaluate(threshold, test_scores, test_labels):
    if threshold is None:
        return dict(accepted=0, abstained=len(test_scores),
                    abstain_rate=1.0, failure_rate=float("nan"))
    accepted = test_scores >= threshold
    n = int(accepted.sum())
    return dict(
        accepted=n,
        abstained=len(test_scores) - n,
        abstain_rate=1 - n / len(test_scores),
        failure_rate=float((~test_labels[accepted]).mean()) if n else float("nan"),
    )


# ----------------------------------------------------------------------
# Leave-one-scene-out with both methods
# ----------------------------------------------------------------------
scenes = np.unique(scene)


def run(method, alpha):
    """Leave-one-scene-out. Returns per-scene rows + summary."""
    rows = []
    for test_s in scenes:
        cal, test = scene != test_s, scene == test_s
        if method == "old":
            t = calibrate(scores[cal], correct[cal], alpha)
        else:
            t = calibrate_frames(scores[cal], correct[cal], frame_id[cal], alpha)
        r = evaluate(t, scores[test], correct[test])
        violated = r["accepted"] > 0 and r["failure_rate"] > alpha
        rows.append((test_s, t, 1 - r["abstain_rate"], r["failure_rate"], violated))
    return rows


# Detail table for alpha = 0.20, both methods
ALPHA = 0.20
for method in ("old", "new"):
    print(f"\n[{method.upper()}] leave-one-scene-out, alpha = {ALPHA:.2f}")
    print(f"{'test scene':>10} {'threshold':>10} {'accepted':>9} {'fail rate':>10}  verdict")
    for s, t, acc, fail, viol in run(method, ALPHA):
        if t is None:
            print(f"{s:>10} {'none':>10} {0:>8.1f}%  {'-':>9}   abstain on everything")
            continue
        print(f"{s:>10} {t:>10.3f} {100 * acc:>8.1f}% {100 * fail:>9.1f}%  "
              f"{'VIOLATED' if viol else 'HELD'}")

# Side-by-side summary
print(f"\n{'':>8}   {'OLD (grasps)':^28}   {'NEW (frames)':^28}")
print(f"{'alpha':>8}   {'accepted':>9} {'fail':>8} {'violated':>9}   "
      f"{'accepted':>9} {'fail':>8} {'violated':>9}")
print(f"{'none':>8}   {100.0:>8.1f}% {100 * (1 - correct.mean()):>7.1f}% {'-':>9}")

for alpha in (0.10, 0.20, 0.30):
    line = f"{alpha:>8.2f}  "
    for method in ("old", "new"):
        rows = run(method, alpha)
        acc = np.mean([r[2] for r in rows])
        fails = [r[3] for r in rows if r[2] > 0]
        fail = np.mean(fails) if fails else float("nan")
        viol = sum(r[4] for r in rows)
        line += f" {100 * acc:>8.1f}% {100 * fail:>7.1f}% {viol:>5} / {len(scenes)}  "
    print(line)

# ----------------------------------------------------------------------
# Answers (write yours here after running):
#   1. Did the violations go down?
#   2. What did it cost in accepted grasps?
#   3. Frames from the SAME scene are also similar to each other, so even
#      ~44 frames overstate the evidence. With only 4 calibration scenes,
#      what would really fix it?
# ----------------------------------------------------------------------