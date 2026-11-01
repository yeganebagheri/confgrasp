from scipy.stats import beta as beta_dist
import numpy as np

data = np.load("week12_calib.npz")
scores  = data["scores"]                  # network score per grasp (higher = better)
correct = data["correct"].astype(bool)    # True = good grasp (by your 1 cm / 30° rule)
scene   = data["scene"]                   # which scene each grasp came from (100..104)

print(f"grasps: {len(scores):,}   correct: {100 * correct.mean():.1f}%")
print("scenes:", np.unique(scene))

def clopper_pearson_upper(failures, n, delta=0.05):
    """Upper confidence bound on the true failure rate.

    The observed rate is failures/n. The true rate could be worse by
    chance, so we need a bound, not the observation.

    Hoeffding's inequality is the famous one, but it is very loose at
    small n: with 50 samples its margin is about 0.17, so even ZERO
    observed failures can only promise "below 17%". That made alphas
    of 0.02-0.10 unreachable regardless of the data.

    Clopper-Pearson inverts the binomial distribution exactly. Much
    tighter, same guarantee.
    """
    if n == 0:
        return 1.0
    if failures == n:
        return 1.0
    return float(beta_dist.ppf(1 - delta, failures + 1, n - failures))


def calibrate(cal_scores, cal_labels, alpha, delta=0.05, min_n=200):
    """Most permissive threshold whose failure rate is provably < alpha.

    We sweep every threshold from strict to permissive and keep the
    one that accepts the most grasps while still holding the bound.

    Note we do NOT stop at the first threshold that fails. The bound
    depends on n, and n grows as the threshold loosens - so a strict
    threshold can fail on small-sample grounds while a looser one
    passes. The relationship is not monotonic.
    """
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



ALPHA = 0.20
scenes = np.unique(scene)

print(f"\nLeave-one-scene-out, alpha = {ALPHA:.2f}")
print(f"{'test scene':>10} {'threshold':>10} {'accepted':>9} {'fail rate':>10}  verdict")

for test_s in scenes:
    cal = scene != test_s          # the other 4 scenes
    test = scene == test_s         # the left-out scene

    t = calibrate(scores[cal], correct[cal], ALPHA)
    if t is None:
        print(f"{test_s:>10}   could not guarantee {ALPHA:.0%} on the calibration scenes")
        continue

    r = evaluate(t, scores[test], correct[test])
    held = r["failure_rate"] <= ALPHA
    print(f"{test_s:>10} {t:>10.3f} {100 * (1 - r['abstain_rate']):>8.1f}% "
          f"{100 * r['failure_rate']:>9.1f}%  {'HELD' if held else 'VIOLATED'}")


print(f"\n{'alpha':>8} {'avg accepted':>13} {'avg fail rate':>14} {'violated':>9}")

# Baseline: no calibration, accept everything
base_fail = 1 - correct.mean()
print(f"{'none':>8} {100.0:>12.1f}% {100 * base_fail:>13.1f}%   (no guarantee)")

for alpha in (0.10, 0.20, 0.30):
    accepted, fails, violated = [], [], 0
    for test_s in scenes:
        cal, test = scene != test_s, scene == test_s
        t = calibrate(scores[cal], correct[cal], alpha)
        r = evaluate(t, scores[test], correct[test])   # t=None -> abstain on everything

        accepted.append(1 - r["abstain_rate"])
        if r["accepted"] > 0:
            fails.append(r["failure_rate"])
            violated += r["failure_rate"] > alpha

    avg_fail = 100 * np.mean(fails) if fails else float("nan")
    print(f"{alpha:>8.2f} {100 * np.mean(accepted):>12.1f}% {avg_fail:>13.1f}% "
          f"{violated:>5} / {len(scenes)}")