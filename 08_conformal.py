"""
ConfGrasp — Week 7
From a score to a guarantee.

You now have a score that ranks grasps reasonably well (AUC 0.779) and
means nothing in absolute terms. 0.6 is better than 0.4, but nobody can
tell you what 0.6 promises.

This week you build the thing your whole project is named after: a
threshold chosen so that among the grasps you ACCEPT, the failure rate
is provably below a level you choose. Everything else, you abstain on.

And then you break it on purpose, which is the actual contribution.

Run:  python 08_conformal.py
"""

import numpy as np

data = np.load("week6_data.npz", allow_pickle=True)


def banner(text):
    print("\n" + "=" * 68)
    print(text)
    print("=" * 68)


# Pool everything; we will re-split it ourselves in different ways.
X = np.concatenate([data["Xtr"], data["Xva"], data["Xte"]])
y = np.concatenate([data["ytr"], data["yva"], data["yte"]]).astype(bool)
objects = np.concatenate([
    np.array(["train"] * len(data["ytr"])),
    np.array(["val"] * len(data["yva"])),
    data["obj_te"],
])

score = X[:, 1]      # the geometric score - the one that actually works

print(f"  pooled dataset : {len(score):,} grasps, {100 * y.mean():.1f}% valid")


# ----------------------------------------------------------------------
# PART 1 — The idea
# ----------------------------------------------------------------------

banner("PART 1 — What a guarantee means")
print("""
  A confidence score says: "this one looks better than that one."
  A guarantee says: "of the grasps I accept, at most 10% will fail."

  The second is a promise about the FUTURE, made before seeing the
  outcome. You cannot get it from the model's own output, because the
  model has no idea how wrong it is. You get it from held-out data.

  The recipe (split conformal, risk-control flavour):

    1. Set aside a CALIBRATION set with known outcomes.
    2. Choose your tolerable failure rate, alpha.
    3. Find the lowest score threshold at which the failure rate
       among accepted grasps is provably below alpha.
    4. On new data: accept above the threshold, ABSTAIN below it.

  Step 3 says "provably", not "observed". If 20 calibration grasps
  happen to show a 5% failure rate, that is weak evidence. A finite-
  sample bound turns the observation into a statement that holds even
  when you were unlucky.
""")


# ----------------------------------------------------------------------
# PART 2 — Calibration
# ----------------------------------------------------------------------

from scipy.stats import beta as beta_dist


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


banner("PART 2 — The guarantee, when the data behaves")

rng = np.random.default_rng(0)
perm = rng.permutation(len(score))
half = len(perm) // 2
cal_i, test_i = perm[:half], perm[half:]

print("  Calibration and test drawn at random from the SAME pool, so")
print("  they are exchangeable - the assumption conformal relies on.\n")
print(f"  {'alpha':>6} {'threshold':>10} {'accepted':>9} {'abstain':>9} "
      f"{'actual fail':>12}  verdict")

for alpha in (0.02, 0.05, 0.10, 0.15, 0.20, 0.30):
    t = calibrate(score[cal_i], y[cal_i], alpha)
    r = evaluate(t, score[test_i], y[test_i])
    ok = "HELD" if r["failure_rate"] <= alpha else "VIOLATED"
    tstr = f"{t:.4f}" if t is not None else "none"
    print(f"  {alpha:>6.2f} {tstr:>10} {r['accepted']:>9,} "
          f"{100 * r['abstain_rate']:>8.1f}% {100 * r['failure_rate']:>11.2f}%"
          f"  {ok}")

print("""
  Read the trade-off down the column. Demand a 2% failure rate and you
  abstain on far more; tolerate 20% and you accept almost everything.
  That is the coverage-versus-throughput curve, and it is the thing a
  warehouse operator actually wants to choose.
""")


# ----------------------------------------------------------------------
# PART 3 — Break it
# ----------------------------------------------------------------------
# Conformal guarantees rest on one assumption: exchangeability. The
# calibration data must look like the test data. In a warehouse it
# does not - tomorrow's parcels are not today's.
#
# Here we calibrate on some OBJECTS and test on others. Nothing else
# changes. If coverage breaks, the assumption is doing real work, and
# that is worth knowing before anyone ships this.

banner("PART 3 — Calibrate on one object, deploy on another")

test_objects = [o for o in np.unique(objects) if o not in ("train", "val")]

print(f"  {'calibrate on':<12} {'deploy on':<12} {'alpha':>6} "
      f"{'abstain':>9} {'actual fail':>12}  verdict")

ALPHA = 0.10
violations = 0
trials = 0

for cal_obj in test_objects:
    cal_mask = objects == cal_obj
    t = calibrate(score[cal_mask], y[cal_mask], ALPHA)
    if t is None:
        print(f"  {cal_obj:<12} {'-':<12} "
              f"could not guarantee {ALPHA:.0%} even on itself")
        continue

    for dep_obj in test_objects:
        if dep_obj == cal_obj:
            continue
        dep_mask = objects == dep_obj
        r = evaluate(t, score[dep_mask], y[dep_mask])
        trials += 1
        held = r["failure_rate"] <= ALPHA
        violations += not held
        print(f"  {cal_obj:<12} {dep_obj:<12} {ALPHA:>6.2f} "
              f"{100 * r['abstain_rate']:>8.1f}% "
              f"{100 * r['failure_rate']:>11.2f}%  "
              f"{'HELD' if held else 'VIOLATED'}")

print(f"\n  guarantee violated in {violations} of {trials} object transfers")
print("""
  This is the result worth writing up.

  Within a distribution, the guarantee holds - that is the theorem
  working. Across objects it can break, because the calibration data
  was no longer exchangeable with the deployment data.

  So conformal prediction does not make the distribution-shift problem
  disappear. What it does is make the failure VISIBLE and the
  assumption EXPLICIT, instead of hiding behind a number between 0
  and 1 that promised nothing in the first place.
""")


# ----------------------------------------------------------------------
# PART 4 — The honest fix
# ----------------------------------------------------------------------

banner("PART 4 — Calibrating across many objects instead of one")

print(f"  {'held out':<12} {'abstain':>9} {'actual fail':>12}  verdict")

violations = 0
for held_out in test_objects:
    cal_mask = np.isin(objects, [o for o in test_objects if o != held_out])
    dep_mask = objects == held_out

    t = calibrate(score[cal_mask], y[cal_mask], ALPHA)
    r = evaluate(t, score[dep_mask], y[dep_mask])
    held = r["failure_rate"] <= ALPHA
    violations += not held
    print(f"  {held_out:<12} {100 * r['abstain_rate']:>8.1f}% "
          f"{100 * r['failure_rate']:>11.2f}%  "
          f"{'HELD' if held else 'VIOLATED'}")

print(f"""
  violated on {violations} of {len(test_objects)} held-out objects

  Calibrating across several objects is more robust than calibrating
  on one, because the calibration set now spans more of the variation
  the deployment will meet. It is still not a proof - a genuinely new
  object can always fall outside - but it is the practical answer, and
  it is what a real deployment would do: recalibrate on a sample of
  whatever is actually coming down the conveyor.
""")

banner("DONE")
print("""
WHAT YOU BUILT

  A calibration procedure that converts an uninterpretable score into
  an operating point with a stated risk level, plus an abstention
  policy, plus an honest account of when the guarantee fails.

  That last part is the contribution. Plenty of papers apply conformal
  prediction and report that coverage held. Showing exactly where it
  breaks, and by how much, is more useful and harder to argue with.

EXERCISES

  1. Plot the coverage-versus-abstention curve from PART 2. That is
     the headline figure of your writeup.

  2. Replace Hoeffding with a Clopper-Pearson bound. It is tighter,
     so you abstain less for the same guarantee. Compare.

  3. Re-run PART 2 with only 100 calibration grasps, then 10,000.
     How does the required abstention change? That answers "how much
     calibration data do I need", which anyone deploying this asks.

  4. Use the week-6 NETWORK score instead of the geometric one. A
     worse score should force more abstention for the same guarantee -
     verify that, because it shows the two parts are separable: the
     guarantee holds regardless of model quality, you just pay for a
     bad model in throughput.

NEXT: GraspNet. Real objects, real clutter, physics-based labels - and
the distribution shift there is real rather than simulated.
""")