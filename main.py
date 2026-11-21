#!/usr/bin/env python3
"""GTA VI trophy rarity forecaster -- make.fun gta6-trophy-rarity-forecast-01.

Reads every *.in case in $MAKEFUN_INPUT_DIR (else /input) and writes one
*.out forecast per case to $MAKEFUN_OUTPUT_DIR (else /output).

Model (per trophy, pure stdlib, fully deterministic):
  Work on the logit scale and blend two views of the history:
  (a) an earned/players decomposition -- fit log earned-counts and log
      players-tracked separately against log(1 + hours since release),
      then rate = earned / players at the target. This captures dilution:
      when new players join faster than they earn trophies, rates fall
      even though nobody un-earns anything.
  (b) a straight logit-trend fit of the rate against log(1 + hours
      since release), the same family as the house curve baseline.
  Both fits are recency-weighted: the last reads before the checkpoint
  say the most about where the target is heading.
  Spread grows with the horizon (a random walk on the logit scale) plus
  the fit residual, so the 90% band stays honest instead of collapsing.
  Launch-night flat histories (no variation yet) get a dilution prior:
  Exophase refreshes lag the launch surge, so the first refresh dilutes
  every rate; the median is shifted down and the band widened.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
LO, HI = 0.01, 99.99


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_release(t):
    return max(0.0, (t - RELEASE).total_seconds() / 3600.0)


def logit(p):
    p = min(max(p, LO), HI) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid(l):
    if l >= 0:
        return 1.0 / (1.0 + math.exp(-l))
    e = math.exp(l)
    return e / (1.0 + e)


def ols(xs, ys, ws=None):
    """Least squares y ~ a + b*x, optionally recency-weighted.
    Returns (a, b, resid_var, n)."""
    n = len(xs)
    if n == 0:
        return (0.0, 0.0, 0.0, 0)
    if ws is None:
        ws = [1.0] * n
    if n == 1:
        return (ys[0], 0.0, 0.0, 1)
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        var = sum(w * (y - my) ** 2 for w, y in zip(ws, ys)) / sw
        return (my, 0.0, var, n)
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = my - b * mx
    var = sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys)) / sw
    return (a, b, var, n)


def recency_weights(n):
    """Linear ramp 0.5 -> 1.0: the last reads before the checkpoint say
    the most about where the target is heading."""
    if n <= 1:
        return [1.0] * n
    return [0.5 + 0.5 * i / (n - 1) for i in range(n)]


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def forecast_trophy(rates, players, xs, x_target, days, flat):
    """rates: list of earn rates (percent) in history order, None skipped.
    players: players_tracked per read. xs: log-time per read.
    Returns (median_percent, spread_logit)."""
    triples = [(x, r, p) for x, r, p in zip(xs, rates, players)
               if r is not None and p]
    if not triples:
        return (None, 1.2)
    last_rate = triples[-1][1]
    last_logit = logit(last_rate)
    xs_p = [t[0] for t in triples]
    ws = recency_weights(len(triples))

    # --- view (b): logit trend of the rate itself ---
    ys_b = [logit(clamp(r, 0.05, 99.95)) for _, r, _ in triples]
    a_b, b_b, var_b, n_b = ols(xs_p, ys_b, ws)
    b_b = clamp(b_b, -2.0, 2.0)
    med_b = a_b + b_b * x_target if n_b >= 2 else last_logit

    # --- view (a): earned-counts over players-tracked ---
    ys_a = [math.log(r / 100.0 * p + 0.5) for _, r, p in triples]
    a_a, b_a, var_a, n_a = ols(xs_p, ys_a, ws)
    b_a = clamp(b_a, -1.5, 2.5)
    med_a = None
    if n_a >= 2 and players:
        lp = [math.log(p) for _, _, p in triples]
        a_p, b_p, _, _ = ols(xs_p, lp, ws)
        b_p = clamp(b_p, -1.5, 2.5)
        players_target = math.exp(a_p + b_p * x_target)
        earned_target = math.exp(a_a + b_a * x_target) - 0.5
        if players_target > 0 and earned_target >= 0:
            med_a = logit(clamp(100.0 * earned_target / players_target,
                                LO, HI))

    # --- blend the two views in logit space ---
    cands = []
    if n_b >= 2:
        cands.append(med_b)
    if med_a is not None:
        cands.append(med_a)
    if not cands:
        med = last_logit
    else:
        med = sum(cands) / len(cands)

    # Never let the extrapolation run away from the last observed rate.
    med = clamp(med, last_logit - 2.5, last_logit + 2.5)

    # --- launch-night flat history: anticipate the first refresh ---
    # The page has not refreshed yet; when it does, a flood of new
    # players dilutes every rate (the one trophy at 100% excepted).
    if flat and last_rate < 99.5:
        med = clamp(med - 1.2, last_logit - 2.5, last_logit + 2.5)

    # --- spread: fit residual + random walk over the horizon ---
    resid = math.sqrt(var_b) if n_b >= 2 else 0.30
    s2 = resid * resid + (0.45 * math.sqrt(days)) ** 2
    if flat:
        s2 += 0.9 * 0.9
    spread = clamp(math.sqrt(s2), 0.25, 1.6)
    return (100.0 * sigmoid(med), spread)


def forecast_case(case):
    trophies = sorted(case["trophies"], key=lambda t: t["id"])
    history = sorted(case.get("history", []), key=lambda r: r["at"])
    target = parse_time(case["target"])
    x_target = math.log1p(hours_since_release(target))
    days = max(case.get("horizon_minutes", 0), 0) / 1440.0

    xs = [math.log1p(hours_since_release(parse_time(r["at"])))
          for r in history]
    players = [r.get("players_tracked") or 0 for r in history]

    flat = False
    if history:
        pvals = [p for p in players if p > 0]
        flat = (len(pvals) >= 2 and max(pvals) - min(pvals) <= 1)

    rows = []
    for t in trophies:
        tid = t["id"]
        rates = []
        for r in history:
            rs = r.get("rates") or []
            rates.append(rs[tid - 1] if tid - 1 < len(rs) else None)
        if not history:
            # No reads at all: fall back to a tier prior, wide band.
            prior = {"platinum": 0.2, "gold": 5.0,
                     "silver": 15.0, "bronze": 25.0}.get(t.get("tier"), 10.0)
            med, spread = prior, 1.2
        else:
            med, spread = forecast_trophy(rates, players, xs, x_target,
                                          days, flat)
            if med is None:
                med, spread = 10.0, 1.2
        mlogit = logit(med)
        qs = []
        prev = -1.0
        for z in Z:
            q = clamp(100.0 * sigmoid(mlogit + z * spread), LO, HI)
            q = round(q, 4)
            if q < prev:
                q = prev
            prev = q
            qs.append(q)
        rows.append({"id": tid, "q": qs})
    return {"trophies": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    if not names:
        print("no .in files in %s" % in_dir, file=sys.stderr)
        return 1
    for name in names:
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as f:
            case = json.load(f)
        out = forecast_case(case)
        text = json.dumps(out, separators=(",", ":"))
        if len(text.encode("utf-8")) > 65536:
            print("output too large for %s" % name, file=sys.stderr)
            return 1
        out_name = name[:-3] + ".out"
        with open(os.path.join(out_dir, out_name), "w",
                  encoding="utf-8") as f:
            f.write(text)
        print("%s -> %s (%d trophies, %d bytes)"
              % (name, out_name, len(out["trophies"]), len(text)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
