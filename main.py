"""Forecast of GTA VI (PS5) trophy earn rates on Exophase at a case's target.

Model, in one paragraph: the page only changes when Exophase refreshes it, so a
case's history is first collapsed to the distinct snapshots it contains.  On
the logit scale each trophy's rate is then extrapolated along ln(1 + hours
since release) with a recency- and size-weighted least-squares line whose
slope is shrunk toward the pooled slope of all trophies; the extrapolation is
capped by what each trophy can plausibly reach in week one (the platinum sits
below the 100%-completion trophy, which sits below the story finale and the
big grinds, and chapters are ordered).  The spread around the median combines
the diffusion the snapshots themselves show, the line's own extrapolation
variance, and extra width for trophies few players have earned.  A history
that never changed means a refresh is still owed, so it gets a dilution shift
and a wide band instead.  Everything is a fixed function of the case: no
clock, no randomness, so two runs write the same bytes.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

RELEASE = datetime(2026, 11, 19, tzinfo=timezone.utc)
NORM = NormalDist()

# Week-one ceilings (percent) for a forecast median, by trophy class.
CAP_PLATINUM = 4.0
CAP_COMPLETION = 6.0
CAP_STORY_GOLD = 65.0
CAP_SILVER = 45.0
CAP_DEFAULT = 95.0

# Fallback medians (percent) for a trophy with no usable rate anywhere.
PRIOR_BY_TIER = {"platinum": 0.03, "gold": 1.0, "silver": 1.2, "bronze": 2.5}

# Story chapters in narrative order: each is earned by everyone who earned the
# next, so their true rates (and our medians) are non-increasing along it.
CHAPTERS = [3, 4, 5, 6, 7]
COMPLETION_ID = 14
PLATINUM_ID = 1
GRIND_IDS = [15, 17, 25, 30, 31]


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_release(moment):
    return (moment - RELEASE).total_seconds() / 3600.0


def log_time(age_hours):
    return math.log1p(max(age_hours, 0.0))


def clamp_rate(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(x):
    if x >= 0:
        return 100.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return 100.0 * e / (1.0 + e)


def round4(x):
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def snapshots_of(history):
    """Distinct page states in the history, oldest first.

    Between Exophase refreshes every read repeats the previous one; keeping
    the repeats would let a long freeze outweigh real movement, so only the
    first read of each distinct (players, rates) state is kept.
    """
    snaps = []
    seen_key = None
    for read in history:
        key = (read.get("players_tracked"), tuple(read.get("rates") or ()))
        if key != seen_key:
            snaps.append(read)
            seen_key = key
    return snaps


def weighted_line(xs, ys, ws):
    """Weighted least squares y ~ a + b*x; returns (a, b, s2, w_sum, x_bar, sxx)."""
    w_sum = sum(ws)
    x_bar = sum(w * x for x, w in zip(xs, ws)) / w_sum
    y_bar = sum(w * y for y, w in zip(ys, ws)) / w_sum
    sxx = sum(w * (x - x_bar) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return y_bar, 0.0, 0.0, w_sum, x_bar, sxx
    sxy = sum(w * (x - x_bar) * (y - y_bar) for x, y, w in zip(xs, ys, ws))
    slope = sxy / sxx
    intercept = y_bar - slope * x_bar
    dof = max(len(xs) - 2, 1)
    s2 = sum(w * (y - (intercept + slope * x)) ** 2 for x, y, w in zip(xs, ys, ws)) / (w_sum / len(xs) * dof)
    return intercept, slope, s2, w_sum, x_bar, sxx


def robust_sigma(values, fallback):
    """A scale estimate that a few wild steps cannot blow up."""
    if not values:
        return fallback
    med = sorted(values)[len(values) // 2]
    devs = sorted(abs(v - med) for v in values)
    sigma = 1.4826 * devs[len(devs) // 2]
    return sigma if sigma > 1e-9 else fallback


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    levels = case.get("levels") or [0.05, 0.25, 0.5, 0.75, 0.95]
    zetas = [NORM.inv_cdf(tau) for tau in levels]
    horizon_days = max(case["horizon_minutes"], 0) / 1440.0
    target_u = log_time(hours_since_release(parse_time(case["target"])))

    snaps = snapshots_of(history)
    snap_u = [log_time(hours_since_release(parse_time(s["at"]))) for s in snaps]
    snap_age = [hours_since_release(parse_time(s["at"])) for s in snaps]
    snap_players = [s.get("players_tracked") or 0 for s in snaps]
    last = history[-1] if history else None
    last_rates = (last.get("rates") if last else None) or [None] * len(trophies)
    last_players = (last.get("players_tracked") if last else None) or 0

    # Diffusion of logit rates per sqrt-day, pooled over trophies and steps:
    # how much rates actually move between refreshes in this recording.
    steps = []
    for a, b in zip(snaps, snaps[1:]):
        dt_days = (parse_time(b["at"]) - parse_time(a["at"])).total_seconds() / 86400.0
        if dt_days <= 0:
            continue
        for ra, rb in zip(a.get("rates") or (), b.get("rates") or ()):
            if ra is not None and rb is not None and 0.0 < ra < 100.0 and 0.0 < rb < 100.0:
                steps.append((logit(rb) - logit(ra)) / math.sqrt(dt_days))
    sigma_rw = min(0.95, max(0.35, robust_sigma(steps, 0.5)))

    # A history whose page never changed is owed a refresh: the tracked base
    # will jump and dilute almost every rate at once.
    frozen = len(snaps) <= 1 and len(history) >= 2
    if frozen:
        span_h = (parse_time(history[-1]["at"]) - parse_time(history[0]["at"])).total_seconds() / 3600.0
        frozen = span_h >= 1.5

    # First pass per trophy: collect its snapshot series and fit its line.
    # Weights favour the recent snapshots (the curve bends as the base fills
    # with slower players, so launch-week steepness must not dominate), the
    # bigger player bases (their rates are measured, not rounded), and the
    # trophies with enough earners to carry signal.
    age_last = snap_age[-1] if snap_age else 0.0
    fits = []
    for i in range(len(trophies)):
        xs, ys, ws = [], [], []
        for s, u, age, players in zip(snaps, snap_u, snap_age, snap_players):
            rates = s.get("rates") or ()
            r = rates[i] if i < len(rates) else None
            if r is None:
                continue
            earners = r / 100.0 * players
            w = math.sqrt(players / max(last_players, 1)) if players else 0.3
            w *= 0.5 ** (max(age_last - age, 0.0) / 18.0)
            w *= earners / (earners + 25.0)
            xs.append(u)
            ys.append(logit(r))
            ws.append(max(w, 1e-4))
        fits.append((xs, ys, ws))

    slopes = []
    for xs, ys, ws in fits:
        if len(xs) >= 3 and max(xs) - min(xs) > 0.05:
            slopes.append(weighted_line(xs, ys, ws)[1])
    # With no fittable trophy at all there is no drift worth extrapolating:
    # the pooled slope is then 0 and the spread does the work.
    pooled_slope = sorted(slopes)[len(slopes) // 2] if slopes else 0.0
    pooled_slope = min(2.0, max(-1.0, pooled_slope))

    # Most earners any snapshot shows per trophy: a trophy nobody has earned
    # yet needs a prior, not a line through clamped zeroes.
    earners_max = []
    for i in range(len(trophies)):
        best = 0.0
        for s, players in zip(snaps, snap_players):
            rates = s.get("rates") or ()
            r = rates[i] if i < len(rates) else None
            if r is not None:
                best = max(best, r / 100.0 * players)
        earners_max.append(best)

    medians = []
    spreads = []
    for i, trophy in enumerate(trophies):
        xs, ys, ws = fits[i]
        last_rate = last_rates[i] if i < len(last_rates) else None
        earners_last = (last_rate / 100.0 * last_players) if last_rate is not None else 0.0

        if last_rate is not None and last_rate >= 99.99:
            # Earned by effectively every tracked player; it can only slip a
            # little as the base widens.
            medians.append(logit(99.93))
            spreads.append(0.55)
            continue

        if earners_max[i] < 1.0:
            # No tracked player has it yet: a small prior median for its
            # class and a wide band, whatever the clamped zeroes suggest.
            prior = min(PRIOR_BY_TIER.get(trophy.get("tier"), 2.0), 0.08)
            if trophy["id"] == PLATINUM_ID:
                prior = PRIOR_BY_TIER["platinum"]
            medians.append(logit(prior))
            spreads.append(1.35)
            continue

        if len(xs) >= 3 and max(xs) - min(xs) > 0.05:
            a, b, s2, w_sum, x_bar, sxx = weighted_line(xs, ys, ws)
            # Shrink the slope toward the pooled one: a single trophy's line
            # is noisy, the field's drift is not.
            shrink = sxx / (sxx + 0.35)
            b = pooled_slope + (b - pooled_slope) * shrink
            b = min(2.2, max(-1.2, b))
            centre = a + b * x_bar
            # Growth keeps bending down as a trophy saturates, so the fitted
            # slope is damped over the extrapolation leg.
            pred = centre + b * (target_u - x_bar) * 0.85
            resid = math.sqrt(max(s2, 0.0144))
            # Prediction variance on a per-observation scale: the weights are
            # relative, so divide the weighted scatter back by the mean weight.
            sxx_per = sxx / max(w_sum / len(xs), 1e-9)
            var = resid * resid * (1.0 / len(xs) + (target_u - x_bar) ** 2 / max(sxx_per, 1e-9))
            sigma_fit = math.sqrt(var)
        elif xs:
            pred = ys[-1] + pooled_slope * (target_u - xs[-1])
            sigma_fit = 0.55
        elif last_rate is not None:
            pred = logit(last_rate) + pooled_slope * 0.5
            sigma_fit = 0.8
        else:
            pred = logit(PRIOR_BY_TIER.get(trophy.get("tier"), 2.0))
            sigma_fit = 1.1

        # Ceilings from what the trophy is: a median above its class cap is a
        # fit artefact, not a forecast.
        tier = trophy.get("tier")
        if trophy["id"] == PLATINUM_ID or tier == "platinum":
            cap = CAP_PLATINUM
        elif trophy["id"] == COMPLETION_ID:
            cap = CAP_COMPLETION
        elif tier == "gold":
            cap = CAP_STORY_GOLD
        elif tier == "silver":
            cap = CAP_SILVER
        elif last_rate is not None and last_rate >= 90.0:
            cap = 99.5
        else:
            cap = CAP_DEFAULT
        pred = min(pred, logit(cap))

        base = 0.18 + min(0.85, 0.5 / math.sqrt(max(earners_last, 1.0)))
        sigma = math.sqrt(sigma_rw * sigma_rw * horizon_days + sigma_fit * sigma_fit + base * base)
        if frozen:
            pred -= 0.5
            sigma = max(sigma, 1.05)
        medians.append(pred)
        spreads.append(min(2.4, max(0.3, sigma)))

    # Ordering the truth obeys: chapters in order, the finale above the
    # completion trophy, the completion trophy below the big grinds, and the
    # platinum below everything.
    by_id = {t["id"]: k for k, t in enumerate(trophies)}
    for earlier, later in zip(CHAPTERS, CHAPTERS[1:]):
        if earlier in by_id and later in by_id:
            medians[by_id[later]] = min(medians[by_id[later]], medians[by_id[earlier]])
    if COMPLETION_ID in by_id:
        ci = by_id[COMPLETION_ID]
        ceilings = [medians[by_id[g]] for g in GRIND_IDS if g in by_id]
        if 7 in by_id:
            ceilings.append(medians[by_id[7]])
        if ceilings:
            medians[ci] = min(medians[ci], min(ceilings))
    if PLATINUM_ID in by_id:
        pi = by_id[PLATINUM_ID]
        others = [medians[k] for k in range(len(trophies)) if k != pi]
        if others:
            medians[pi] = min(medians[pi], min(others))

    rows = []
    for i, trophy in enumerate(trophies):
        centre, sigma = medians[i], spreads[i]
        q = []
        for z in zetas:
            value = inv_logit(centre + z * sigma)
            q.append(round4(min(99.999, max(0.001, value))))
        for k in range(1, len(q)):
            if q[k] < q[k - 1]:
                q[k] = q[k - 1]
        rows.append({"id": trophy["id"], "q": q})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        result = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(result, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
