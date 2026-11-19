"""GTA VI trophy rarity forecaster (gta6-trophy-rarity-forecast-01).

Reads every case file (<id>.in) in the input directory and writes, for each
trophy in the case, five quantiles of its Exophase earn rate (percent of
tracked players who hold it) at the case's target instant. Python 3 standard
library only, no network, deterministic: the same case always produces the
same bytes.

Model, in short:

  * Exophase only changes its counts on a refresh, so the house's reads are
    a step function: consecutive identical reads are collapsed into one
    snapshot before anything is fitted.
  * Each trophy's logit(rate) is fitted against ln(1 + hours since release)
    with a weighted least-squares line. Weights reward recency (18 h
    half-life), a larger tracked-player base, and more earners behind the
    rate. Slopes are shrunk toward the pooled cross-trophy slope, clamped,
    and damped over the extrapolation leg (saturation).
  * Medians are then held to structural caps and orderings that the trophy
    list itself implies: the platinum cannot exceed anything, 100%
    completion cannot exceed the finale or the big grinds, chapter medians
    cannot rise as the story advances.
  * The spread combines pooled snapshot-to-snapshot logit diffusion, the
    fit's own extrapolation variance, and a low-earner width term.
  * Special cases: a trophy at the ceiling stays pinned; a trophy nobody
    has earned yet gets a small class prior; a fully frozen history (a
    refresh is owed and the tracked base will jump) shifts medians down
    for dilution and widens the spread.

Run: python3 -P -s main.py with MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR set,
or /input and /output in the grader's container.
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

DEFAULT_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
_NORM = NormalDist()

# The scorer clamps rates to [0.01, 99.99] before taking logits; so do we.
RATE_LO = 0.01
RATE_HI = 99.99

# --- model constants -------------------------------------------------------
RECENCY_HALF_LIFE_H = 18.0     # weight half-life of a snapshot, in hours
SLOPE_DAMPING = 0.85           # saturation damping over the forecast leg
SLOPE_MIN = -2.0               # clamp on a shrunk slope (logit per ln-hour)
SLOPE_MAX = 3.0
MAX_MEDIAN_MOVE = 3.0          # a median may not move more than this (logit)
SIGMA_FLOOR = 0.30
SIGMA_CAP = 2.40
DIFFUSION_FALLBACK = 0.5       # logit per sqrt-day, the house walk's value
FROZEN_SHIFT = -0.5            # logit shift when a refresh is owed
FROZEN_SIGMA = 1.05            # spread floor when a refresh is owed
PINNED_MEDIAN = 99.93          # a trophy at the ceiling stays at the ceiling
PINNED_SIGMA = 0.55
ZERO_SIGMA = 1.35              # spread for a trophy with no earners yet
SINGLE_POINT_SLOPE_SHARE = 0.5 # a one-point trophy takes half the pool slope

# Median caps (percent) by trophy class.
CAP_PLATINUM = 4.0
CAP_COMPLETION = 6.0
CAP_GOLD = 65.0
CAP_SILVER = 45.0
CAP_HIGH = 99.5                # a trophy already at >= 90% stays under this

# Median priors (percent) for a trophy with no earners in any read.
PRIOR_PLATINUM = 0.03
PRIOR_COMPLETION = 0.04
PRIOR_GOLD = 0.08
PRIOR_SILVER = 0.08
PRIOR_BRONZE = 0.12

_CHAPTER_RE = re.compile(r"complete chapter\s*(\d+)", re.IGNORECASE)
_FINALE_RE = re.compile(r"complete the final mission\s*\.?\s*$", re.IGNORECASE)
_COMPLETION_RE = re.compile(r"100\s*%\s*completion", re.IGNORECASE)
_COUNT_RE = re.compile(r"\d[\d,]*")


# --- small helpers ---------------------------------------------------------

def parse_instant(text):
    """Parse an ISO 8601 timestamp to an aware UTC datetime, or None."""
    if not isinstance(text, str):
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def clamp_rate(rate):
    return min(RATE_HI, max(RATE_LO, rate))


def logit_pct(rate):
    p = clamp_rate(rate) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(value):
    if value >= 0:
        return 100.0 / (1.0 + math.exp(-value))
    e = math.exp(value)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up, returning an int for whole numbers."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def median_of(values):
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


# --- trophy classification -------------------------------------------------

def classify(trophy):
    """Sort a trophy into the classes the caps and orderings care about.

    Returns a dict: kind ('platinum', 'completion', 'finale', 'chapter' or
    'other'), chapter (int or None), grind (bool), cap (percent or None),
    prior (percent).
    """
    tier = str(trophy.get("tier", "")).lower()
    desc = str(trophy.get("description", "") or "")
    name = str(trophy.get("name", "") or "")
    text = desc or name

    if tier == "platinum":
        return {"kind": "platinum", "chapter": None, "grind": False,
                "cap": CAP_PLATINUM, "prior": PRIOR_PLATINUM}
    if _COMPLETION_RE.search(text):
        return {"kind": "completion", "chapter": None, "grind": False,
                "cap": CAP_COMPLETION, "prior": PRIOR_COMPLETION}

    chapter = None
    m = _CHAPTER_RE.search(text)
    if m:
        chapter = int(m.group(1))
    is_finale = chapter is None and _FINALE_RE.search(text) is not None

    big_count = False
    for token in _COUNT_RE.findall(text):
        try:
            if int(token.replace(",", "")) >= 50:
                big_count = True
        except ValueError:
            pass
    grind = tier == "silver" or big_count

    if tier == "gold":
        cap, prior = CAP_GOLD, PRIOR_GOLD
    elif tier == "silver":
        cap, prior = CAP_SILVER, PRIOR_SILVER
    else:
        cap, prior = None, PRIOR_BRONZE

    kind = "chapter" if chapter is not None else ("finale" if is_finale else "other")
    return {"kind": kind, "chapter": chapter, "grind": grind,
            "cap": cap, "prior": prior}


# --- snapshots ---------------------------------------------------------------

def collapse_snapshots(history):
    """Collapse consecutive identical reads into snapshots.

    A snapshot keeps the first and last read times that showed it, the
    players tracked, and the rates tuple.
    """
    snapshots = []
    for read in history:
        if not isinstance(read, dict):
            continue
        rates = read.get("rates")
        if not isinstance(rates, list):
            continue
        when = parse_instant(read.get("at")) or parse_instant(read.get("slot"))
        players = read.get("players_tracked")
        if not isinstance(players, (int, float)) or players <= 0:
            players = None
        key = (players, tuple(rates))
        if snapshots and snapshots[-1]["key"] == key:
            snapshots[-1]["last_time"] = when
        else:
            snapshots.append({"key": key, "first_time": when, "last_time": when,
                              "players": players, "rates": rates})
    return snapshots


def release_instant(case):
    game = case.get("game") or {}
    text = str(game.get("release_date", "") or "")
    dt = parse_instant(text)
    if dt is not None:
        return dt
    return datetime(2026, 11, 19, tzinfo=timezone.utc)


def x_of(when, release):
    """The fit's x coordinate: ln(1 + hours since release), floored."""
    if when is None:
        return 0.0
    hours = (when - release).total_seconds() / 3600.0
    return math.log1p(max(hours, -0.5))


# --- weighted least squares --------------------------------------------------

def fit_line(points):
    """Weighted least squares of y on x. points: [(x, y, w)] with n >= 2.

    Returns dict(a, b, s, var_b, xbar, sxx, n) or None when the points share
    a single x. Weights are rescaled to sum to n so the variance formulas
    read in count units.
    """
    n = len(points)
    sw = sum(w for _, _, w in points)
    if sw <= 0:
        return None
    scale = n / sw
    pts = [(x, y, w * scale) for x, y, w in points]
    xbar = sum(w * x for x, _, w in pts) / n
    ybar = sum(w * y for _, y, w in pts) / n
    sxx = sum(w * (x - xbar) ** 2 for x, _, w in pts)
    if sxx < 1e-9:
        return None
    sxy = sum(w * (x - xbar) * (y - ybar) for x, y, w in pts)
    b = sxy / sxx
    a = ybar - b * xbar
    s2 = sum(w * (y - (a + b * x)) ** 2 for x, y, w in pts) / n
    s = math.sqrt(max(s2, 0.0025))
    return {"a": a, "b": b, "s": s, "var_b": s * s / sxx,
            "xbar": xbar, "sxx": sxx, "n": n}


def point_weight(snap, checkpoint, rate, players_max):
    """One snapshot's weight for one trophy: recency x base size x earners."""
    last_time = snap["last_time"]
    if checkpoint is not None and last_time is not None:
        age_h = max(0.0, (checkpoint - last_time).total_seconds() / 3600.0)
    else:
        age_h = 0.0
    recency = 0.5 ** (age_h / RECENCY_HALF_LIFE_H)

    players = snap["players"]
    if players is not None and players_max:
        base = min(1.0, max(0.1, players / players_max))
    else:
        base = 0.5

    if players is not None and rate is not None:
        earners = rate / 100.0 * players
        earner_factor = max(earners / (earners + 25.0), 0.05)
    else:
        earner_factor = 0.5
    return recency * base * earner_factor


def low_earner_width(last_rate, last_players):
    """Spread term for a rate held by few players: it can still move a lot."""
    if last_rate is None:
        return 0.45
    if last_players is None:
        return 0.45
    earners = last_rate / 100.0 * last_players
    return min(1.0 / math.sqrt(max(earners, 1.0)), 1.25)


def spread_from_variances(*variances):
    """Total spread from independent variance contributions.

    Every sigma in the forecast is a quadrature sum of independent terms
    (diffusion, fit variance, low-earner width, ...); keeping the
    combination in one place means the call sites just name their terms.
    """
    return math.sqrt(sum(variances))


# --- the forecast ------------------------------------------------------------

def forecast_case(case):
    trophies = case["trophies"]
    n_trophies = len(trophies)
    classes = [classify(t) for t in trophies]

    checkpoint = parse_instant(case.get("at"))
    target = parse_instant(case.get("target"))
    horizon_minutes = case.get("horizon_minutes")
    if not isinstance(horizon_minutes, (int, float)):
        if checkpoint is not None and target is not None:
            horizon_minutes = (target - checkpoint).total_seconds() / 60.0
        else:
            horizon_minutes = 0.0
    horizon_days = max(float(horizon_minutes), 0.0) / 1440.0
    release = release_instant(case)
    if target is not None:
        x_target = x_of(target, release)
    elif checkpoint is not None:
        x_target = x_of(checkpoint, release) + math.log1p(
            max(float(horizon_minutes), 0.0) / 60.0)
    else:
        x_target = 0.0

    history = case.get("history") or []
    snapshots = collapse_snapshots(history)
    frozen = len(snapshots) == 1 and len(history) >= 1
    players_max = None
    for snap in snapshots:
        if snap["players"] is not None:
            players_max = snap["players"] if players_max is None else max(
                players_max, snap["players"])

    # Per-trophy observation series: (snapshot index, rate).
    series = []
    for i in range(n_trophies):
        obs = []
        for j, snap in enumerate(snapshots):
            rates = snap["rates"]
            rate = rates[i] if i < len(rates) else None
            if isinstance(rate, (int, float)):
                obs.append((j, float(rate)))
        series.append(obs)

    # (a) pooled snapshot-to-snapshot logit diffusion, per sqrt-day.
    steps = []
    for i in range(n_trophies):
        obs = series[i]
        for (j0, r0), (j1, r1) in zip(obs, obs[1:]):
            t0 = snapshots[j0]["first_time"]
            t1 = snapshots[j1]["first_time"]
            if t0 is None or t1 is None:
                continue
            dt_days = (t1 - t0).total_seconds() / 86400.0
            if dt_days < 0.005:
                continue
            if not (RATE_LO < r0 < RATE_HI and RATE_LO < r1 < RATE_HI):
                continue
            steps.append((logit_pct(r1) - logit_pct(r0)) / math.sqrt(dt_days))
    if len(steps) >= 5:
        centre = median_of(steps)
        mad = median_of([abs(s - centre) for s in steps])
        sigma_day = min(2.5, max(0.08, 1.4826 * mad))
    else:
        sigma_day = DIFFUSION_FALLBACK
    diffusion = sigma_day * math.sqrt(horizon_days)

    # Per-trophy fits and the pooled slope.
    fits = {}
    for i in range(n_trophies):
        obs = series[i]
        if len(obs) < 2:
            continue
        points = []
        for j, rate in obs:
            snap = snapshots[j]
            x = x_of(snap["last_time"], release)
            w = point_weight(snap, checkpoint, rate, players_max)
            points.append((x, logit_pct(rate), w))
        fit = fit_line(points)
        if fit is not None:
            fits[i] = fit

    if fits:
        slope_list = [(f["b"], f["var_b"]) for f in fits.values()]
        mean_b = sum(b for b, _ in slope_list) / len(slope_list)
        var_obs = sum((b - mean_b) ** 2 for b, _ in slope_list) / len(slope_list)
        mean_var = sum(v for _, v in slope_list) / len(slope_list)
        tau2 = max(var_obs - mean_var, 0.01)
        pool_w = [1.0 / (v + tau2) for _, v in slope_list]
        pooled_slope = sum(w * b for w, (b, _) in zip(pool_w, slope_list)) / sum(pool_w)
    else:
        tau2 = 0.09
        pooled_slope = 0.0

    def shrunk_slope(fit):
        b = (fit["b"] / fit["var_b"] + pooled_slope / tau2) / (
            1.0 / fit["var_b"] + 1.0 / tau2)
        return min(SLOPE_MAX, max(SLOPE_MIN, b))

    medians = [None] * n_trophies   # in logit units until caps are applied
    sigmas = [None] * n_trophies
    pinned = [False] * n_trophies
    last_rates = [None] * n_trophies

    for i in range(n_trophies):
        obs = series[i]
        cls = classes[i]
        if obs:
            last_j, last_rate = obs[-1]
            last_rates[i] = last_rate
            last_players = snapshots[last_j]["players"]
        else:
            last_rate = None
            last_players = None

        # No earners ever: every rate missing or zero, or fewer than one
        # expected earner behind the largest rate seen.
        max_earners = None
        for j, rate in obs:
            players = snapshots[j]["players"]
            if players is not None:
                e = rate / 100.0 * players
                max_earners = e if max_earners is None else max(max_earners, e)
        no_earners = (not obs) or all(r <= 0.0 for _, r in obs) or (
            max_earners is not None and max_earners < 1.0)
        if no_earners:
            medians[i] = logit_pct(cls["prior"])
            sigmas[i] = ZERO_SIGMA
            continue

        # A trophy at the ceiling stays at the ceiling.
        if last_rate is not None and last_rate >= RATE_HI:
            medians[i] = logit_pct(PINNED_MEDIAN)
            sigmas[i] = PINNED_SIGMA
            pinned[i] = True
            continue

        low = low_earner_width(last_rate, last_players)
        fit = fits.get(i)
        if frozen or fit is None and len(obs) == 1:
            # One snapshot only: the last rate, plus (when other trophies
            # did fit) a damped share of the pooled slope.
            j, rate = obs[-1]
            x_last = x_of(snapshots[j]["last_time"], release)
            med = logit_pct(rate)
            if not frozen and fits:
                med += SLOPE_DAMPING * pooled_slope * SINGLE_POINT_SLOPE_SHARE * (
                    x_target - x_last)
            medians[i] = med
            sigmas[i] = spread_from_variances(diffusion ** 2, low ** 2, 0.10 ** 2)
        elif fit is not None:
            b_shrunk = shrunk_slope(fit)
            x_last = x_of(snapshots[obs[-1][0]]["last_time"], release)
            # The shrunk line through the fit's centroid, read at the last
            # snapshot, then the damped slope over the remaining leg.
            ybar = fit["a"] + fit["b"] * fit["xbar"]
            anchor = ybar + b_shrunk * (x_last - fit["xbar"])
            med = anchor + SLOPE_DAMPING * b_shrunk * (x_target - x_last)
            # Do not let an extrapolation run away from the last read.
            base = logit_pct(last_rate)
            med = base + min(MAX_MEDIAN_MOVE, max(-MAX_MEDIAN_MOVE, med - base))
            var_mean = fit["s"] ** 2 * (
                1.0 / fit["n"] + (x_target - fit["xbar"]) ** 2 / fit["sxx"])
            medians[i] = med
            sigmas[i] = spread_from_variances(diffusion ** 2, var_mean, low ** 2)
        else:
            # Several observations but no fit (a single distinct x).
            medians[i] = logit_pct(last_rate)
            sigmas[i] = spread_from_variances(diffusion ** 2, low ** 2, 0.10 ** 2)

        if frozen and not pinned[i]:
            medians[i] += FROZEN_SHIFT
            sigmas[i] = max(sigmas[i], FROZEN_SIGMA)
        sigmas[i] = min(SIGMA_CAP, max(SIGMA_FLOOR, sigmas[i]))

    # Caps and orderings, applied to medians in percent.
    pct = [inv_logit_pct(m) for m in medians]
    for i in range(n_trophies):
        if pinned[i]:
            continue
        cap = classes[i]["cap"]
        if last_rates[i] is not None and last_rates[i] >= 90.0:
            cap = CAP_HIGH if cap is None else min(cap, CAP_HIGH)
        if cap is not None:
            pct[i] = min(pct[i], cap)

    # Story chain: chapters in order, then the finale, non-increasing.
    chain = sorted(
        (classes[i]["chapter"], i) for i in range(n_trophies)
        if classes[i]["kind"] == "chapter")
    chain_idx = [i for _, i in chain]
    finale_idx = [i for i in range(n_trophies) if classes[i]["kind"] == "finale"]
    chain_idx += finale_idx
    for k in range(1, len(chain_idx)):
        prev, cur = chain_idx[k - 1], chain_idx[k]
        if pct[cur] > pct[prev]:
            pct[cur] = pct[prev]

    # 100% completion sits below the finale and the big grinds.
    grind_idx = [i for i in range(n_trophies) if classes[i]["grind"]]
    for i in range(n_trophies):
        if classes[i]["kind"] != "completion":
            continue
        bound = None
        for j in finale_idx + grind_idx:
            if j != i:
                bound = pct[j] if bound is None else min(bound, pct[j])
        if bound is not None:
            pct[i] = min(pct[i], bound)

    # The platinum sits below everything else.
    for i in range(n_trophies):
        if classes[i]["kind"] != "platinum":
            continue
        others = [pct[j] for j in range(n_trophies) if j != i]
        if others:
            pct[i] = min(pct[i], min(others))

    levels = case.get("levels") or DEFAULT_LEVELS
    zetas = [_NORM.inv_cdf(float(level)) for level in levels]

    rows = []
    for i, trophy in enumerate(trophies):
        med_logit = logit_pct(pct[i])
        sigma = sigmas[i]
        qs = []
        for z in zetas:
            q = inv_logit_pct(med_logit + z * sigma)
            qs.append(round4(min(RATE_HI, max(RATE_LO, q))))
        # Non-decreasing, and strictly increasing wherever there is room.
        for k in range(1, len(qs)):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        for k in range(1, len(qs)):
            if qs[k] <= qs[k - 1] and qs[k] < RATE_HI:
                qs[k] = round4(min(RATE_HI, qs[k - 1] + 0.0001))
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def fallback_case(case):
    """Persistence rows from the last read, used if the model ever fails:
    a valid forecast beats an invalid case."""
    rows = []
    history = case.get("history") or []
    last = history[-1] if history else {}
    rates = last.get("rates") or []
    for i, trophy in enumerate(case.get("trophies") or []):
        cls = classify(trophy)
        rate = rates[i] if i < len(rates) else None
        value = clamp_rate(rate) if isinstance(rate, (int, float)) and rate > 0 \
            else cls["prior"]
        rows.append({"id": trophy["id"], "q": [round4(value)] * 5})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case)
        except Exception:
            forecast = fallback_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
