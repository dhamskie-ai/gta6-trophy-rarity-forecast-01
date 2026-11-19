"""GTA VI trophy rarity forecast — gta6-trophy-rarity-forecast-01.

Model (fresh, stdlib only, no network):
  For each trophy, earn rate = earned / tracked.  We fit the case's OWN
  history in count space, not on the rate itself: least-squares lines of
  log(earned) and log(tracked) against ln(1 + hours since release), each
  slope shrunk toward zero when the history is short or flat, then
  extrapolate both counts to the target and divide.  When tracked players
  grow faster than earners, the rate dilutes even while earned counts
  rise — a logit-linear fit on the rate itself cannot see that, and a
  persistence forecast cannot see it either.

  When the history is flat (launch night: every read identical), the
  trend fit is exactly zero, so we fall back to a dilution-aware
  persistence median: modestly below the last rate (less for common
  early-story trophies, more for rare ones), with a wide spread skewed
  downward, because a single Exophase refresh can cut most rates
  several-fold as new players flood in.  The size of one past refresh
  is never hardcoded as a prediction.

  Spread: a random-walk term growing with sqrt(horizon) plus a
  dilution term that grows with horizon and with how fast tracked
  players were growing in the case's history, skewed downward.

Special cases: the 100% trophy stays 100%; the platinum is capped low
(nobody plats in a week); every quantile is clamped to [0.01, 99.99]
on the logit scale and [0, 100] in percent, and forced non-decreasing.

Safety: any malformed case falls back to a safe persistence forecast
instead of crashing, so an output file is always written.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]

# Random-walk spread, logit units per sqrt(day).
RW_SIGMA = 0.50
# Extra dilution spread per sqrt(day), scaled by tracked-growth elasticity.
DIL_SIGMA = 0.30
# Flat-history fallback: base spread (logit) before horizon scaling.
FLAT_SIGMA = 0.95


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    if lg >= 20.0:
        return 100.0
    if lg <= -20.0:
        return 0.0
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    """Round to 4 decimals, half up; whole numbers come back as ints."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def parse_time(s):
    try:
        if isinstance(s, str) and s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s)
    except Exception:
        return None


def release_time(case):
    game = case.get("game") or {}
    rel = game.get("release_date")
    dt = None
    if isinstance(rel, str):
        dt = parse_time(rel)
        if dt is None:
            try:
                dt = datetime.fromisoformat(rel).replace(tzinfo=timezone.utc)
            except Exception:
                dt = None
    if dt is None:
        dt = datetime(2026, 11, 19, tzinfo=timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def ols(xs, ys):
    """Least squares slope/intercept; returns (slope, intercept, n, sxx)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0, 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return 0.0, my, n, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return slope, my - slope * mx, n, sxx


def shrink(slope, sxx, prior_scale=0.35):
    """Shrink a slope toward zero; weak evidence (small spread in x) shrinks hardest."""
    if sxx <= 1e-12:
        return 0.0
    w = sxx / (sxx + prior_scale)
    return slope * w


def flat_median_factor(last_rate, tier):
    """Dilution-aware median as a fraction of the last rate, flat history only."""
    if last_rate >= 100.0:
        return 1.0
    if tier == "platinum":
        return 0.22
    if last_rate >= 2.0:
        return 0.40
    if last_rate >= 1.0:
        return 0.30
    if last_rate >= 0.7:
        return 0.25
    return 0.22


def quantiles_from_logit(median_logit, sigma_down, sigma_up):
    qs = []
    for z in Z:
        s = sigma_down if z < 0 else sigma_up
        qs.append(inv_logit(median_logit + z * s))
    qs = [min(100.0, max(0.0, q)) for q in qs]
    for i in range(1, len(qs)):
        if qs[i] < qs[i - 1]:
            qs[i] = qs[i - 1]
    return [round4(q) for q in qs]


def forecast_case(case):
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    horizon_min = float(case.get("horizon_minutes") or 0)
    days = max(horizon_min / 1440.0, 0.0)
    release = release_time(case)

    # Target x in the same ln(1 + hours) scale as the history.
    target_at = parse_time(case.get("target")) or parse_time(case.get("at"))
    if target_at is None:
        target_at = release
    hours_target = max(0.0, (target_at - release).total_seconds() / 3600.0)
    x_target = math.log1p(hours_target)

    # Build per-read x values and tracked counts.
    xs, tracked = [], []
    for read in history:
        at = parse_time(read.get("at"))
        players = read.get("players_tracked")
        if at is None or players is None:
            continue
        try:
            players = float(players)
        except Exception:
            continue
        if players <= 0:
            continue
        hours = max(0.0, (at - release).total_seconds() / 3600.0)
        xs.append(math.log1p(hours))
        tracked.append(players)

    distinct_x = len(set(round(x, 6) for x in xs))
    span_x = (max(xs) - min(xs)) if xs else 0.0
    # A trend is only real evidence if the COUNTS actually moved, not
    # merely the clock: launch-night reads at different times with
    # identical players and rates carry no trend at all.
    distinct_tracked = len(set(tracked))
    # True when one trophy's own rate moved between reads.
    rates_vary = False
    if history:
        ntr = len(trophies)
        for i in range(ntr):
            vals = set()
            for read in history:
                rates = read.get("rates") or []
                if i < len(rates) and rates[i] is not None:
                    vals.add(rates[i])
            if len(vals) > 1:
                rates_vary = True
                break
    informative = (len(xs) >= 3 and distinct_x >= 3 and span_x > 0.05
                   and (distinct_tracked >= 2 or rates_vary))

    # Tracked-count trend (shared by all trophies).
    if len(xs) >= 2:
        b_t, a_t, _, sxx_t = ols(xs, [math.log(t) for t in tracked])
        b_t = shrink(max(-3.0, min(3.0, b_t)), sxx_t)
    else:
        b_t, a_t = 0.0, math.log(tracked[-1]) if tracked else math.log(638.0)
    log_t_target = a_t + b_t * x_target
    t_target = math.exp(max(-5.0, min(25.0, log_t_target)))
    t_last = tracked[-1] if tracked else t_target

    rows = []
    for i, trophy in enumerate(trophies):
        tid = trophy.get("id")
        tier = (trophy.get("tier") or "").lower()
        # Last non-null rate for this trophy.
        last_rate = None
        for read in reversed(history):
            rates = read.get("rates") or []
            if i < len(rates) and rates[i] is not None:
                last_rate = float(rates[i])
                break
        if last_rate is None:
            last_rate = {"platinum": 0.1, "gold": 0.5, "silver": 0.5}.get(tier, 1.0)

        if last_rate >= 100.0:
            rows.append({"id": tid, "q": [100, 100, 100, 100, 100]})
            continue
        if last_rate <= 0.0:
            rows.append({"id": tid, "q": quantiles_from_logit(logit(0.03), 1.2, 0.9)})
            continue

        if informative:
            # Per-trophy earned-count trend.  Earned counts are
            # rate/100 * players; the +0.5 keeps log(0) finite and is
            # subtracted back out after extrapolation, so a flat series
            # extrapolates to exactly its own level, not level + 0.5.
            ys = []
            ok = True
            for j, read in enumerate(history):
                rates = read.get("rates") or []
                if i >= len(rates) or rates[i] is None:
                    ok = False
                    break
                earned = max(0.0, float(rates[i])) / 100.0 * tracked[j]
                ys.append(math.log(earned + 0.5))
            if ok and len(ys) == len(xs) and len(ys) >= 3:
                b_e, a_e, _, sxx_e = ols(xs, ys)
                b_e = shrink(max(-3.0, min(3.0, b_e)), sxx_e)
                log_e_target = a_e + b_e * x_target
                e_target = math.exp(max(-5.0, min(25.0, log_e_target))) - 0.5
                e_target = max(e_target, 0.05)
                trend_rate = 100.0 * e_target / max(t_target, 1.0)
            else:
                trend_rate = None
        else:
            trend_rate = None

        if trend_rate is not None and math.isfinite(trend_rate):
            # Blend trend with the dilution-aware flat prior: the longer
            # and more varied the history, the more we trust the trend.
            w = min(1.0, span_x / 1.0)  # span in ln(1+h); ~1 once history spans a day+
            w = 0.55 + 0.45 * w
            flat_rate = last_rate * flat_median_factor(last_rate, tier)
            # Dilution over a long horizon also pulls the flat prior down.
            flat_rate *= 1.0 / (1.0 + 0.9 * days)
            med = math.exp(w * math.log(max(trend_rate, 0.01))
                           + (1.0 - w) * math.log(max(flat_rate, 0.01)))
            sigma_rw = RW_SIGMA * math.sqrt(days)
            sigma_dil = DIL_SIGMA * math.sqrt(days) * min(1.5, max(0.0, b_t))
            sigma = math.sqrt(sigma_rw ** 2 + sigma_dil ** 2 + 0.15 ** 2)
            sigma_down, sigma_up = sigma * 1.2, sigma * 0.9
        else:
            base_f = flat_median_factor(last_rate, tier)
            # A longer horizon means more refreshes and more dilution.
            med = last_rate * base_f / (1.0 + 0.9 * days)
            sigma = FLAT_SIGMA + 0.35 * math.sqrt(days)
            sigma_down, sigma_up = sigma * 1.4, sigma * 0.8

        # Platinum: nobody plats in a week; cap the median low.
        if tier == "platinum":
            med = min(med, 2.0)
        med = min(max(med, 0.01), 99.99)
        rows.append({"id": tid, "q": quantiles_from_logit(logit(med), sigma_down, sigma_up)})

    rows.sort(key=lambda r: (r["id"] is None, r["id"]))
    return {"trophies": rows}


def safe_forecast(case):
    """Persistence fallback used when anything about a case is unusable."""
    rows = []
    history = case.get("history") or []
    last = history[-1] if history else {}
    rates = last.get("rates") or []
    for i, trophy in enumerate(case.get("trophies") or []):
        rate = rates[i] if i < len(rates) and rates[i] is not None else 1.0
        rate = float(rate)
        if rate >= 100.0:
            rows.append({"id": trophy.get("id"), "q": [100, 100, 100, 100, 100]})
        else:
            rows.append({"id": trophy.get("id"),
                         "q": quantiles_from_logit(logit(max(rate, 0.01)), 1.0, 0.8)})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    in_dir = INPUT_DIR
    files = sorted(in_dir.glob("*.in")) if in_dir.is_dir() else []
    for path in files:
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            forecast = forecast_case(case)
            # Validate shape before writing.
            assert isinstance(forecast, dict) and set(forecast) == {"trophies"}
            assert len(forecast["trophies"]) == len(case.get("trophies") or [])
            for row in forecast["trophies"]:
                q = row["q"]
                assert len(q) == 5 and all(math.isfinite(v) and 0.0 <= v <= 100.0 for v in q)
                assert all(q[k] <= q[k + 1] for k in range(4))
        except Exception:
            try:
                case = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                case = {"trophies": [], "history": []}
            forecast = safe_forecast(case)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        if len(text.encode("utf-8")) > 65536:  # pragma: no cover - 55 trophies never reach this
            forecast = safe_forecast(case)
            text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
