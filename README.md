![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)
![Rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/dhamskie-ai/gta6-trophy-rarity-forecast-01/rank.svg)
![Reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/dhamskie-ai/gta6-trophy-rarity-forecast-01/reward.svg)

# GTA VI Trophy Rarity Forecast — week one

Forecasts every Grand Theft Auto VI PS5 trophy's Exophase earn rate at
2026-11-26T00:00Z, as five quantiles (0.05 / 0.25 / 0.5 / 0.75 / 0.95)
per trophy. Built for the [make.fun competition](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
**gta6-trophy-rarity-forecast-01** ($250, free entry).

## The model

Earn rate = earned / tracked, and the model works in **count space**,
not on the rate itself. For each trophy it fits least-squares lines of
log(earned) and log(tracked) against ln(1 + hours since release) over
the case's own history, shrinks each slope toward zero when the history
is short or flat, extrapolates both counts to the target, and divides:
rate = E(target) / T(target). When tracked players grow faster than
earners, the rate dilutes even while earned counts rise — a persistence
forecast and a logit-linear fit on the rate itself both miss that, and
launch night proved it happens: one Exophase refresh took tracked
players 638 → 3,868 and cut every rate but the 100% one.

When the history is flat (launch night: every read identical), the
trend fit is exactly zero, so the model falls back to a dilution-aware
persistence median — modestly below the last rate (less for common
early-story trophies, more for rare ones) — with a wide spread skewed
downward to allow for a refresh/dilution shock. The size of one past
refresh is never hardcoded as a prediction.

The spread is a random-walk term growing with √horizon plus a dilution
term that grows with horizon and with how fast tracked players were
growing in the case's history, skewed downward. The 100% trophy stays
100%; the platinum is capped low (nobody plats in a week). All
quantiles are clamped to [0.01, 99.99] on the logit scale and [0, 100]
in percent, and forced non-decreasing.

## Validation

- **Format:** runs on practice cases p00/p01 in ~0.07 s for both cases
  (limit 8 s each), outputs valid (55 trophies, 5 non-decreasing
  quantiles each, ≤ 65,536 bytes), byte-identical across re-runs.
- **Proxy backtest** (true practice answers are not in the kit, so the
  known refresh facts are used as assumed answers — 638 → 3,868
  players): mean pinball loss on the logit scale —
  pure dilution: **mine 0.301 vs house baseline 1.688**;
  partial dilution (earned +30%): **mine 0.252 vs baseline 1.430**.
  These are proxy numbers under stated assumptions, not exact practice
  losses.
- **Synthetic trend test** (fake case with tracked players growing
  638 → ~20,000 over 48 h and earned counts growing per trophy):
  **mine 0.340 vs house baseline 0.430**, exercising the trend branch
  the final case will rely on.
- **Robustness:** malformed cases fall back to a safe persistence
  forecast instead of crashing, so an output file is always written.

## How to run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no network. Reads every `*.in`
case in `$MAKEFUN_INPUT_DIR` (else `/input`) and writes `<id>.out`
to `$MAKEFUN_OUTPUT_DIR` (else `/output`):
`{"trophies": [{"id": N, "q": [q05, q25, q50, q75, q95]}, ...]}`.

## License

MIT — see [LICENSE](LICENSE).
