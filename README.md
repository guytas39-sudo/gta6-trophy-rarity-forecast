# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/guytas39-sudo/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/guytas39-sudo/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A forecaster for GTA VI's first-week trophy earn rates on Exophase. For each of the 55 PS5 trophies it predicts five quantiles (0.05 / 0.25 / 0.5 / 0.75 / 0.95) of the earn rate at the target instant, scored by pinball loss on the logit scale.

How it works:

- **Snapshot collapse** — Exophase only changes on refresh, so consecutive identical reads collapse into one snapshot; a long freeze can't outweigh real movement.
- **Per-trophy extrapolation** — a weighted least-squares line of logit(rate) against ln(1 + hours since release), weighted by recency (18h half-life), player-base size and earner counts; the slope is shrunk toward the pooled cross-trophy slope, clamped, and damped over the extrapolation leg for saturation.
- **Structural caps and orderings** — medians are capped by trophy class (platinum 4%, 100%-completion 6%, story gold 65%, silver 45%), chapter medians are forced non-increasing, completion sits below the finale and the big grinds, and the platinum sits below everything.
- **Calibrated spread** — sigma combines pooled snapshot-to-snapshot logit diffusion, the fit's own extrapolation variance and a low-earner width term, floored and capped.
- **Special cases** — a trophy at ≥99.99% stays pinned; a trophy with no earners yet gets a small class prior; a fully frozen history (a refresh is owed and the tracked base will jump) gets a dilution shift and a wider spread.

The model was validated on a synthetic simulator of the final case (seeded player growth, ~3h refreshes, per-trophy saturation curves, scored with the exact pinball-on-logit rule), where it beat both the persistence and curve house baselines across base, slow-saturation, hourly-refresh, ~7h-refresh, small-base and huge-base worlds.

## Run it

```sh
python3 main.py
```

The program reads every `*.in` case file from `$MAKEFUN_INPUT_DIR` (else `/input/`) and writes one forecast file per case to `$MAKEFUN_OUTPUT_DIR` (else `/output/`). Python 3 standard library only, no network.

Entered in [make.fun's GTA VI trophy rarity forecast bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01).

---

<sub>Entered on [make.fun](https://make.fun/?ref=g2zdh0s8&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
