# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/innalillahi/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/innalillahi/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for every GTA VI PlayStation 5 trophy, the percent of Exophase-tracked players who will have earned it a week after launch — five quantiles per trophy, scored by pinball loss on the logit scale.

## How it works

`main.py` (Python 3.13, standard library only, no network) reads each case and forecasts on the logit scale, blending two views of the history:

1. **Earned/players decomposition** — log earned-counts and log players-tracked are fitted separately against log(1 + hours since release), then rate = earned / players at the target. This captures dilution: when new players join faster than they earn trophies, rates fall even though nobody un-earns anything.
2. **Logit-trend fit** — a straight fit of logit(rate) against log(1 + hours since release), the same family as the house curve baseline.

The spread grows with the horizon (a random walk on the logit scale) plus the fit residual, so the 90% band stays honest. A launch-night flat history (no refresh yet) gets a dilution prior: the first Exophase refresh floods in new players, so the median is shifted down and the band widened.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` file in the input directory and writes one `*.out` forecast per case. Output is `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}`, one row per trophy, quantiles in percent, non-decreasing. Deterministic: two runs produce byte-identical output, well inside the 8-second per-case limit.

---

<sub>Entered on [make.fun](https://make.fun/?ref=wcgbp5n6&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
