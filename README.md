# FIFA World Cup 2026 Predictor

A live experiment: **one Poisson model per national team, predicting every round of
the 2026 World Cup while the tournament is being played.** Predictions are committed
before each round and scored against reality after it — no retroactive edits.

**Site:** https://migue-rc.github.io/fwc2026 (published with Quarto)

## The idea

Comparing two Elo ratings always picks the stronger team, says nothing about the
score, and never sees an upset coming. Instead of one global model, this project
trains a **separate Poisson regression for every team** on that team's full match
history: goals scored as a function of opponent Elo, match location (home advantage
matters when the hosts are US/Mexico/Canada), match type, and recency (exponential
time decay, two-year half-life). To predict a match, each team's model answers "how
many goals would you score against this opponent, here?" — the two answers give a
scoreline.

Two workflows run side by side on identical fixtures:

- **Naive** — team-level history only.
- **Ensemble** — the same predictions scaled by a player layer: average Elo of the
  four best attackers vs the opponent's defensive block (four defenders, best
  midfielder, best goalkeeper), damped as `(atk/def)^2.5`. The exponent was tuned
  by sweeping it over the completed rounds (see `compare_results.ipynb`): winner
  accuracy plateaus at 59.1% for exponents in [2.3, 4.5], vs 50% without the layer.

## Repository layout

| Path | Purpose |
| --- | --- |
| `index.ipynb` | The story: methodology, live scoreboard, knockout bracket |
| `rounds_naive.ipynb` / `rounds_ensemble.ipynb` | Predictions and per-round evaluation for each workflow |
| `compare_results.ipynb` | Head-to-head: both workflows vs reality, plus the draw-rule analysis |
| `EDA_elo_ratings_wc2026.ipynb` | EDA of the Elo snapshot dataset (the dead end that shaped the design) |
| `EDA_elo_scrapper.ipynb` | EDA of the scraped match histories + model prototype |
| `fixtures.py` | Tournament calendar — the only file edited by hand between rounds |
| `model_utils.py` | Per-team Poisson training and match prediction |
| `report_utils.py` | Prediction storage, scoring, scoreboard, bracket and charts |
| `tsv_utils.py` / `validators.py` / `elo_utils.py` | Data loading, country-name normalization, player ratings |
| `elo_scraper.js` | Downloads each team's match history TSV from eloratings.net |
| `data/` | Scraped match histories (`countries/`), player ratings (`player/`), Elo snapshots |
| `predictions/` | One CSV per round per workflow (`naive_*.csv`, `ensemble_*.csv`) |

## Updating the tracker after each round

```sh
node elo_scraper.js       # refresh match histories (adds the played round)
# edit fixtures.py         # add the next round's matchups once drawn
make execute              # re-run all notebooks (predict next round, score last one)
make render               # rebuild the site
```

Each round's model trains only on matches played **before** that round started, so
every prediction is reproducible after the fact.

## Requirements

Python ≥ 3.14 managed with [uv](https://docs.astral.sh/uv/) (`uv sync`), Node for the
scraper, and [Quarto](https://quarto.org) for the site.
