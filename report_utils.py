"""Shared reporting layer for the live tracker.

Everything the notebooks need to predict, score and visualize a round lives
here, so the notebooks stay thin and the whole site can be refreshed after
each round with a scrape + re-run + re-render.

Conventions
-----------
- Predictions are written to ``predictions/{prefix}_{round_key}.csv`` where
  ``prefix`` identifies the workflow (``naive`` or ``ensemble``). The two
  workflows never share files, so they can always be compared.
- A round's model trains only on matches played strictly BEFORE the round
  starts, which makes every prediction reproducible after the fact.
- Real results are read from the scraped eloratings.net histories, so
  "updating reality" is just re-running ``node elo_scraper.js``.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

from fixtures import ROUND_LABELS, ROUNDS, UPCOMING_LABELS, UPCOMING_ROUNDS
from model_utils import Workflow, predict_round
from tsv_utils import load_country_data
from validators import country_to_country_code

PREDICTIONS_DIR = "predictions"
CHARTS_DIR = "charts"


# --- Predictions --------------------------------------------------------------

def prediction_path(prefix: str, round_key: str) -> str:
    return os.path.join(PREDICTIONS_DIR, f"{prefix}_{round_key}.csv")


def predict_all_rounds(workflow: Workflow, prefix: str) -> dict[str, pd.DataFrame]:
    """Predict every round in the calendar, one CSV per round.

    The training cutoff is the day before the round starts: the model never
    sees the matches it is asked to predict, even when re-run later.
    """
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    predictions = {}
    for round_key, (fixtures, start_date, _end_date) in ROUNDS.items():
        cutoff = start_date - pd.Timedelta(days=1)
        predictions[round_key] = predict_round(
            fixtures,
            cutoff,
            prediction_path(prefix, round_key),
            workflow=workflow,
        )
    return predictions


def load_predictions(prefix: str) -> dict[str, pd.DataFrame]:
    """Load previously generated predictions for one workflow."""
    predictions = {}
    for round_key in ROUNDS:
        path = prediction_path(prefix, round_key)
        if os.path.exists(path):
            predictions[round_key] = pd.read_csv(path, index_col=0)
    return predictions


# --- Real results ---------------------------------------------------------------

def real_results(round_key: str) -> pd.DataFrame:
    """Actual scores for a round, read from the scraped match histories.

    Note: for knockout matches this is the score after 90/120 minutes, so a
    match decided on penalties shows up as a draw here. ``advancing_teams``
    below resolves who actually went through.
    """
    fixtures, start_date, end_date = ROUNDS[round_key]
    rows = []
    for team_1, team_2, _location in fixtures:
        team_1_df = load_country_data(team_1, start_date, end_date)
        team_2_df = load_country_data(team_2, start_date, end_date)
        if team_1_df.empty or team_2_df.empty:
            continue  # match not played yet (or scrape not refreshed)
        goals_1 = team_1_df["goals_converted"].iloc[0]
        goals_2 = team_2_df["goals_converted"].iloc[0]
        winner = team_1 if goals_1 > goals_2 else team_2 if goals_2 > goals_1 else "Draw"
        rows.append([team_1, team_2, goals_1, goals_2, winner])
    return pd.DataFrame(rows, columns=["team_1", "team_2", "goals_1", "goals_2", "winner"])


def advancing_teams(round_key: str) -> set[str] | None:
    """Country codes that actually advanced from a knockout round.

    Derived from the NEXT round's fixture list, which makes it robust to
    penalty shoot-outs. Returns None when the next round has not been drawn.
    """
    keys = list(ROUNDS)
    idx = keys.index(round_key)
    if idx + 1 >= len(keys):
        return None
    next_fixtures = ROUNDS[keys[idx + 1]][0]
    return {
        country_to_country_code(team)
        for team_1, team_2, _loc in next_fixtures
        for team in (team_1, team_2)
    }


# --- Evaluation -----------------------------------------------------------------

def evaluate_results(results_df: pd.DataFrame, predictions_df: pd.DataFrame) -> tuple[float, float]:
    """Winner accuracy and mean absolute goal error of predictions vs results."""
    merged_df = results_df.merge(predictions_df, on=["team_1", "team_2"], how="inner")
    if merged_df.empty:
        return float("nan"), float("nan")
    merged_df["goal_err"] = (
        abs(merged_df["goals_1_x"] - merged_df["goals_1_y"])
        + abs(merged_df["goals_2_x"] - merged_df["goals_2_y"])
    ) / 2
    merged_df["correct_prediction"] = merged_df["winner_x"] == merged_df["winner_y"]
    return float(merged_df["correct_prediction"].mean()), float(merged_df["goal_err"].mean())


def scoreboard(prefixes: tuple[str, ...] = ("naive", "ensemble")) -> pd.DataFrame:
    """Per-round accuracy and goal MAE for each workflow, completed rounds only."""
    rows = []
    for round_key in ROUNDS:
        results_df = real_results(round_key)
        if results_df.empty:
            continue
        row = {"round": ROUND_LABELS[round_key], "matches": len(results_df)}
        for prefix in prefixes:
            path = prediction_path(prefix, round_key)
            if not os.path.exists(path):
                continue
            predictions_df = pd.read_csv(path, index_col=0)
            accuracy, goal_mae = evaluate_results(results_df, predictions_df)
            row[f"{prefix}_accuracy"] = round(accuracy, 3)
            row[f"{prefix}_goal_mae"] = round(goal_mae, 2)
        rows.append(row)
    return pd.DataFrame(rows)


def round_detail(prefix: str, round_key: str) -> pd.DataFrame:
    """Side-by-side view of one round: predicted score/winner vs the real ones."""
    path = prediction_path(prefix, round_key)
    if not os.path.exists(path):
        return pd.DataFrame()
    predictions_df = pd.read_csv(path, index_col=0)
    results_df = real_results(round_key)
    if results_df.empty:
        detail = predictions_df.copy()
        detail["predicted_score"] = (
            detail["goals_1"].round(2).astype(str) + " - " + detail["goals_2"].round(2).astype(str)
        )
        return detail[["team_1", "team_2", "predicted_score", "winner"]].rename(
            columns={"winner": "predicted_winner"}
        )
    merged = results_df.merge(predictions_df, on=["team_1", "team_2"],
                              how="inner", suffixes=("_real", "_pred"))
    merged["predicted_score"] = (
        merged["goals_1_pred"].round(2).astype(str) + " - " + merged["goals_2_pred"].round(2).astype(str)
    )
    merged["real_score"] = (
        merged["goals_1_real"].astype(int).astype(str) + " - " + merged["goals_2_real"].astype(int).astype(str)
    )
    merged["correct"] = merged["winner_real"] == merged["winner_pred"]
    return merged[["team_1", "team_2", "predicted_score", "real_score",
                   "winner_pred", "winner_real", "correct"]]


def sweep_player_layer(reg_values=None) -> pd.DataFrame:
    """Tune the player-layer damping exponent on the matches played so far.

    The ensemble prediction is the naive Poisson goals scaled by
    ``(atk/def) ** reg``, so the sweep reuses the stored naive predictions and
    only recomputes the scaling - no model retraining. ``reg = 0`` is exactly
    the naive workflow, which makes the curve a direct answer to "does the
    player layer help at all, and how hard should it push?".
    """
    import numpy as np

    from elo_utils import get_elo_atk_def

    if reg_values is None:
        reg_values = np.round(np.arange(0.0, 1.51, 0.05), 2)

    ratings_cache: dict[str, tuple[float, float]] = {}

    def ratings(team: str) -> tuple[float, float]:
        if team not in ratings_cache:
            ratings_cache[team] = get_elo_atk_def(team)
        return ratings_cache[team]

    matches = []
    for round_key in ROUNDS:
        results_df = real_results(round_key)
        path = prediction_path("naive", round_key)
        if results_df.empty or not os.path.exists(path):
            continue
        naive_df = pd.read_csv(path, index_col=0)
        merged = results_df.merge(naive_df, on=["team_1", "team_2"],
                                  how="inner", suffixes=("_real", "_pred"))
        for row in merged.itertuples():
            atk_1, def_1 = ratings(row.team_1)
            atk_2, def_2 = ratings(row.team_2)
            matches.append({
                "team_1": row.team_1, "team_2": row.team_2,
                "base_1": row.goals_1_pred, "base_2": row.goals_2_pred,
                "ratio_1": atk_1 / def_2, "ratio_2": atk_2 / def_1,
                "real_1": row.goals_1_real, "real_2": row.goals_2_real,
                "real_winner": row.winner_real,
            })

    rows = []
    for reg in reg_values:
        correct = 0
        goal_err = 0.0
        for m in matches:
            goals_1 = m["base_1"] * m["ratio_1"] ** reg
            goals_2 = m["base_2"] * m["ratio_2"] ** reg
            winner = (m["team_1"] if goals_1 > goals_2
                      else m["team_2"] if goals_2 > goals_1 else "Draw")
            correct += winner == m["real_winner"]
            goal_err += (abs(goals_1 - m["real_1"]) + abs(goals_2 - m["real_2"])) / 2
        rows.append({"reg": float(reg),
                     "accuracy": correct / len(matches),
                     "goal_mae": goal_err / len(matches)})
    return pd.DataFrame(rows)


# --- Bracket visual ---------------------------------------------------------------

SHORT_NAMES = {
    "Bosnia and Herzegovina": "Bosnia-Herz.",
    "C\u00f4te d'Ivoire": "C\u00f4te d'Iv.",
}

ACCENT = "#4878cf"   # model's pick
HIT_GREEN = "#2e9e5b"  # advanced
MISS_RED = "#c0504d"   # pick eliminated
NEUTRAL = "#888888"    # readable on both light and dark backgrounds


def _bracket_data(prefix: str) -> list[tuple[str, list[dict]]]:
    """Collect per-match display data for the knockout rounds."""
    columns = []
    for round_key in ("round_of_32", "round_of_16"):
        fixtures, _start, _end = ROUNDS[round_key]
        predictions = {}
        path = prediction_path(prefix, round_key)
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0)
            predictions = {(r.team_1, r.team_2): r.winner for r in df.itertuples()}
        results = {(r.team_1, r.team_2): (int(r.goals_1), int(r.goals_2))
                   for r in real_results(round_key).itertuples()}
        advanced_codes = advancing_teams(round_key)

        matches = []
        for team_1, team_2, _loc in fixtures:
            advanced = None
            if advanced_codes is not None:
                for team in (team_1, team_2):
                    if country_to_country_code(team) in advanced_codes:
                        advanced = team
            matches.append({
                "teams": (team_1, team_2),
                "predicted": predictions.get((team_1, team_2)),
                "advanced": advanced,
                "score": results.get((team_1, team_2)),
            })
        columns.append((ROUND_LABELS[round_key], matches))

    for upcoming_key in UPCOMING_ROUNDS:
        slots = {"quarter_finals": 4, "semi_finals": 2, "final": 1}[upcoming_key]
        columns.append((UPCOMING_LABELS[upcoming_key],
                        [{"teams": ("TBD", "TBD"), "predicted": None,
                          "advanced": None, "score": None}] * slots))
    return columns


def plot_bracket(prefix: str = "ensemble", filename: str = "charts/bracket.png"):
    """Knockout bracket, prediction vs reality, as a matplotlib figure.

    Blue bold = the model's pick; green shading = the team that actually
    advanced; red = a pick that was eliminated. Rounds not yet drawn appear
    as dashed TBD slots. Transparent background so it works on both themes.
    """
    from matplotlib.patches import FancyBboxPatch

    columns = _bracket_data(prefix)
    n_cols = len(columns)
    fig, ax = plt.subplots(figsize=(13, 11), dpi=150)
    ax.set_xlim(0, n_cols)
    ax.set_ylim(-0.05, 1)
    ax.axis("off")

    box_w = 0.86
    for col_idx, (title, matches) in enumerate(columns):
        n = len(matches)
        box_h = min(0.052, 0.9 / n * 0.82)
        ax.text(col_idx + box_w / 2, 0.985, title.upper(), ha="center", va="top",
                fontsize=9, fontweight="bold", color=NEUTRAL, alpha=.9)
        for row_idx, match in enumerate(matches):
            y_center = 0.93 - (row_idx + 0.5) * (0.93 / n)
            y0 = y_center - box_h / 2
            is_tbd = match["teams"][0] == "TBD"
            box = FancyBboxPatch(
                (col_idx + (1 - box_w) / 2, y0), box_w, box_h,
                boxstyle="round,pad=0.004",
                linewidth=0.9,
                edgecolor=NEUTRAL,
                facecolor="none",
                linestyle="--" if is_tbd else "-",
                alpha=.45 if is_tbd else .8,
            )
            ax.add_patch(box)

            for t_idx, team in enumerate(match["teams"]):
                ty = y_center + (box_h / 4 if t_idx == 0 else -box_h / 4)
                label = SHORT_NAMES.get(team, team)
                color, weight = NEUTRAL, "normal"
                if match["predicted"] == team:
                    weight = "bold"
                    if match["advanced"] is None:
                        color = ACCENT
                    elif match["advanced"] == team:
                        color = HIT_GREEN
                    else:
                        color = MISS_RED
                elif match["advanced"] == team:
                    color = HIT_GREEN
                if match["advanced"] == team:
                    ax.fill_between(
                        [col_idx + (1 - box_w) / 2 + 0.015, col_idx + (1 - box_w) / 2 + box_w - 0.015],
                        ty - box_h / 4.6, ty + box_h / 4.6,
                        color=HIT_GREEN, alpha=.14, linewidth=0)
                ax.text(col_idx + (1 - box_w) / 2 + 0.03, ty, label,
                        ha="left", va="center", fontsize=7.2,
                        color=color, fontweight=weight,
                        alpha=.5 if is_tbd else 1)
                if match["score"] is not None:
                    ax.text(col_idx + (1 - box_w) / 2 + box_w - 0.03, ty,
                            str(match["score"][t_idx]),
                            ha="right", va="center", fontsize=7.2, color=NEUTRAL)

    legend_y = -0.03
    ax.text(0.06, legend_y, "bold = model's pick (%s workflow)" % prefix,
            fontsize=8, color=ACCENT, fontweight="bold", ha="left")
    ax.text(1.75, legend_y, "green = advanced", fontsize=8, color=HIT_GREEN, ha="left")
    ax.text(2.85, legend_y, "red = pick eliminated", fontsize=8, color=MISS_RED, ha="left")
    ax.text(4.0, legend_y, "dashed = not drawn yet", fontsize=8, color=NEUTRAL, ha="left")

    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    os.makedirs(CHARTS_DIR, exist_ok=True)
    fig.savefig(filename, transparent=True, bbox_inches="tight")
    return fig


# --- ELO landscape ------------------------------------------------------------------

def plot_elo_landscape(top_n: int = 8, since: str = "2018-01-01",
                       filename: str = "charts/elo-landscape.png"):
    """ELO trajectories of the strongest teams in the tournament since 2018.

    Saved with a transparent background and mid-tone colors so it reads well
    on both the light and dark theme.
    """
    from fixtures import first_round

    teams = sorted({t for t1, t2, _ in first_round for t in (t1, t2)})
    latest = {}
    series = {}
    for team in teams:
        df = load_country_data(team).reset_index().sort_values("date")
        df = df[df["date"] >= pd.to_datetime(since)]
        if df.empty:
            continue
        series[team] = df[["date", "current_team_elo"]]
        latest[team] = df["current_team_elo"].iloc[-1]

    top_teams = sorted(latest, key=latest.get, reverse=True)[:top_n]

    palette = ["#4878cf", "#e07b39", "#2e9e5b", "#c0504d",
               "#8064a2", "#4bacc6", "#b5a02f", "#d16587"]
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    for color, team in zip(palette, top_teams):
        df = series[team]
        ax.plot(df["date"], df["current_team_elo"], color=color, linewidth=1.4, alpha=.85)

    # End-of-line labels, nudged apart so close ratings stay readable.
    ends = sorted(
        ((series[t]["current_team_elo"].iloc[-1], t, c)
         for t, c in zip(top_teams, palette[: len(top_teams)])),
        reverse=True,
    )
    min_gap = 14  # Elo units between label baselines
    label_y = []
    for value, _team, _color in ends:
        y = value
        if label_y and label_y[-1] - y < min_gap:
            y = label_y[-1] - min_gap
        label_y.append(y)
    last_date = max(s["date"].iloc[-1] for s in series.values())
    for (value, team, color), y in zip(ends, label_y):
        ax.annotate(team.replace("_", " "), xy=(last_date, y),
                    xytext=(8, 0), textcoords="offset points",
                    color=color, fontsize=9, fontweight="bold", va="center")

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(alpha=.2)
    ax.tick_params(colors="#888888", labelsize=9)
    ax.set_ylabel("Elo rating", color="#888888")
    ax.set_title(f"The favorites, by Elo (top {top_n} entering the tournament)",
                 color="#888888", fontsize=12)
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    ax.margins(x=.08)
    fig.tight_layout()
    os.makedirs(CHARTS_DIR, exist_ok=True)
    fig.savefig(filename, transparent=True, bbox_inches="tight")
    return fig
