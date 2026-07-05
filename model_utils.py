import math
from enum import Enum
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

from elo_utils import get_elo_atk_def
from tsv_utils import (
    GROUP_A,
    GROUP_B,
    GROUP_C,
    GROUP_D,
    GROUP_E,
    GROUP_F,
    GROUP_G,
    GROUP_H,
    GROUP_I,
    GROUP_J,
    GROUP_K,
    GROUP_L,
    MATCH_TYPE_OFFICIAL,
    MATCH_TYPE_WORLD_CUP,
    load_country_data, normalize_match_type,
    tsv_converter,
    tsv_loader,
)

Workflow = Enum("Workflow", [("NAIVE", 1), ("ATK_DEF", 2), ("HIST", 3)])
# INFO: every value should be a tuple with the name of the file and the country code.

# Exponent of the player-layer adjustment (atk/def)^reg in the ensemble
# workflow. (atk/def)^reg is 1 for balanced matchups, boosts a superior attack
# and discounts one facing a stronger block. Tuned on the 88 matches played
# through the Round of 32 (see the sweep in compare_results.ipynb): winner
# accuracy plateaus at 59.1% for reg in [2.3, 4.5] vs 51.1% at the original
# 0.3, so we take the middle of the plateau rather than the in-sample peak.
PLAYER_LAYER_REG = 2.5

HOST_LOCATIONS = {"MX", "US", "CA"}
MIN_CATEGORY_COUNT = 10
CATEGORY_FALLBACK = "other"
CATEGORICAL_FEATURES = ("match_type", "location")


def collapse_sparse_categories(
    team_df: pd.DataFrame,
) -> tuple[dict[str, set[str]], dict[str, str]]:
    category_levels = {}
    category_fallbacks = {}

    for col in CATEGORICAL_FEATURES:
        if col not in team_df.columns:
            continue

        values = team_df[col].astype("string").fillna(CATEGORY_FALLBACK)
        counts = values.value_counts()

        if col == "match_type":
            common_values = set(counts.index)
        else:
            common_values = set(counts[counts >= MIN_CATEGORY_COUNT].index)

        if col == "location":
            common_values |= HOST_LOCATIONS.intersection(set(counts.index))

        team_df[col] = values.where(values.isin(common_values), CATEGORY_FALLBACK)
        team_df[col] = team_df[col].astype(str)

        levels = set(team_df[col].unique())
        category_levels[col] = levels

        if col == "match_type" and MATCH_TYPE_OFFICIAL in levels:
            category_fallbacks[col] = MATCH_TYPE_OFFICIAL
        elif CATEGORY_FALLBACK in levels:
            category_fallbacks[col] = CATEGORY_FALLBACK
        else:
            category_fallbacks[col] = str(team_df[col].mode(dropna=True).iloc[0])

    return category_levels, category_fallbacks


def prepare_prediction_categories(model, prediction_df: pd.DataFrame) -> pd.DataFrame:
    prediction_df = prediction_df.copy()
    category_levels = getattr(model, "_category_levels", {})
    category_fallbacks = getattr(model, "_category_fallbacks", {})

    for col, levels in category_levels.items():
        if col not in prediction_df.columns:
            continue

        fallback = category_fallbacks[col]
        values = prediction_df[col].astype("string").fillna(fallback)
        prediction_df[col] = values.where(values.isin(levels), fallback).astype(str)

    return prediction_df


def load_group(group: list, date: Union[str, None] = None) -> list[pd.DataFrame]:
    dfs: list[pd.DataFrame] = []
    for country, code in group:
        df = tsv_loader(country)
        if date is not None:
            df = df[df["date"] <= pd.to_datetime(date)]
        dfs.append(tsv_converter(df, code))
    return dfs


def plot_group(group):
    groups = load_group(group)
    plot_data = None
    wrc_2016_first_match_dt = pd.to_datetime("2016-01-01")
    for index, df in enumerate(groups):
        df["team"] = group[index][0]
        df = df.reset_index()
        df = df[df["date"] > wrc_2016_first_match_dt]
        plot_data = pd.concat([plot_data, df]) if plot_data is not None else df

    if plot_data is None:
        raise ValueError("No data found")
    sorted_dates = plot_data.sort_values(by=["team", "date"], ascending=False)
    avg_elo = (
        sorted_dates.groupby("team", group_keys=False)
        .head(1)
        # .agg({'current_team_elo': 'mean'})
        .groupby("team")["current_team_elo"]
        .mean()
    )
    three_best_elo = avg_elo.sort_values(ascending=False).head(3)
    ax = sns.lineplot(data=plot_data, x="date", y="current_team_elo", hue="team")
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    ax.set_ylabel("Elo Rating")

    ax.set_xlabel(
        f"avg elo:{math.floor(avg_elo.mean())} - top 3 avg:{math.floor(three_best_elo.mean())}"
    )
    plt.show()


def add_time_decay_weights(
    df: pd.DataFrame, date_col: str = "date", half_life_days: float = 365 * 2
):
    """
    Add exponential time-decay weights to a dataframe.

    More recent matches have higher weights.

    The max date is not today, its the max date of the data.

    Parameters
    ----------
    df : DataFrame
        Match data with a date column.

    date_col : str
        Column containing match dates.

    half_life_days : float
        How fast old matches lose importance.
        Example:
        - 365*2 → every 2 years weight halves
        - smaller → faster adaptation
    """

    df = df.copy()

    # Ensure datetime format
    df[date_col] = pd.to_datetime(df[date_col])

    # Use most recent match as reference point
    max_date = df[date_col].max()

    # Compute time difference in days
    df["days_since"] = (max_date - df[date_col]).dt.days

    # Exponential decay:
    # weight = exp(-lambda * t)
    # lambda = ln(2) / half_life
    decay_rate: float = np.log(2) / half_life_days
    # print(decay_rate)
    df["time_weight"] = np.exp(-decay_rate * df["days_since"])
    return df


def train_poisson_model(team_df: pd.DataFrame):

    # 1. Apply time weights
    team_df_copy = team_df.copy()
    team_df_copy = add_time_decay_weights(team_df_copy, date_col="date")

    time_weight = "time_weight"
    dep_var = "goals_converted"

    # 2. Get features
    feature_cols = team_df_copy.drop(
        [dep_var, time_weight, "date"], axis=1, errors="ignore"
    ).columns.tolist()

    # 3. Clean invalid rows and cap minimum weight
    numeric_cols = [
        dep_var,
        time_weight,
        "days_since",
        "current_team_elo",
        "current_opp_elo",
    ]
    for col in numeric_cols:
        if col in team_df_copy.columns:
            team_df_copy[col] = pd.to_numeric(team_df_copy[col], errors="coerce")

    team_df_copy = team_df_copy.replace([np.inf, -np.inf], np.nan)
    team_df_copy = team_df_copy.dropna(subset=feature_cols + [dep_var, time_weight])
    team_df_copy[time_weight] = team_df_copy[time_weight].clip(lower=0.001)

    if team_df_copy.empty:
        raise ValueError("Cannot train Poisson model: no valid rows after cleaning.")

    category_levels, category_fallbacks = collapse_sparse_categories(team_df_copy)

    # 4. Scale large numbers safely inside the formula
    for col in ["days_since", "current_team_elo", "current_opp_elo"]:
        if col in feature_cols:
            feature_cols.remove(col)
            if col == "days_since":
                feature_cols.append(f"I({col} / 365.25)")
            else:
                # Scales ELO from 1500 -> 1.5, stopping the math explosion
                feature_cols.append(f"I({col} / 1000)")

    ind_vars = " + ".join(feature_cols)
    formula = f"{dep_var} ~ {ind_vars}"

    # 5. Fit model (No global categorical means no empty 0-count categories!)
    model = smf.glm(
        family=sm.families.Poisson(),
        formula=formula,
        data=team_df_copy,
        freq_weights=team_df_copy[time_weight],
    ).fit()

    model._category_levels = category_levels
    model._category_fallbacks = category_fallbacks

    return model

def predict_match(
    team_1: pd.DataFrame,
    team_2: pd.DataFrame,
    location: str,
    match_type: str = MATCH_TYPE_WORLD_CUP,
    workflow: Workflow = Workflow.NAIVE,
    team_1_name: Union[str, None] = None,
    team_2_name: Union[str, None] = None,
):
    """
    With two dataframes, predicts the result of the match.
    This includes training the model.
    :param team_1: pandas dataframe
    :param team_2: pandas dataframe
    :param location: location of the match
    :param match_type: match type code or category for the prediction
    :param workflow: Workflow enum
    :param team_1_name: name of team 1
    :param team_2_name: name of team 2
    :return: tuple (goals_1, goals_2)
    """
    team_1_copy = team_1.copy()
    team_1_copy = team_1_copy.reset_index()
    team_2_copy = team_2.copy()
    team_2_copy = team_2_copy.reset_index()
    dep_var = "goals_converted"
    model_1 = train_poisson_model(team_1_copy)
    model_2 = train_poisson_model(team_2_copy)
    current_2 = (
        team_2_copy.sort_values(by="date", ascending=False).head(1).reset_index()
    )
    current_2["days_since"] = 0
    current_2["location"] = location
    current_2["match_type"] = normalize_match_type(match_type)
    current_1 = (
        team_1_copy.sort_values(by="date", ascending=False).head(1).reset_index()
    )
    current_1["days_since"] = 0
    current_1["location"] = location
    current_1["match_type"] = normalize_match_type(match_type)

    current_2 = prepare_prediction_categories(
        model_1,
        current_2.drop([dep_var, "date"], axis=1, errors="ignore"),
    )
    current_1 = prepare_prediction_categories(
        model_2,
        current_1.drop([dep_var, "date"], axis=1, errors="ignore"),
    )

    goals_converted_1_to_2 = model_1.predict(current_2)
    goals_converted_2_to_1 = model_2.predict(current_1)

    if np.isnan(goals_converted_1_to_2.iloc[0]) or np.isnan(goals_converted_2_to_1.iloc[0]):
        raise ValueError(
            f"Model failed to predict goals: {goals_converted_1_to_2} {goals_converted_2_to_1}"
        )

    if workflow == Workflow.ATK_DEF:
        if not (isinstance(team_1_name, str) and isinstance(team_2_name, str)):
            raise ValueError(
                "ATK_DEF workflow requires team_1_name and team_2_name to look up player ratings."
            )
        team_1_atk, team_1_def = get_elo_atk_def(team_1_name)
        team_2_atk, team_2_def = get_elo_atk_def(team_2_name)
        return (
            goals_converted_1_to_2.iloc[0] * (team_1_atk / team_2_def) ** PLAYER_LAYER_REG,
            goals_converted_2_to_1.iloc[0] * (team_2_atk / team_1_def) ** PLAYER_LAYER_REG,
        )

    return goals_converted_1_to_2.iloc[0], goals_converted_2_to_1.iloc[0]


def predict_round(round: list[tuple[str, str, str]], end_date: pd.Timestamp, filename: str | None = None, workflow: Workflow = Workflow.ATK_DEF):
    results_df = pd.DataFrame(columns=["team_1", "team_2", "goals_1", "goals_2", "winner", "location"])
    for team_1, team_2, location in round:
        team_1_df = load_country_data(country=team_1, end_date=end_date)
        team_2_df = load_country_data(country=team_2, end_date=end_date)
        goals_1, goals_2 = predict_match(team_1=team_1_df,
                                         team_2=team_2_df,
                                         workflow=workflow,
                                         location=location,
                                         team_1_name=team_1,
                                         team_2_name=team_2,
                                         )
        winner = team_1 if goals_1 > goals_2 else team_2 if goals_2 > goals_1 else "Draw"
        new_row = pd.DataFrame(
            {"team_1"  : team_1, "team_2": team_2, "goals_1": goals_1, "goals_2": goals_2, "winner": winner,
             "location": location}, index=[0])
        results_df = pd.concat([results_df, new_row], ignore_index=True)

    if filename is not None:
        results_df.to_csv(filename)
    return results_df
