from validators import add_country_code, country_to_country_code


def get_elo_atk_def(team: str) -> tuple[float, float]:
    """
    return the average elo of the attackers and defenders of a team
    """
    import pandas as pd

    data_folder = "data/player"
    player_elo = pd.read_csv(data_folder + "/players.csv")
    # adding the country code column to the dataframe
    player_elo = add_country_code(
        df=player_elo, target_column="nationality", output_column="country_code"
    )
    country_code = country_to_country_code(team)
    # we should ponderate 4 attackers/forwars and 3 defenders/ and the best goalkeeper/ best midfielder
    team_df = player_elo.loc[player_elo["country_code"] == country_code].sort_values(
        "elo", ascending=False
    )
    attackers = team_df.loc[team_df["position"].isin(["Attacker", "Forward"])].head(4)
    # print(attackers.head())
    defenders = team_df.loc[team_df["position"].isin(["Defender"])].head(4)
    # print(defenders.head())
    midfielder = team_df.loc[team_df["position"] == "Midfielder"].head(1)
    # print(midfielder.head())
    goalkeeper = team_df.loc[team_df["position"] == "Goalkeeper"].head(1)
    # print(goalkeeper.head())
    # columns= ['player_name', 'elo', 'position']
    return attackers["elo"].mean(), pd.concat([defenders, midfielder, goalkeeper])[
        "elo"
    ].mean()
