from validators import build_country_mapper, country_to_country_code
import numpy as np
import pandas as pd

MATCH_TYPE_FRIENDLY = "friendly"
MATCH_TYPE_OFFICIAL = "official"
MATCH_TYPE_WORLD_CUP = "world_cup"

FRIENDLY_MATCH_CODES = {"F", "FT", "JAM"}
WORLD_CUP_MATCH_CODES = {"WC"}

GROUP_A = [('Mexico', 'MX'), ('South_Africa', 'ZA'), ('South_Korea', 'KR'), ('Czechia', 'CZ')]
GROUP_B = [('Canada', 'CA'), ('Bosnia_and_Herzegovina', 'BA'), ('Qatar', 'QA'), ('Switzerland', 'CH')]
GROUP_C = [('Brazil', 'BR'), ('Morocco', 'MA'), ('Haiti', 'HT'), ('Scotland', 'SQ')]
GROUP_D = [('United_States', 'US'), ('Paraguay', 'PY'), ('Australia', 'AU'), ('Turkey', 'TR')]
GROUP_E = [('Germany', 'DE'), ('Curacao', 'CW'), ('Ivory_Coast', 'CI'), ('Ecuador', 'EC')]
GROUP_F = [('Netherlands', 'NL'), ('Japan', 'JP'), ('Sweden', 'SE'), ('Tunisia', 'TN')]
GROUP_G = [('Belgium', 'BE'), ('Egypt', 'EG'), ('Iran', 'IR'), ('New_Zealand', 'NZ')]
GROUP_H = [('Spain', 'ES'), ('Cape_Verde', 'CV'), ('Saudi_Arabia', 'SA'), ('Uruguay', 'UY')]
GROUP_I = [('France', 'FR'), ('Senegal', 'SN'), ('Iraq', 'IQ'), ('Norway', 'NO')]
GROUP_J = [('Argentina', 'AR'), ('Algeria', 'DZ'), ('Austria', 'AT'), ('Jordan', 'JO')]
GROUP_K = [('Portugal', 'PT'), ('DR_Congo', 'CD'), ('Uzbekistan', 'UZ'), ('Colombia', 'CO')]
GROUP_L = [('England', 'EN'), ('Croatia', 'HR'), ('Ghana', 'GH'), ('Panama', 'PA')]


def check_country_code_compatibility():
    all_codes_from_groups = [country_code for group in
                             [GROUP_A, GROUP_B, GROUP_C, GROUP_D, GROUP_E, GROUP_F, GROUP_G, GROUP_H, GROUP_I, GROUP_J,
                              GROUP_K, GROUP_L] for _, country_code in group]
    all_codes_from_builder, _ = build_country_mapper()
    assert len(all_codes_from_groups) == len(all_codes_from_builder)
    for code in all_codes_from_builder.values():
        assert code in all_codes_from_groups
    return True


assert check_country_code_compatibility()


def normalize_match_type(match_type: str) -> str:
    code = str(match_type).strip().upper()

    if code in {
        MATCH_TYPE_FRIENDLY.upper(),
        MATCH_TYPE_OFFICIAL.upper(),
        MATCH_TYPE_WORLD_CUP.upper(),
    }:
        return code.lower()

    if code in FRIENDLY_MATCH_CODES:
        return MATCH_TYPE_FRIENDLY

    if code in WORLD_CUP_MATCH_CODES:
        return MATCH_TYPE_WORLD_CUP

    return MATCH_TYPE_OFFICIAL


def is_valid_country_code(code: str) -> bool:
    all_codes = [country_code for group in
                 [GROUP_A, GROUP_B, GROUP_C, GROUP_D, GROUP_E, GROUP_F, GROUP_G, GROUP_H, GROUP_I, GROUP_J, GROUP_K,
                  GROUP_L] for _, country_code in group]
    return code in all_codes


def file_name_from_country_code(country_code: str) -> str:
    all_groups = [*GROUP_A, *GROUP_B, *GROUP_C, *GROUP_D, *GROUP_E, *GROUP_F, *GROUP_G, *GROUP_H, *GROUP_I, *GROUP_J,
                  *GROUP_K, *GROUP_L]
    for country, code in all_groups:
        if country_code == code:
            return country
    raise ValueError(f"Country code {country_code} not found in the country codes.")


from typing import Optional, Union


def tsv_loader(country: str,
               start_date: Optional[Union[str, pd.Timestamp]] = None,
               end_date: Optional[Union[str, pd.Timestamp]] = None
               ) -> pd.DataFrame:
    # match_type is normalized in tsv_converter:
    # F, FT, JAM -> friendly; WC -> world_cup; everything else -> official.
    # to know the previous elo rating, we should apply the elo_delta to the resulting elo of both countries
    col_names = ['year', 'month', 'day', 'local_team_code', 'visit_team_code', 'local_goals', 'visit_goals',
                 'match_type', 'match_location_code', 'local_elo_delta', 'local_elo_total', 'visit_elo_total',
                 'local_rank_delta', 'visit_rank_delta', 'local_rank', 'visit_rank']
    if len(country.strip()) == 2 and is_valid_country_code(country.upper()):
        country_name = file_name_from_country_code(country.upper())
    else:
        # we should normalize the country name
        country_code = country_to_country_code(country)
        country_name = file_name_from_country_code(country_code)

    df = pd.read_csv(f"data/countries/{country_name}.tsv", sep="\t", names=col_names)

    # reeplace the year,month,day for datetime
    def replace_for_1(value: int) -> int:
        if value < 1:
            return 1
        return value

    df['month'] = df['month'].apply(replace_for_1)
    df['day'] = df['day'].apply(replace_for_1)
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']], errors='coerce')
    df.set_index('date', inplace=True)
    if start_date is not None:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date is not None:
        df = df[df.index <= pd.to_datetime(end_date)]
    df.drop(['year', 'month', 'day'], axis=1, inplace=True)
    df['local_elo_prev'] = df['local_elo_total'] - df['local_elo_delta']
    df['visit_elo_prev'] = df['visit_elo_total'] + df['local_elo_delta']
    return df


def tsv_converter(df: pd.DataFrame, team_code: str,
                  ) -> pd.DataFrame:
    # current_team_elo, current_opp_elo, match_type, location, goals_converted
    local_df = df.copy()
    local_df['target_code'] = team_code
    local_df['is_local'] = np.where(local_df['local_team_code'] == team_code, True, False)
    local_df['current_team_elo'] = np.where(local_df['is_local'],
                                            local_df['local_elo_prev'],
                                            local_df['visit_elo_prev'])

    local_df['current_opp_elo'] = np.where(local_df['is_local'],
                                           local_df['visit_elo_prev'],
                                           local_df['local_elo_prev'])

    local_df['goals_converted'] = np.where(local_df['is_local'],
                                           local_df['local_goals'],
                                           local_df['visit_goals'])

    local_df['location'] = np.where(local_df['match_location_code'].isna(),
                                    team_code,
                                    local_df['match_location_code'])
    local_df['match_type'] = local_df['match_type'].apply(normalize_match_type)
    return local_df[['current_team_elo', 'current_opp_elo', 'match_type', 'location', 'goals_converted']]


def load_country_data(country: str,
                      start_date: Optional[Union[str, pd.Timestamp]] = None,
                      end_date: Optional[Union[str, pd.Timestamp]] = None
                      ) -> pd.DataFrame:
    df = tsv_loader(country=country,
                    start_date=start_date,
                    end_date=end_date)
    country_code = country_to_country_code(country)
    return tsv_converter(df, country_code)
