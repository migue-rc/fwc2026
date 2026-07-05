import re
from typing import Union

import pandas as pd
import pytest


def build_country_mapper() -> tuple[dict[str, str], dict[str, str]]:
    """
    Builds a reusable mapping system:
    - canonical country → code
    - aliases → canonical country
    """

    # -----------------------------
    # 1. Canonical mapping (source of truth)
    # -----------------------------
    country_codes = {
        "mexico": "MX",
        "south_africa": "ZA",
        "south_korea": "KR",
        "czechia": "CZ",
        "canada": "CA",
        "bosnia_and_herzegovina": "BA",
        "qatar": "QA",
        "switzerland": "CH",
        "brazil": "BR",
        "morocco": "MA",
        "haiti": "HT",
        "scotland": "SQ",
        "united_states": "US",
        "paraguay": "PY",
        "australia": "AU",
        "turkey": "TR",
        "germany": "DE",
        "curaçao": "CW",
        "ivory_coast": "CI",
        "ecuador": "EC",
        "netherlands": "NL",
        "japan": "JP",
        "sweden": "SE",
        "tunisia": "TN",
        "belgium": "BE",
        "egypt": "EG",
        "iran": "IR",
        "new_zealand": "NZ",
        "spain": "ES",
        "cape_verde": "CV",
        "saudi_arabia": "SA",
        "uruguay": "UY",
        "france": "FR",
        "senegal": "SN",
        "iraq": "IQ",
        "norway": "NO",
        "argentina": "AR",
        "algeria": "DZ",
        "austria": "AT",
        "jordan": "JO",
        "portugal": "PT",
        "dr_congo": "CD",
        "uzbekistan": "UZ",
        "colombia": "CO",
        "england": "EN",
        "croatia": "HR",
        "ghana": "GH",
        "panama": "PA",
    }

    # -----------------------------
    # 2. Aliases (messy real-world inputs)
    # -----------------------------
    aliases = {
        "south korea": "south_korea",
        "korea republic": "south_korea",
        "dr congo": "dr_congo",
        "congo dr": "dr_congo",
        "cote d'ivoire": "ivory_coast",
        "côte d'ivoire": "ivory_coast",
        "ivory coast": "ivory_coast",
        "bosnia and herzegovina": "bosnia_and_herzegovina",
        "cape verde": "cape_verde",
        "cabo verde": "cape_verde",
        "czech republic": "czechia",
        "south africa": "south_africa",
        "new zealand": "new_zealand",
        "saudi arabia": "saudi_arabia",
        "united states": "united_states",
        "usa": "united_states",
        "czechia": "czechia",
        "türkiye": "turkey",
        "ir iran": "iran",
        "curacao": "curaçao",
    }

    return country_codes, aliases


def normalize_country_name(name: str) -> str:
    """
    Normalizes raw input to improve matching stability.
    """
    name = str(name).strip()

    # unify separators
    name = name.replace("-", " ")
    name = name.replace("_", " ")

    # remove repeated spaces
    name = re.sub(r"\s+", " ", name)

    return name.lower()


def add_country_code(
    df: pd.DataFrame,
    target_column: str,
    output_column: str = "country_code",
    replace_none: Union[str, None] = None,
) -> pd.DataFrame:
    """
    Extensible country-to-code mapper.

    Pipeline:
    raw input → normalize → alias lookup → canonical lookup → code
    """

    # country_codes, aliases = build_country_mapper()

    result = df.copy()

    def resolve(country: str) -> Union[str, None]:
        if pd.isna(country):
            # print(f"Country {country} not found in the country codes.")
            return replace_none
        try:
            country_code = country_to_country_code(country)
            return country_code
        except ValueError:
            # print(f"Country {country} not found in the country codes.")
            return replace_none

    result[output_column] = result[target_column].apply(resolve)

    return result


def country_to_country_code(country: str) -> str:
    """
    :param country: The literal name of a country.
    :return: The code for the country, or None if not found.
    """
    if len(country.strip()) == 0:
        raise ValueError(
            f"Country {country} is empty. Please provide a valid country name."
        )
    if len(country.strip()) == 2:
        return country.upper()
    country_normalized = normalize_country_name(country.lower())
    country_codes, aliases = build_country_mapper()
    # we check first on the aliases, then on the canonical names
    alias = aliases.get(country_normalized, None)
    if alias is not None:
        country_code = country_codes.get(alias, None)
        if isinstance(country_code, str):
            return country_code
    country_code = country_codes.get(country_normalized, None)

    if isinstance(country_code, str):
        return country_code

    raise ValueError(f"Country {country} not found in the country codes.")


assert country_to_country_code("South Korea") == "KR"
assert country_to_country_code("USA") == "US"
with pytest.raises(ValueError) as exc_info:
    country_to_country_code("Unknown Country")
    assert "Country Unknown Country not found in the country codes." in str(
        exc_info.value
    )
    assert isinstance(exc_info.value, ValueError)


with pytest.raises(ValueError) as exc_info:
    country_to_country_code("USAA")
    assert "Country USAA not found in the country codes." in str(exc_info.value)
    assert isinstance(exc_info.value, ValueError)
