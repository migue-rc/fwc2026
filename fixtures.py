"""Tournament fixtures and calendar for the FIFA World Cup 2026.

This is the single place to update as the tournament progresses:

1. Add the new round's fixtures as a list of ``(team_1, team_2, location)``
   tuples. Team names can be any alias known to ``validators.py``; the
   location is the ISO code of the host country (``US``, ``MX`` or ``CA``).
2. Add the round's start/end dates to ``ROUNDS`` so the evaluation code can
   slice the scraped match history to exactly that window.
3. Re-scrape the ratings (``node elo_scraper.js``), re-run the prediction
   notebooks and re-render the site.

Fixtures for rounds that have not been drawn yet simply don't exist here;
downstream code treats a missing round as "not played / not predicted".
"""

import pandas as pd

# --- Group stage -------------------------------------------------------------
# Matchday 1 (Jun 11-17)
first_round = [
    # Group A
    ("Mexico", "South Africa", "MX"),
    ("South Korea", "Czechia", "MX"),
    # Group B
    ("Canada", "Bosnia and Herzegovina", "CA"),
    ("Qatar", "Switzerland", "US"),
    # Group C
    ("Brazil", "Morocco", "US"),
    ("Haiti", "Scotland", "US"),
    # Group D
    ("USA", "Paraguay", "US"),
    ("Australia", "Türkiye", "CA"),
    # Group E
    ("Germany", "Curaçao", "US"),
    ("Côte d'Ivoire", "Ecuador", "US"),
    # Group F
    ("Netherlands", "Japan", "US"),
    ("Sweden", "Tunisia", "MX"),
    # Group G
    ("Belgium", "Egypt", "US"),
    ("IR Iran", "New Zealand", "US"),
    # Group H
    ("Spain", "Cabo Verde", "US"),
    ("Saudi Arabia", "Uruguay", "US"),
    # Group I
    ("France", "Senegal", "US"),
    ("Iraq", "Norway", "US"),
    # Group J
    ("Argentina", "Algeria", "US"),
    ("Austria", "Jordan", "US"),
    # Group K
    ("Portugal", "Congo DR", "US"),
    ("Uzbekistan", "Colombia", "MX"),
    # Group L
    ("Ghana", "Panama", "CA"),
    ("England", "Croatia", "US"),
]

# Matchday 2 (Jun 18-23)
second_round = [
    # Group A
    ("Czechia", "South Africa", "US"),
    ("Mexico", "South Korea", "MX"),
    # Group B
    ("Switzerland", "Bosnia and Herzegovina", "US"),
    ("Canada", "Qatar", "CA"),
    # Group C
    ("Brazil", "Haiti", "US"),
    ("Scotland", "Morocco", "US"),
    # Group D
    ("Türkiye", "Paraguay", "US"),
    ("USA", "Australia", "US"),
    # Group E
    ("Germany", "Côte d'Ivoire", "CA"),
    ("Ecuador", "Curaçao", "US"),
    # Group F
    ("Netherlands", "Sweden", "US"),
    ("Tunisia", "Japan", "MX"),
    # Group G
    ("Belgium", "IR Iran", "US"),
    ("New Zealand", "Egypt", "CA"),
    # Group H
    ("Spain", "Saudi Arabia", "US"),
    ("Uruguay", "Cabo Verde", "US"),
    # Group I
    ("France", "Iraq", "US"),
    ("Norway", "Senegal", "US"),
    # Group J
    ("Argentina", "Austria", "US"),
    ("Jordan", "Algeria", "US"),
    # Group K
    ("Portugal", "Uzbekistan", "US"),
    ("Colombia", "Congo DR", "MX"),
    # Group L
    ("England", "Ghana", "US"),
    ("Panama", "Croatia", "CA"),
]

# Matchday 3 (Jun 24-27)
third_round = [
    # Group A
    ("Czechia", "Mexico", "MX"),
    ("South Africa", "South Korea", "MX"),
    # Group B
    ("Switzerland", "Canada", "CA"),
    ("Bosnia and Herzegovina", "Qatar", "US"),
    # Group C
    ("Scotland", "Brazil", "US"),
    ("Morocco", "Haiti", "US"),
    # Group D
    ("Türkiye", "USA", "US"),
    ("Paraguay", "Australia", "US"),
    # Group E
    ("Curaçao", "Côte d'Ivoire", "US"),
    ("Ecuador", "Germany", "US"),
    # Group F
    ("Japan", "Sweden", "US"),
    ("Tunisia", "Netherlands", "US"),
    # Group G
    ("Egypt", "IR Iran", "US"),
    ("New Zealand", "Belgium", "CA"),
    # Group H
    ("Uruguay", "Spain", "MX"),
    ("Cabo Verde", "Saudi Arabia", "US"),
    # Group I
    ("Norway", "France", "US"),
    ("Senegal", "Iraq", "CA"),
    # Group J
    ("Argentina", "Jordan", "US"),
    ("Algeria", "Austria", "US"),
    # Group K
    ("Colombia", "Portugal", "US"),
    ("Congo DR", "Uzbekistan", "US"),
    # Group L
    ("Panama", "England", "US"),
    ("Croatia", "Ghana", "US"),
]

# --- Knockout stage ----------------------------------------------------------
# Round of 32 (Jun 28 - Jul 3)
round_of_32 = [
    # jun 28
    ("South Africa", "Canada", "US"),
    # mon 29
    ("Brazil", "Japan", "US"),
    ("Germany", "Paraguay", "US"),
    ("Netherlands", "Morocco", "MX"),
    # tue 30
    ("Côte d'Ivoire", "Norway", "US"),
    ("France", "Sweden", "US"),
    ("Mexico", "Ecuador", "MX"),
    # wed 01
    ("England", "Congo DR", "US"),
    ("Belgium", "Senegal", "US"),
    ("USA", "Bosnia and Herzegovina", "US"),
    # thu 02
    ("Spain", "Austria", "US"),
    ("Portugal", "Croatia", "CA"),
    ("Switzerland", "Algeria", "CA"),
    # fri 03
    ("Australia", "Egypt", "US"),
    ("Argentina", "Cabo Verde", "US"),
    ("Colombia", "Ghana", "US"),
]

# Round of 16 (Jul 4-7)
round_of_16 = [
    # sat 04
    ("Morocco", "Canada", "US"),
    ("France", "Paraguay", "US"),
    # sun 05
    ("Brazil", "Norway", "US"),
    ("Mexico", "England", "MX"),
    # mon 06
    ("Spain", "Portugal", "US"),
    ("USA", "Belgium", "US"),
    # tue 07
    ("Argentina", "Egypt", "US"),
    ("Colombia", "Switzerland", "CA"),
]

# --- Calendar ----------------------------------------------------------------
# Each entry: (fixtures, start_date, end_date). Predictions for a round train
# only on matches played BEFORE start_date (no leakage on re-runs), and its
# real results are read from the [start_date, end_date] window of the scraped
# match history.
ROUNDS = {
    "first_round": (first_round, pd.to_datetime("2026-06-11"), pd.to_datetime("2026-06-17")),
    "second_round": (second_round, pd.to_datetime("2026-06-18"), pd.to_datetime("2026-06-23")),
    "third_round": (third_round, pd.to_datetime("2026-06-24"), pd.to_datetime("2026-06-27")),
    "round_of_32": (round_of_32, pd.to_datetime("2026-06-28"), pd.to_datetime("2026-07-03")),
    "round_of_16": (round_of_16, pd.to_datetime("2026-07-04"), pd.to_datetime("2026-07-07")),
}

ROUND_LABELS = {
    "first_round": "Group stage - Matchday 1",
    "second_round": "Group stage - Matchday 2",
    "third_round": "Group stage - Matchday 3",
    "round_of_32": "Round of 32",
    "round_of_16": "Round of 16",
}

# Rounds still to be drawn (placeholders for the bracket visual).
UPCOMING_ROUNDS = ["quarter_finals", "semi_finals", "final"]
UPCOMING_LABELS = {
    "quarter_finals": "Quarter-finals",
    "semi_finals": "Semi-finals",
    "final": "Final",
}
