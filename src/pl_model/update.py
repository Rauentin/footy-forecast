import pandas as pd

E0_URL = "https://www.football-data.co.uk/mmz4281/{season_code}/E0.csv"

# football-data.co.uk column -> epl_final.csv column, in epl_final.csv order
E0_COLUMNS = {
    "Date": "MatchDate",
    "HomeTeam": "HomeTeam",
    "AwayTeam": "AwayTeam",
    "FTHG": "FullTimeHomeGoals",
    "FTAG": "FullTimeAwayGoals",
}

MATCH_KEY = ["MatchDate", "HomeTeam", "AwayTeam"]


def read_e0(source):
    """Read an E0.csv from a path or URL. utf-8-sig strips the byte-order mark."""
    return pd.read_csv(source, encoding="utf-8-sig")


def convert_e0(raw):
    """Convert a football-data.co.uk E0 table into the epl_final.csv format."""
    missing = set(E0_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"E0 data is missing columns: {sorted(missing)}")

    df = raw[list(E0_COLUMNS)].rename(columns=E0_COLUMNS)
    df = df.dropna(how="all")

    df["MatchDate"] = pd.to_datetime(df["MatchDate"], format="%d/%m/%Y")
    df["FullTimeHomeGoals"] = df["FullTimeHomeGoals"].astype(int)
    df["FullTimeAwayGoals"] = df["FullTimeAwayGoals"].astype(int)

    return df.sort_values("MatchDate", kind="stable").reset_index(drop=True)


def add_new_matches(existing, new):
    """Append the matches in `new` that are not in `existing` yet.

    A match is identified by date, home team and away team. Running this twice with the
    same data changes nothing. If a match is in both but the scores differ, a ValueError
    is raised instead of silently keeping either version.
    """
    new = new[list(existing.columns)]

    overlap = existing.merge(new, on=MATCH_KEY, suffixes=("_existing", "_new"))
    changed = overlap[
        (overlap["FullTimeHomeGoals_existing"] != overlap["FullTimeHomeGoals_new"])
        | (overlap["FullTimeAwayGoals_existing"] != overlap["FullTimeAwayGoals_new"])
    ]
    if not changed.empty:
        matches = ", ".join(
            f"{m.HomeTeam} v {m.AwayTeam} on {m.MatchDate.date()}" for m in changed.itertuples()
        )
        raise ValueError(f"Scores differ from the existing data: {matches}")

    combined = pd.concat([existing, new], ignore_index=True)
    combined = combined.drop_duplicates(subset=MATCH_KEY, keep="first")
    return combined.sort_values("MatchDate", kind="stable").reset_index(drop=True)
