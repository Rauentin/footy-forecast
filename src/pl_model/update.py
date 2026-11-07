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
