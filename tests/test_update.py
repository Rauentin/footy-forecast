from pathlib import Path

import pandas as pd
import pytest

from pl_model.update import E0_COLUMNS, convert_e0, read_e0

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def e0_2026_27():
    return read_e0(FIXTURES / "e0_2026_27.csv")


def test_columns_match_epl_final_format(e0_2026_27):
    converted = convert_e0(e0_2026_27)
    assert list(converted.columns) == list(E0_COLUMNS.values())


def test_known_row_converts(e0_2026_27):
    converted = convert_e0(e0_2026_27)
    row = converted.iloc[-1]
    assert row["MatchDate"] == pd.Timestamp("2026-11-02")
    assert (row["HomeTeam"], row["AwayTeam"]) == ("Newcastle", "Everton")
    assert (row["FullTimeHomeGoals"], row["FullTimeAwayGoals"]) == (0, 1)


def test_dates_are_day_first(e0_2026_27):
    dates = convert_e0(e0_2026_27)["MatchDate"]
    assert list(dates.dt.month) == [10, 11, 11]


def test_goals_are_integers(e0_2026_27):
    converted = convert_e0(e0_2026_27)
    assert converted["FullTimeHomeGoals"].dtype == "int64"
    assert converted["FullTimeAwayGoals"].dtype == "int64"


def test_columns_selected_by_name_not_position():
    # last season's file has no HxG/AxG columns, so every later column sits at a different position
    raw = read_e0(FIXTURES / "e0_2025_26.csv")
    assert "HxG" not in raw.columns

    row = convert_e0(raw).iloc[0]
    assert (row["HomeTeam"], row["AwayTeam"]) == ("Liverpool", "Bournemouth")
    assert (row["FullTimeHomeGoals"], row["FullTimeAwayGoals"]) == (4, 2)


def test_written_dates_match_epl_final(e0_2026_27, tmp_path):
    path = tmp_path / "out.csv"
    convert_e0(e0_2026_27).to_csv(path, index=False)
    assert path.read_text().splitlines()[-1] == "2026-11-02,Newcastle,Everton,0,1"


def test_trailing_empty_rows_are_dropped(e0_2026_27):
    padded = pd.concat([e0_2026_27, pd.DataFrame([{}] * 2, columns=e0_2026_27.columns)])
    assert len(convert_e0(padded)) == 3


def test_missing_column_raises(e0_2026_27):
    with pytest.raises(ValueError, match="FTHG"):
        convert_e0(e0_2026_27.drop(columns="FTHG"))
