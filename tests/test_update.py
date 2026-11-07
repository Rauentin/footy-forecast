from pathlib import Path

import pandas as pd
import pytest

from pl_model.update import E0_COLUMNS, add_new_matches, convert_e0, read_e0

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


def matches(*rows):
    return pd.DataFrame(
        [(pd.Timestamp(date), home, away, hg, ag) for date, home, away, hg, ag in rows],
        columns=list(E0_COLUMNS.values()),
    )


@pytest.fixture
def existing():
    return matches(
        ("2026-10-18", "Leeds", "Man United", 1, 1),
        ("2026-10-19", "Tottenham", "Coventry", 3, 1),
    )


@pytest.fixture
def new():
    return matches(
        ("2026-10-19", "Tottenham", "Coventry", 3, 1),
        ("2026-11-01", "Liverpool", "Arsenal", 0, 1),
        ("2026-11-01", "Chelsea", "Brentford", 2, 0),
    )


def test_add_new_matches_skips_known_matches(existing, new):
    combined = add_new_matches(existing, new)
    assert len(combined) == 4
    assert list(combined["HomeTeam"]) == ["Leeds", "Tottenham", "Liverpool", "Chelsea"]


def test_add_new_matches_twice_changes_nothing(existing, new):
    once = add_new_matches(existing, new)
    assert add_new_matches(once, new).equals(once)


def test_add_new_matches_keeps_existing_rows_first(existing, new):
    combined = add_new_matches(existing, new)
    pd.testing.assert_frame_equal(combined.iloc[:2], existing)


def test_add_new_matches_sorts_by_date(existing):
    earlier = matches(("2026-10-01", "Arsenal", "Fulham", 2, 2))
    combined = add_new_matches(existing, earlier)
    assert combined["MatchDate"].is_monotonic_increasing
    assert combined.iloc[0]["HomeTeam"] == "Arsenal"


def test_add_new_matches_ignores_extra_columns(existing):
    new = matches(("2026-11-01", "Liverpool", "Arsenal", 0, 1)).assign(HxG=1.4)
    assert list(add_new_matches(existing, new).columns) == list(existing.columns)


def test_add_new_matches_rejects_changed_score(existing):
    corrected = matches(("2026-10-19", "Tottenham", "Coventry", 2, 1))
    with pytest.raises(ValueError, match="Tottenham v Coventry on 2026-10-19"):
        add_new_matches(existing, corrected)


def test_converted_e0_adds_to_existing(e0_2026_27, existing):
    combined = add_new_matches(existing, convert_e0(e0_2026_27))
    assert len(combined) == 5
