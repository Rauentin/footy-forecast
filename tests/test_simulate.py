import numpy as np
import pytest

from pl_model.simulate import league_ranks, simulate_seasons


def test_every_team_plays_every_match(params, fixtures):
    table = simulate_seasons(params, fixtures, n_simulations=200)
    games_per_team = 2 * (len(table) - 1)
    np.testing.assert_allclose(table["AvgWon"] + table["AvgDrawn"] + table["AvgLost"], games_per_team)


def test_results_are_consistent(params, fixtures):
    table = simulate_seasons(params, fixtures, n_simulations=200)
    assert table["AvgWon"].sum() == pytest.approx(table["AvgLost"].sum())
    assert table["AvgGoalsScored"].sum() == pytest.approx(table["AvgGoalsConceded"].sum())
    np.testing.assert_allclose(table["AvgPoints"], 3 * table["AvgWon"] + table["AvgDrawn"])


def test_finishing_probabilities(params, fixtures):
    table = simulate_seasons(params, fixtures, n_simulations=500)
    assert table["TitlePct"].sum() == pytest.approx(100)
    assert table["Top4Pct"].sum() == pytest.approx(400)
    assert table["RelegationPct"].sum() == pytest.approx(300)
    assert table.index[0] == "Alpha"


def test_same_seed_same_result(params, fixtures):
    first = simulate_seasons(params, fixtures, n_simulations=100, seed=7)
    second = simulate_seasons(params, fixtures, n_simulations=100, seed=7)
    assert first.equals(second)


def test_ties_broken_by_goal_difference_then_goals_scored():
    points = np.array([[10, 10, 10, 12]])
    goals_for = np.array([[5, 6, 6, 1]])
    goals_against = np.array([[5, 6, 4, 9]])
    ranks = league_ranks(points, goals_for, goals_against, np.random.default_rng(0))
    assert ranks.tolist() == [[3, 2, 1, 0]]


def test_full_ties_are_not_broken_alphabetically():
    points = np.full((1000, 3), 10)
    goals = np.full((1000, 3), 5)
    ranks = league_ranks(points, goals, goals, np.random.default_rng(0))
    champions = np.bincount(ranks.argmin(axis=1), minlength=3) / 1000
    np.testing.assert_allclose(champions, 1 / 3, atol=0.05)
