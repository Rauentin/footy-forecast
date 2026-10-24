import numpy as np
import pytest

from pl_model.model import ModelParams, fit, predict_match, scoreline_grid, tau_rho, tau_rho_vec


def test_tau_rho_vec_matches_tau_rho():
    x = np.array([0, 0, 1, 1, 2, 3])
    y = np.array([0, 1, 0, 1, 1, 0])
    lam = np.array([1.2, 1.5, 0.8, 2.0, 1.1, 1.7])
    mu = np.array([0.9, 1.3, 1.4, 0.7, 1.0, 1.2])
    rho = -0.12

    expected = [tau_rho(*args, rho) for args in zip(x, y, lam, mu)]
    np.testing.assert_allclose(tau_rho_vec(x, y, lam, mu, rho), expected)


def test_scoreline_grid_is_a_probability_distribution(params):
    grid = scoreline_grid(params, "Alpha", "Delta")
    assert grid.shape == (16, 16)
    assert np.all(grid >= 0)
    assert grid.sum() == pytest.approx(1.0)


def test_predict_match_probabilities_add_up(params):
    prediction = predict_match(params, "Alpha", "Delta")
    total = prediction["home_win"] + prediction["draw"] + prediction["away_win"]
    assert total == pytest.approx(1.0)


def test_stronger_team_is_favourite(params):
    prediction = predict_match(params, "Alpha", "Delta")
    assert prediction["home_win"] > prediction["away_win"]
    assert prediction["expected_home_goals"] > prediction["expected_away_goals"]


def test_most_likely_scorelines_are_sorted(params):
    scorelines = predict_match(params, "Bravo", "Charlie")["most_likely_scorelines"]
    probabilities = [s["probability"] for s in scorelines]
    assert len(scorelines) == 5
    assert probabilities == sorted(probabilities, reverse=True)


def test_fit_recovers_known_strengths(params, synthetic_matches):
    fitted = fit(synthetic_matches, xi=0.0)

    assert fitted.info["converged"]
    assert fitted.home_adv == pytest.approx(params.home_adv, abs=0.1)
    true_attack = np.array([params.attack[t] for t in fitted.teams])
    fitted_attack = np.array([fitted.attack[t] for t in fitted.teams])
    np.testing.assert_allclose(fitted_attack - fitted_attack.mean(), true_attack - true_attack.mean(), atol=0.15)


def test_params_round_trip(params, tmp_path):
    path = tmp_path / "params.json"
    params.save(path)
    assert ModelParams.load(path) == params
