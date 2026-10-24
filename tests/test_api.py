import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def predict(home_team, away_team):
    return client.get("/predict", params={"home_team": home_team, "away_team": away_team})

def simulate(body):
    return client.post("/simulate", json=body)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "teams": 46}


def test_teams():
    teams = client.get("/teams").json()["teams"]
    assert len(teams) == 46
    assert "Arsenal" in teams


def test_predict():
    response = predict("Arsenal", "Chelsea")
    assert response.status_code == 200

    body = response.json()
    assert body["home_win"] + body["draw"] + body["away_win"] == pytest.approx(1.0)
    assert len(body["most_likely_scorelines"]) == 1


def test_predict_unknown_team():
    response = predict("Arsenal", "Wrexham")
    assert response.status_code == 422
    assert "Wrexham" in response.json()["detail"]


def test_predict_same_team():
    response = predict("Arsenal", "Arsenal")
    assert response.status_code == 422


def test_predict_missing_team():
    response = client.get("/predict", params={"home_team": "Arsenal"})
    assert response.status_code == 422


def test_simulate_default_season():
    response = simulate({})
    assert response.status_code == 200

    body = response.json()
    assert body["n_simulations"] == 1000
    assert [row["position"] for row in body["table"]] == list(range(1, 21))
    assert sum(row["title_pct"] for row in body["table"]) == pytest.approx(100)


def test_simulate_table_is_sorted_by_points():
    table = simulate({"n_simulations": 50}).json()["table"]
    points = [row["avg_points"] for row in table]
    assert points == sorted(points, reverse=True)


def test_simulate_own_fixtures():
    body = {
        "fixtures": [
            {"home_team": "Arsenal", "away_team": "Chelsea"},
            {"home_team": "Chelsea", "away_team": "Arsenal"},
        ],
        "n_simulations": 50,
    }
    response = simulate(body)
    assert response.status_code == 200
    assert {row["team"] for row in response.json()["table"]} == {"Arsenal", "Chelsea"}


@pytest.mark.parametrize("n_simulations", [0, -5, 20000])
def test_simulate_n_simulations_out_of_range(n_simulations):
    response = simulate({"n_simulations": n_simulations})
    assert response.status_code == 422


def test_simulate_empty_fixtures():
    response = simulate({"fixtures": []})
    assert response.status_code == 422


def test_simulate_unknown_team():
    response = simulate({"fixtures": [{"home_team": "Arsenal", "away_team": "Wrexham"}]})
    assert response.status_code == 422
    assert "Wrexham" in response.json()["detail"]


def test_simulate_same_team():
    response = simulate({"fixtures": [{"home_team": "Arsenal", "away_team": "Arsenal"}]})
    assert response.status_code == 422


def test_simulate_same_seed_same_result():
    first = simulate({"n_simulations": 50, "seed": 7}).json()
    second = simulate({"n_simulations": 50, "seed": 7}).json()
    assert first == second
