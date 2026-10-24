import numpy as np
import pandas as pd
import pytest

from pl_model.model import ModelParams

TEAMS = ["Alpha", "Bravo", "Charlie", "Delta"]


@pytest.fixture
def params():
    return ModelParams(
        attack={"Alpha": 0.4, "Bravo": 0.1, "Charlie": -0.1, "Delta": -0.4},
        defense={"Alpha": -0.4, "Bravo": -0.1, "Charlie": 0.1, "Delta": 0.4},
        home_adv=0.25,
        rho=-0.1,
        info={"note": "test params"},
    )


@pytest.fixture
def fixtures():
    return [(h, a) for h in TEAMS for a in TEAMS if h != a]


@pytest.fixture
def synthetic_matches(params):
    rng = np.random.default_rng(0)
    rows = []
    dates = pd.date_range("2020-01-01", periods=60, freq="7D")
    for date in dates:
        for h in TEAMS:
            for a in TEAMS:
                if h == a:
                    continue
                lam = np.exp(params.attack[h] + params.defense[a] + params.home_adv)
                mu = np.exp(params.attack[a] + params.defense[h])
                rows.append({
                    "MatchDate": date,
                    "HomeTeam": h,
                    "AwayTeam": a,
                    "FullTimeHomeGoals": rng.poisson(lam),
                    "FullTimeAwayGoals": rng.poisson(mu),
                })
    return pd.DataFrame(rows)
