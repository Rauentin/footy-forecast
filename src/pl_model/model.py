import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df["MatchDate"] = pd.to_datetime(df["MatchDate"])
    df = df.sort_values("MatchDate").reset_index(drop=True)
    return df

def time_decay(MatchDates, xi):
    most_recent = max(MatchDates)
    deltas = (most_recent - MatchDates).dt.days.values
    return np.exp(-xi * deltas)

def tau_rho(x, y, lam, mu, rho):
    if x == 0 and y == 0:
        return 1 - (lam * mu * rho)
    elif x == 0 and y == 1:
        return 1 + (lam * rho)
    elif x == 1 and y == 0:
        return 1 + (mu * rho)
    elif x == 1 and y == 1:
        return 1 - rho
    else:
        return 1.0

def tau_rho_vec(x, y, lam, mu, rho):
    tau = np.ones_like(lam)
    for i in range(len(x)):
        tau[i] = tau_rho(x[i], y[i], lam[i], mu[i], rho)
    return tau

def build_match_arrays(df, teams):
    team_idx = {t: i for i, t in enumerate(teams)}
    home_idx = df['HomeTeam'].map(team_idx).to_numpy()
    away_idx = df['AwayTeam'].map(team_idx).to_numpy()
    hg = df['FullTimeHomeGoals'].to_numpy()
    ag = df['FullTimeAwayGoals'].to_numpy()
    return home_idx, away_idx, hg, ag

def neg_log_likelihood_vec(params, n_teams, home_idx, away_idx, hg, ag, weights):
    attack = params[:n_teams]
    defense = params[n_teams:2 * n_teams]
    home_adv = params[2 * n_teams]
    rho = params[2 * n_teams + 1]

    lam = np.exp(attack[home_idx] + defense[away_idx] + home_adv)
    mu = np.exp(attack[away_idx] + defense[home_idx])

    correction = tau_rho_vec(hg, ag, lam, mu, rho)

    match_ll = poisson.logpmf(hg, lam) + poisson.logpmf(ag, mu) + np.log(correction)
    return -np.sum(weights * match_ll)


class ModelParams:
    def __init__(self, attack, defense, home_adv, rho, info=None):
        self.attack = attack
        self.defense = defense
        self.home_adv = home_adv
        self.rho = rho
        self.info = info if info is not None else {}

    def __repr__(self):
        return (
            f"ModelParams(teams={len(self.attack)}, home_adv={self.home_adv:.4f}, "
            f"rho={self.rho:.4f}, info={self.info})"
        )

    def __eq__(self, other):
        if not isinstance(other, ModelParams):
            return NotImplemented
        return (
            self.attack == other.attack
            and self.defense == other.defense
            and self.home_adv == other.home_adv
            and self.rho == other.rho
            and self.info == other.info
        )

    @property
    def teams(self):
        return sorted(self.attack)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "attack": self.attack,
            "defense": self.defense,
            "home_adv": self.home_adv,
            "rho": self.rho,
            "info": self.info,
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        return cls(data["attack"], data["defense"], data["home_adv"], data["rho"], data.get("info", {}))


def fit(df, xi=0.0018):
    teams = sorted(pd.concat([df['HomeTeam'], df['AwayTeam']]).unique())
    n_teams = len(teams)

    home_idx, away_idx, hg, ag = build_match_arrays(df, teams)
    weights = time_decay(df['MatchDate'], xi)
    init_attack = np.zeros(n_teams)
    init_defense = np.zeros(n_teams)
    home_adv = 0.1
    rho = 0.0
    params0 = np.concatenate([init_attack, init_defense, [home_adv], [rho]])

    bounds = [(None, None)] * (2 * n_teams + 1) + [(-1, 1)]

    result = minimize(
        neg_log_likelihood_vec,
        params0,
        args=(n_teams, home_idx, away_idx, hg, ag, weights),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxfun": 200000, "maxiter": 200000},
    )

    attack = {t: float(v) for t, v in zip(teams, result.x[:n_teams])}
    defense = {t: float(v) for t, v in zip(teams, result.x[n_teams:2 * n_teams])}
    info = {
        "xi": xi,
        "n_matches": len(df),
        "first_match_date": df["MatchDate"].min().date().isoformat(),
        "last_match_date": df["MatchDate"].max().date().isoformat(),
        "log_likelihood": float(-result.fun),
        "converged": bool(result.success),
    }
    return ModelParams(attack, defense, float(result.x[2 * n_teams]), float(result.x[2 * n_teams + 1]), info)

def scoreline_grid(params, home, away, max_goals=15):
    lam = np.exp(params.attack[home] + params.defense[away] + params.home_adv)
    mu = np.exp(params.attack[away] + params.defense[home])

    goal_range = np.arange(max_goals + 1)
    grid = np.outer(poisson.pmf(goal_range, lam), poisson.pmf(goal_range, mu))

    for x in (0, 1):
        for y in (0, 1):
            grid[x, y] *= tau_rho(x, y, lam, mu, params.rho)

    grid /= grid.sum()
    return grid

def predict_match(params, home, away, max_goals=15, n_scorelines=5):
    grid = scoreline_grid(params, home, away, max_goals)
    goal_range = np.arange(max_goals + 1)

    most_likely = np.argsort(grid, axis=None)[::-1][:n_scorelines]
    scorelines = [
        {"home_goals": int(x), "away_goals": int(y), "probability": float(grid[x, y])}
        for x, y in zip(*np.unravel_index(most_likely, grid.shape))
    ]

    return {
        "home_win": float(np.tril(grid, -1).sum()),
        "draw": float(np.trace(grid)),
        "away_win": float(np.triu(grid, 1).sum()),
        "expected_home_goals": float((grid.sum(axis=1) * goal_range).sum()),
        "expected_away_goals": float((grid.sum(axis=0) * goal_range).sum()),
        "most_likely_scorelines": scorelines,
    }
