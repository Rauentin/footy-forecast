import numpy as np
import pandas as pd

from pl_model.model import scoreline_grid


def load_fixtures(fixtures_path):
    fixtures = pd.read_csv(fixtures_path)
    return list(zip(fixtures["HomeTeam"], fixtures["AwayTeam"]))

def league_ranks(points, goals_for, goals_against, rng):
    goal_difference = goals_for - goals_against
    tiebreak = rng.random(points.shape)
    order = np.lexsort((tiebreak, -goals_for, -goal_difference, -points), axis=1)
    return order.argsort(axis=1)

def simulate_seasons(params, fixtures, n_simulations=1000, seed=42):
    season_teams = sorted({team for fixture in fixtures for team in fixture})
    team_idx = {team: i for i, team in enumerate(season_teams)}
    n_teams = len(season_teams)

    rng = np.random.default_rng(seed)
    goals_for = np.zeros((n_simulations, n_teams))
    goals_against = np.zeros((n_simulations, n_teams))
    won = np.zeros((n_simulations, n_teams))
    drawn = np.zeros((n_simulations, n_teams))
    lost = np.zeros((n_simulations, n_teams))
    points = np.zeros((n_simulations, n_teams))

    for home, away in fixtures:
        h, a = team_idx[home], team_idx[away]
        grid = scoreline_grid(params, home, away)
        samples = rng.choice(grid.size, size=n_simulations, p=grid.ravel())
        hg, ag = np.unravel_index(samples, grid.shape)

        home_win = hg > ag
        draw = hg == ag
        away_win = hg < ag

        goals_for[:, h] += hg
        goals_against[:, h] += ag
        goals_for[:, a] += ag
        goals_against[:, a] += hg

        won[:, h] += home_win
        drawn[:, h] += draw
        lost[:, h] += away_win
        won[:, a] += away_win
        drawn[:, a] += draw
        lost[:, a] += home_win

        points[:, h] += 3 * home_win + draw
        points[:, a] += 3 * away_win + draw

    ranks = league_ranks(points, goals_for, goals_against, rng)

    return pd.DataFrame({
        'AvgGoalsScored': goals_for.mean(axis=0),
        'AvgGoalsConceded': goals_against.mean(axis=0),
        'AvgWon': won.mean(axis=0),
        'AvgDrawn': drawn.mean(axis=0),
        'AvgLost': lost.mean(axis=0),
        'AvgPoints': points.mean(axis=0),
        'TitlePct': (ranks == 0).mean(axis=0) * 100,
        'Top4Pct': (ranks < 4).mean(axis=0) * 100,
        'RelegationPct': (ranks >= n_teams - 3).mean(axis=0) * 100,
    }, index=pd.Index(season_teams, name='Team')).sort_values('AvgPoints', ascending=False)
