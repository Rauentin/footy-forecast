from collections import Counter

import numpy as np
import pandas as pd

from pl_model.model import scoreline_grid


def load_fixtures(fixtures_path):
    fixtures = pd.read_csv(fixtures_path)
    return list(zip(fixtures["HomeTeam"], fixtures["AwayTeam"]))

def remaining_fixtures(fixtures, played):
    """Return the fixtures that have not been played yet, in fixture-list order.

    `played` must hold this season's matches only: in a season every home/away pairing
    happens exactly once, so a pairing played twice or not in the fixture list means
    the input mixes seasons or has misspelt team names.
    """
    played_pairs = Counter(zip(played["HomeTeam"], played["AwayTeam"]))

    duplicates = sorted(pair for pair, count in played_pairs.items() if count > 1)
    if duplicates:
        raise ValueError(f"Matches played more than once (more than one season?): {duplicates}")

    unknown = sorted(set(played_pairs) - set(fixtures))
    if unknown:
        raise ValueError(f"Played matches not in the fixture list: {unknown}")

    return [fixture for fixture in fixtures if fixture not in played_pairs]

def league_ranks(points, goals_for, goals_against, rng):
    goal_difference = goals_for - goals_against
    tiebreak = rng.random(points.shape)
    order = np.lexsort((tiebreak, -goals_for, -goal_difference, -points), axis=1)
    return order.argsort(axis=1)

TABLE_STATS = ("goals_for", "goals_against", "won", "drawn", "lost", "points")


def record_results(table, h, a, hg, ag):
    """Add results to the table arrays at team positions h and a.

    Works for one real match (scalar goals, 1-D arrays) and for one fixture across
    all simulations (goal vectors, 2-D arrays): `...` selects every leading axis.
    """
    home_win = hg > ag
    draw = hg == ag
    away_win = hg < ag

    table["goals_for"][..., h] += hg
    table["goals_against"][..., h] += ag
    table["goals_for"][..., a] += ag
    table["goals_against"][..., a] += hg

    table["won"][..., h] += home_win
    table["drawn"][..., h] += draw
    table["lost"][..., h] += away_win
    table["won"][..., a] += away_win
    table["drawn"][..., a] += draw
    table["lost"][..., a] += home_win

    table["points"][..., h] += 3 * home_win + draw
    table["points"][..., a] += 3 * away_win + draw


def simulate_seasons(params, fixtures, n_simulations=1000, seed=42, played=None):
    """Simulate `fixtures` n_simulations times and average the final tables.

    Without `played`, every fixture is simulated and all teams start on zero.
    With `played` (this season's results so far), `fixtures` must be the full season:
    the played matches count with their real scores and only the rest is simulated.
    """
    season_teams = sorted({team for fixture in fixtures for team in fixture})
    team_idx = {team: i for i, team in enumerate(season_teams)}
    n_teams = len(season_teams)

    current = {stat: np.zeros(n_teams) for stat in TABLE_STATS}
    if played is not None:
        fixtures = remaining_fixtures(fixtures, played)
        for home, away, hg, ag in zip(
            played["HomeTeam"], played["AwayTeam"], played["FullTimeHomeGoals"], played["FullTimeAwayGoals"]
        ):
            record_results(current, team_idx[home], team_idx[away], int(hg), int(ag))

    table = {stat: np.tile(values, (n_simulations, 1)) for stat, values in current.items()}

    rng = np.random.default_rng(seed)
    for home, away in fixtures:
        grid = scoreline_grid(params, home, away)
        samples = rng.choice(grid.size, size=n_simulations, p=grid.ravel())
        hg, ag = np.unravel_index(samples, grid.shape)
        record_results(table, team_idx[home], team_idx[away], hg, ag)

    ranks = league_ranks(table["points"], table["goals_for"], table["goals_against"], rng)

    return pd.DataFrame({
        'AvgGoalsScored': table["goals_for"].mean(axis=0),
        'AvgGoalsConceded': table["goals_against"].mean(axis=0),
        'AvgWon': table["won"].mean(axis=0),
        'AvgDrawn': table["drawn"].mean(axis=0),
        'AvgLost': table["lost"].mean(axis=0),
        'AvgPoints': table["points"].mean(axis=0),
        'TitlePct': (ranks == 0).mean(axis=0) * 100,
        'Top4Pct': (ranks < 4).mean(axis=0) * 100,
        'RelegationPct': (ranks >= n_teams - 3).mean(axis=0) * 100,
    }, index=pd.Index(season_teams, name='Team')).sort_values('AvgPoints', ascending=False)
