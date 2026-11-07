import gradio as gr
from fastapi import FastAPI, HTTPException
from pl_model.config import Settings
from pl_model.simulate import load_fixtures, remaining_fixtures, simulate_seasons
from pl_model.model import ModelParams, load_data, predict_match
from api.schemas import SimulationRequest, SimulationResponse
from ui.app import build_ui

settings = Settings()
params = ModelParams.load(settings.params_path)
default_fixtures = load_fixtures(settings.fixtures_path)
season_results = load_data(settings.results_path)
# fails at startup if the results don't fit the fixture list (wrong season, misspelt team)
n_remaining = len(remaining_fixtures(default_fixtures, season_results))
app = FastAPI()

TABLE_COLUMNS = {
    "Team": "team",
    "AvgPoints": "avg_points",
    "AvgWon": "avg_won",
    "AvgDrawn": "avg_drawn",
    "AvgLost": "avg_lost",
    "AvgGoalsScored": "avg_goals_scored",
    "AvgGoalsConceded": "avg_goals_conceded",
    "TitlePct": "title_pct",
    "Top4Pct": "top4_pct",
    "RelegationPct": "relegation_pct",
}


def check_teams(home_team, away_team):
    for team in (home_team, away_team):
        if team not in params.attack:
            raise HTTPException(status_code=422, detail=f"Unknown team: {team}. See GET /teams for valid names.")

    if home_team == away_team:
        raise HTTPException(status_code=422, detail="Home and away teams must be different.")


@app.get("/health")
async def get_health():
    return {"status": "ok", "teams": len(params.teams)}

@app.get("/teams")
async def get_teams():
    return {"teams": params.teams}

@app.get("/predict")
async def get_predict(home_team: str, away_team: str):
    check_teams(home_team, away_team)
    return predict_match(params, home_team, away_team, n_scorelines=1)

@app.post("/simulate", response_model=SimulationResponse)
def simulate(request: SimulationRequest):
    if request.fixtures is not None:
        mode, played = "custom", None
        fixtures = [(f.home_team, f.away_team) for f in request.fixtures]
        for home_team, away_team in fixtures:
            check_teams(home_team, away_team)
        matches_played, matches_simulated = 0, len(fixtures)
    elif request.mode == "remaining":
        mode, played, fixtures = "remaining", season_results, default_fixtures
        matches_played, matches_simulated = len(season_results), n_remaining
    else:
        mode, played, fixtures = "full", None, default_fixtures
        matches_played, matches_simulated = 0, len(fixtures)

    table = simulate_seasons(
        params, fixtures, n_simulations=request.n_simulations, seed=request.seed, played=played
    )

    rows = table.reset_index().rename(columns=TABLE_COLUMNS).to_dict(orient="records")
    for position, row in enumerate(rows, start=1):
        row["position"] = position

    return {
        "mode": mode,
        "matches_played": matches_played,
        "matches_simulated": matches_simulated,
        "n_simulations": request.n_simulations,
        "seed": request.seed,
        "table": rows,
    }


demo = build_ui(params, default_fixtures, season_results, settings.max_n_simulations)
app = gr.mount_gradio_app(app, demo, path="/ui")
