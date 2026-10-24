import gradio as gr
from fastapi import FastAPI, HTTPException
from pl_model.config import Settings
from pl_model.simulate import load_fixtures, simulate_seasons
from pl_model.model import ModelParams, predict_match
from api.schemas import SimulationRequest, SimulationResponse
from ui.app import build_ui

settings = Settings()
params = ModelParams.load(settings.params_path)
default_fixtures = load_fixtures(settings.fixtures_path)
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
    if request.fixtures is None:
        fixtures = default_fixtures
    else:
        fixtures = [(f.home_team, f.away_team) for f in request.fixtures]
        for home_team, away_team in fixtures:
            check_teams(home_team, away_team)

    table = simulate_seasons(params, fixtures, n_simulations=request.n_simulations, seed=request.seed)

    rows = table.reset_index().rename(columns=TABLE_COLUMNS).to_dict(orient="records")
    for position, row in enumerate(rows, start=1):
        row["position"] = position

    return {"n_simulations": request.n_simulations, "seed": request.seed, "table": rows}


demo = build_ui(params, default_fixtures, settings.max_n_simulations)
app = gr.mount_gradio_app(app, demo, path="/ui")
