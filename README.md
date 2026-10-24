# Premier League Predictor

A Dixon-Coles goal model fitted on Premier League results since 2000/01 (up to 19 Oct 2026),
served as a FastAPI app with a Gradio UI. Predicts single matches and simulates seasons with Monte Carlo.

## Run

```bash
docker compose up -d --build   # http://localhost:8000
docker compose down
```

| Endpoint | |
|---|---|
| `GET /health` | status and number of teams |
| `GET /teams` | valid team names |
| `GET /predict?home_team=Arsenal&away_team=Chelsea` | win/draw/loss probabilities, expected goals, likeliest score |
| `POST /simulate` | projected table; body `{"n_simulations": 1000, "seed": 42}`, optional `fixtures` |
| `/ui` | Gradio page |
| `/docs` | interactive API docs |

## Local development

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt   # app + pytest, httpx, ipykernel
uvicorn api.main:app --app-dir src --reload --reload-dir src
PYTHONPATH=src pytest
```

## Refit

The API only loads `models/params.json`, so the fitted model runs without any match data.
Refitting needs the match history `data/epl_final.csv`, which is not included in this repository;
it is built from the season files on [football-data.co.uk](https://www.football-data.co.uk/englandm.php). Then:

```bash
PYTHONPATH=src python -m pl_model.train   # ~2 min
docker compose up -d --build              # params.json is baked into the image
```

Settings come from `PL_`-prefixed environment variables (see `src/pl_model/config.py`),
e.g. `PL_XI` (time decay per day, default `0.0018`) or `PL_MAX_N_SIMULATIONS` (default `10000`).

## Layout

```
src/pl_model/   model, season simulation, training, config
src/api/        FastAPI app and Pydantic schemas
src/ui/         Gradio page mounted at /ui
tests/          pytest suite
notebooks/      matchday predictions
data/, models/  match history, fixtures, fitted parameters
```

## License and data

The code is MIT licensed (see `LICENSE`). The match data is not covered by that license:
the results the model was fitted on come from [football-data.co.uk](https://www.football-data.co.uk) and remain subject to their terms.
Predictions are for educational purposes only and are not betting advice.

## Roadmap (v2)

- **Data updates:** a command that fetches the latest results from football-data.co.uk, converts the
  `E0.csv` format and appends new matches to `data/epl_final.csv`, then refits.
- data analysis for finding a xi that is found in updated data
- **In-season simulation:** start `/simulate` from the current standings and only simulate the
  remaining fixtures.
- **Promoted teams:** rated only on this season's few matches; give them a prior instead.
- european top leagues support
- hardest challenge, predict UCL Games with data from top leagues
- develop functionality for player actions
- continous performance tracking
- **Theory document:** a LaTeX PDF covering the Poisson model, the Dixon-Coles correction, time decay,
  estimation and the season simulation.
