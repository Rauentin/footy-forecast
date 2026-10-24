from pydantic import BaseModel, Field

from pl_model.config import Settings

settings = Settings()


class Fixture(BaseModel):
    home_team: str
    away_team: str


class SimulationRequest(BaseModel):
    fixtures: list[Fixture] | None = Field(default=None, min_length=1)
    n_simulations: int = Field(default=settings.default_n_simulations, ge=1, le=settings.max_n_simulations)
    seed: int = 42


class TeamProjection(BaseModel):
    position: int
    team: str
    avg_points: float
    avg_won: float
    avg_drawn: float
    avg_lost: float
    avg_goals_scored: float
    avg_goals_conceded: float
    title_pct: float
    top4_pct: float
    relegation_pct: float


class SimulationResponse(BaseModel):
    n_simulations: int
    seed: int
    table: list[TeamProjection]
