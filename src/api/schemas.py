from typing import Literal

from pydantic import BaseModel, Field, model_validator

from pl_model.config import Settings

settings = Settings()


class Fixture(BaseModel):
    home_team: str
    away_team: str


class SimulationRequest(BaseModel):
    fixtures: list[Fixture] | None = Field(default=None, min_length=1)
    mode: Literal["remaining", "full"] = "remaining"
    n_simulations: int = Field(default=settings.default_n_simulations, ge=1, le=settings.max_n_simulations)
    seed: int = 42

    @model_validator(mode="after")
    def mode_needs_default_fixtures(self):
        # own fixtures are always simulated from scratch, so a mode would be ignored
        if self.fixtures is not None and "mode" in self.model_fields_set:
            raise ValueError("mode only applies to the default fixture list, not to your own fixtures")
        return self


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
    mode: Literal["remaining", "full", "custom"]
    matches_played: int
    matches_simulated: int
    n_simulations: int
    seed: int
    table: list[TeamProjection]
