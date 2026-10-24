from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PL_")

    data_path: Path = Path("data/epl_final.csv")
    fixtures_path: Path = Path("data/epl_2026_27_fixtures.csv")
    params_path: Path = Path("models/params.json")

    xi: float = 0.0018
    default_n_simulations: int = 1000
    max_n_simulations: int = 10000
