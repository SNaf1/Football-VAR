from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Atletico Intelligence API"
    app_env: str = "local"
    storage_dir: str = "backend/storage"
    model_device: str = "auto"
    cors_origins: str = "http://localhost:5173"

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[1] / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def backend_dir(self) -> Path:
        return Path(__file__).resolve().parents[1]

    @property
    def project_root(self) -> Path:
        return self.backend_dir.parent

    @property
    def storage_path(self) -> Path:
        path = Path(self.storage_dir)
        if not path.is_absolute():
            path = self.project_root / path
        return path

    @property
    def media_path(self) -> Path:
        return self.storage_path / "media"

    @property
    def sample_clips_path(self) -> Path:
        return self.project_root / "assets" / "sample-clips"

    @property
    def metadata_path(self) -> Path:
        return self.storage_path / "metadata.json"

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
