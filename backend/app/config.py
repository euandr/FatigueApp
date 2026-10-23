import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


@dataclass
class Config:
    supabase_url: str = os.getenv("SUPABASE_URL", "").rstrip("/")
    service_key: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    origins: tuple = tuple(origin.strip() for origin in os.getenv(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",") if origin.strip())
    event_schema: str = os.getenv("EVENT_SCHEMA", "numeric")
    reconnect_seconds: int = int(os.getenv("RECONNECT_SECONDS", "30"))

    def validate(self):
        if not self.supabase_url or not self.service_key:
            raise RuntimeError("Configure SUPABASE_URL e SUPABASE_SERVICE_ROLE_KEY em backend/.env")
        if self.event_schema not in {"numeric", "legacy"}:
            raise RuntimeError("EVENT_SCHEMA deve ser numeric ou legacy")
        if self.reconnect_seconds < 1:
            raise RuntimeError("RECONNECT_SECONDS deve ser positivo")
