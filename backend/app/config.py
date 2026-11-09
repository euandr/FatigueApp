"""Configuração da API de detecção, carregada de backend/.env."""
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# os paramentros é apenas para encontrar o caminho correto do arquivo .env
load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def _origins():
    return tuple(
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
        ).split(",")
        if origin.strip()
    )


@dataclass
class Config:
    # tuple[str] so permite um item
    origins: tuple[str, ...] = field(default_factory=_origins)
    # Tamanho máximo do frame recebido, em caracteres base64 (2 MB por padrão).
    max_frame_base64: int = int(os.getenv("MAX_FRAME_BASE64", "2000000"))

    def validate(self):
        if self.max_frame_base64 < 1:
            raise RuntimeError("MAX_FRAME_BASE64 deve ser positivo")
