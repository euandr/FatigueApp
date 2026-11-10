"""API de detecção de fadiga: apenas WebSocket, sem auth e sem banco.

Protocolo:
  1. Cliente abre ws://host:8000/ws.
  2. Servidor responde {"type": "ready"}.
  3. Cliente envia {"frame": "<jpeg em base64>"}.
  4. Servidor responde com o resultado da detecção.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import Config
from .routes import health, monitoring, cameras


def create_app(config=None):
    config = config or Config()
    config.validate()

    app = FastAPI(title="API Detec. F.")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(config.origins),
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["Content-Type"],
    )
    app.state.config = config

    app.include_router(health.router)
    app.include_router(monitoring.router)
    app.include_router(cameras.router)

    return app


app = create_app()
