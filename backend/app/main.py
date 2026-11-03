"""API de detecção de fadiga: apenas WebSocket, sem auth e sem banco.

Protocolo:
  1. Cliente abre ws://host:8000/ws.
  2. Servidor responde {"type": "ready"}.
  3. Cliente envia {"frame": "<jpeg em base64>"}.
  4. Servidor responde com o resultado da detecção.
"""
import asyncio
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .config import Config
from .monitoring import Monitor

logger = logging.getLogger(__name__)


def create_app(config=None):
    config = config or Config()
    config.validate()

    app = FastAPI(title="API Detec. F.")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(config.origins),
        allow_methods=["GET"],
        allow_headers=["Content-Type"],
    )
    app.state.config = config

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.websocket("/ws")
    async def websocket(socket: WebSocket):
        await socket.accept()
        monitor = Monitor()
        await socket.send_json({"type": "ready"})
        try:
            while True:
                try:
                    data = await socket.receive_json()
                except ValueError:
                    await socket.send_json(
                        {"type": "error", "message": "Mensagem JSON inválida"}
                    )
                    continue

                if not isinstance(data, dict):
                    await socket.send_json(
                        {"type": "error", "message": 'Envie {"frame": "<base64>"}'}
                    )
                    continue

                # Mensagem de controle antiga (token/sessão): ignorada para não
                # derrubar clientes que ainda fazem o handshake anterior.
                if data.get("type") == "auth":
                    continue

                if data.get("type") == "ping":
                    await socket.send_json({"type": "pong"})
                    continue

                frame = data.get("frame")
                if not isinstance(frame, str) or not frame:
                    await socket.send_json(
                        {"type": "error", "message": "Envie uma mensagem com frame base64"}
                    )
                    continue
                if len(frame) > config.max_frame_base64:
                    await socket.send_json(
                        {
                            "type": "error",
                            "message": "Frame excede o limite configurado em MAX_FRAME_BASE64",
                        }
                    )
                    continue

                try:
                    result = await asyncio.get_running_loop().run_in_executor(
                        monitor.executor, monitor.process, frame
                    )
                except ValueError as error:
                    await socket.send_json({"type": "error", "message": str(error)})
                    continue

                await socket.send_json(result)
        except WebSocketDisconnect:
            pass
        except Exception:
            logger.exception("Erro no processamento WebSocket")
            try:
                await socket.close(code=1011)
            except (RuntimeError, OSError):
                pass
        finally:
            await monitor.close()

    return app


app = create_app()
