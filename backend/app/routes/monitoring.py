"""WebSocket para processamento de frames de fadiga."""
import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..monitoring import Monitor

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws")
async def websocket(socket: WebSocket):
    config = socket.app.state.config
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
