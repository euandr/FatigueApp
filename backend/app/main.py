import asyncio
import logging
from contextlib import asynccontextmanager, suppress
from time import monotonic
from uuid import UUID

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .config import Config
from .database import Database
from .monitoring import Monitoring
from .routes import router

logger = logging.getLogger(__name__)


def create_app(config=None, database=None):
    config = config or Config()

    @asynccontextmanager
    async def lifespan(app):
        if database is None:
            config.validate()
        db = database or Database(config)
        manager = Monitoring(db, config)
        app.state.db = db
        app.state.monitoring = manager
        sweeper = None
        try:
            await db.check_schema()
            await manager.reconcile()
            sweeper = asyncio.create_task(manager.sweep())
            yield
        finally:
            if sweeper:
                sweeper.cancel()
                with suppress(asyncio.CancelledError):
                    await sweeper
            for session_id in list(manager.monitors):
                try:
                    async with manager.lock:
                        await manager.finish_locked(session_id)
                except Exception:
                    logger.exception("Falha ao encerrar sessão no desligamento")
                    monitor = manager.monitors.pop(session_id, None)
                    if monitor:
                        await monitor.close()
            await db.close()

    app = FastAPI(title="API de Detecção de Fadiga", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=list(config.origins),
                       allow_methods=["GET", "POST", "PATCH"],
                       allow_headers=["Authorization", "Content-Type"])
    app.include_router(router)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.websocket("/ws")
    async def websocket(socket: WebSocket):
        # O navegador não permite Authorization no handshake WebSocket.
        # Token na primeira mensagem evita credenciais em URLs e logs de acesso.
        origin = socket.headers.get("origin")
        if origin and origin not in config.origins:
            await socket.close(code=1008)
            return
        await socket.accept()
        manager = socket.app.state.monitoring
        monitor = None
        try:
            auth = await asyncio.wait_for(socket.receive_json(), timeout=10)
            if not isinstance(auth, dict) or auth.get("type") != "auth" or not isinstance(auth.get("token"), str):
                raise HTTPException(401, "Autenticação WebSocket obrigatória")
            user = await manager.db.user(auth["token"])
            session_id = str(UUID(str(auth.get("session_id", ""))))
            async with manager.lock:
                session = await manager.owned_session(session_id, user["id"])
                monitor = manager.monitors.get(session_id)
                if session["ended_at"] or monitor is None:
                    raise HTTPException(409, "Sessão encerrada; inicie um novo monitoramento")
                if monitor.socket is not None:
                    raise HTTPException(409, "Sessão já conectada")
                if not (await manager.device(session["device_id"]))["is_active"]:
                    raise HTTPException(409, "Câmera desabilitada")
                monitor.socket = socket
                monitor.last_seen = monotonic()
            await socket.send_json({"type": "ready", "session_id": session_id})
            validated_at = monotonic()
            device_checked_at = monotonic()
            while True:
                data = await asyncio.wait_for(socket.receive_json(), timeout=config.reconnect_seconds)
                if not isinstance(data, dict) or not isinstance(data.get("frame"), str):
                    raise ValueError("Envie uma mensagem com frame base64")
                if len(data["frame"]) > 2_000_000:
                    raise ValueError("Frame excede o limite de 2 MB em base64")
                if monotonic() - validated_at >= 30:
                    await manager.db.user(auth["token"])
                    validated_at = monotonic()
                async with manager.lock:
                    if manager.monitors.get(session_id) is not monitor:
                        break
                    monitor.last_seen = monotonic()
                    # A API já encerra o monitoramento ao desabilitar a câmera.
                    # Conferir alterações externas periodicamente, não a cada frame:
                    # uma ida ao Supabase por imagem limita o FPS à latência do banco.
                    if monotonic() - device_checked_at >= 5:
                        if not (await manager.device(session["device_id"]))["is_active"]:
                            await manager.finish_locked(session_id)
                            break
                        device_checked_at = monotonic()
                    result = await asyncio.get_running_loop().run_in_executor(
                        monitor.executor, monitor.process, data["frame"]
                    )
                    try:
                        await manager.persist_alerts(monitor, result)
                    except HTTPException:
                        # A falha fica visível sem interromper vídeo e métricas.
                        logger.exception("Falha ao persistir alerta")
                        await socket.send_json({"type": "error", "message": "Falha ao salvar evento no banco"})
                await socket.send_json(result)
        except WebSocketDisconnect:
            pass
        except (HTTPException, ValueError, asyncio.TimeoutError) as error:
            code = 1008
            if isinstance(error, HTTPException):
                if error.status_code == 503:
                    code = 1013  # Indisponibilidade temporária: permitir reconexão.
                elif error.status_code == 401 and monitor and monitor.socket is socket:
                    code = 1012  # Reconectar com o token renovado pelo Supabase Auth.
            with suppress(RuntimeError, OSError):
                await socket.send_json({"type": "error", "message": getattr(error, "detail", str(error)) or "Tempo de espera excedido"})
                await socket.close(code=code)
        except Exception:
            logger.exception("Erro no processamento WebSocket")
            with suppress(RuntimeError, OSError):
                await socket.close(code=1011)
        finally:
            if monitor and monitor.socket is socket:
                monitor.socket = None

    return app


app = create_app()
