"""Estado transitório e ciclo de vida das sessões (um processo Uvicorn)."""
import asyncio
import base64
import binascii
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from time import monotonic
from uuid import uuid4

from fastapi import HTTPException

logger = logging.getLogger(__name__)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


class Monitor:
    def __init__(self, session):
        self.session = session
        self.socket = None
        self.last_seen = monotonic()
        self.detector = None
        # MediaPipe de cada câmera é criado e utilizado sempre na mesma thread.
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.alerts = {}

    def process(self, encoded):
        import cv2
        import numpy as np
        from .detector import Detector

        try:
            raw = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error):
            raise ValueError("Frame base64 inválido")
        frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Imagem inválida")
        if self.detector is None:
            self.detector = Detector()
        return self.detector.process_frame(frame)

    async def close(self):
        if self.socket:
            try:
                await self.socket.close(code=1000)
            except (RuntimeError, OSError):
                pass
        if self.detector:
            await asyncio.get_running_loop().run_in_executor(self.executor, self.detector.close)
        self.executor.shutdown(wait=False)


class Monitoring:
    def __init__(self, db, config):
        self.db = db
        self.config = config
        self.monitors = {}
        self.lock = asyncio.Lock()

    async def device(self, device_id):
        rows = await self.db.query("devices", params={
            "id": f"eq.{device_id}", "select": "id,name,created_at,is_active",
        })
        if not rows:
            raise HTTPException(404, "Câmera não encontrada")
        return rows[0]

    async def owned_session(self, session_id, user_id):
        rows = await self.db.query("sessions", params={
            "id": f"eq.{session_id}", "user_id": f"eq.{user_id}", "select": "*",
        })
        if not rows:
            raise HTTPException(404, "Sessão não encontrada")
        return rows[0]

    async def start(self, device_id, user_id):
        async with self.lock:
            device = await self.device(device_id)
            if not device["is_active"]:
                raise HTTPException(409, "Câmera desabilitada")
            opened = await self.db.query("sessions", params={
                "device_id": f"eq.{device_id}", "ended_at": "is.null", "select": "id",
            })
            if opened:
                raise HTTPException(409, "Câmera já possui uma sessão aberta")
            rows = await self.db.query("sessions", "POST", data={
                "id": str(uuid4()), "device_id": device_id, "user_id": user_id,
                "started_at": timestamp(),
            })
            session = rows[0]
            self.monitors[session["id"]] = Monitor(session)
            return session

    async def finish_locked(self, session_id):
        rows = await self.db.query("sessions", "PATCH", params={
            "id": f"eq.{session_id}", "ended_at": "is.null",
        }, data={"ended_at": timestamp()})
        monitor = self.monitors.pop(session_id, None)
        if monitor:
            await monitor.close()
        return rows[0] if rows else None

    async def reconcile(self):
        # Apenas um processo pode gerenciar este banco de monitoramento.
        await self.db.query("sessions", "PATCH", params={"ended_at": "is.null"},
                            data={"ended_at": timestamp()})

    async def persist_alerts(self, monitor, result):
        alerts = [
            ("yawnDetected", "Bocejo detectado", "MAR", "mar"),
            ("excessBlinks", f"Excesso de piscadas detectado ({result['blinks']} em 30 frames)", "blinks", "blinks"),
            ("fatigueAlert", "Alerta de fadiga detectado - olhos fechados por muito tempo", "EAR", "ear"),
        ]
        for flag, message, metric, key in alerts:
            active = bool(result.get(flag))
            if active and not monitor.alerts.get(flag):
                value = result[key]
                data = {
                    "id": str(uuid4()), "session_id": monitor.session["id"],
                    "event_type": message, "value": value, "created_at": timestamp(),
                }
                if self.config.event_schema == "numeric":
                    data["metadata"] = {"metric": metric}
                else:
                    data["value"] = f"Blinks: {value}" if metric == "blinks" else f"{metric}: {value:.3f}"
                await self.db.query("events", "POST", data=data)
            # Persistência por início/rearme; não altera os alertas do detector.
            monitor.alerts[flag] = active

    async def sweep(self):
        while True:
            await asyncio.sleep(1)
            try:
                async with self.lock:
                    for session_id, monitor in list(self.monitors.items()):
                        if monotonic() - monitor.last_seen > self.config.reconnect_seconds:
                            await self.finish_locked(session_id)
            except Exception:
                logger.exception("Erro ao encerrar sessões sem frames; nova tentativa em 1 segundo")
