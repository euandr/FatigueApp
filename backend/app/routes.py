from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Request

from .auth import authenticated_user
from .monitoring import timestamp
from .schemas import DeviceCreate, DeviceUpdate, SessionCreate

router = APIRouter(prefix="/api", dependencies=[Depends(authenticated_user)])


@router.get("/devices")
async def devices(request: Request, active: bool | None = None):
    params = {"select": "id,name,created_at,is_active", "order": "created_at.desc"}
    if active is not None:
        params["is_active"] = f"eq.{str(active).lower()}"
    return await request.app.state.db.query("devices", params=params)


@router.post("/devices", status_code=201)
async def create_device(body: DeviceCreate, request: Request):
    # Câmeras compartilhadas: decisão explícita do responsável pelo projeto.
    async with request.app.state.monitoring.lock:
        rows = await request.app.state.db.query("devices", params={
            "name": f"eq.{body.name}", "is_active": "eq.true",
            "select": "id,name,created_at,is_active", "limit": "1",
        })
        if rows:
            return rows[0]
        rows = await request.app.state.db.query("devices", "POST", params={
            "select": "id,name,created_at,is_active",
        }, data={
            "id": str(uuid4()), "name": body.name, "is_active": True, "created_at": timestamp(),
        })
        return rows[0]


@router.get("/devices/{device_id}")
async def get_device(device_id: UUID, request: Request):
    return await request.app.state.monitoring.device(str(device_id))


@router.patch("/devices/{device_id}")
async def update_device(device_id: UUID, body: DeviceUpdate, request: Request):
    manager = request.app.state.monitoring
    async with manager.lock:
        await manager.device(str(device_id))
        updates = body.model_dump(exclude_none=True)
        if not updates:
            raise HTTPException(422, "Informe name ou is_active")
        rows = await manager.db.query("devices", "PATCH", params={
            "id": f"eq.{device_id}", "select": "id,name,created_at,is_active",
        }, data=updates)
        if updates.get("is_active") is False:
            sessions = await manager.db.query("sessions", params={
                "device_id": f"eq.{device_id}", "ended_at": "is.null", "select": "id",
            })
            for session in sessions:
                await manager.finish_locked(session["id"])
        return rows[0]


@router.post("/sessions", status_code=201)
async def start_session(body: SessionCreate, request: Request, user=Depends(authenticated_user)):
    return await request.app.state.monitoring.start(str(body.device_id), user["id"])


@router.get("/sessions")
async def list_sessions(request: Request, active: bool = False, user=Depends(authenticated_user)):
    params = {"user_id": f"eq.{user['id']}", "select": "*", "order": "started_at.desc"}
    if active:
        params["ended_at"] = "is.null"
    return await request.app.state.db.query("sessions", params=params)


@router.post("/sessions/{session_id}/end")
async def end_session(session_id: UUID, request: Request, user=Depends(authenticated_user)):
    manager = request.app.state.monitoring
    async with manager.lock:
        session = await manager.owned_session(str(session_id), user["id"])
        return await manager.finish_locked(str(session_id)) or session


@router.get("/sessions/{session_id}/events")
async def session_events(session_id: UUID, request: Request, user=Depends(authenticated_user)):
    manager = request.app.state.monitoring
    await manager.owned_session(str(session_id), user["id"])
    return await manager.db.query("events", params={
        "session_id": f"eq.{session_id}", "select": "*", "order": "created_at.desc",
    })
