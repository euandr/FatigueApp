from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class DeviceCreate(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    name: str = Field(default="CAM_01", min_length=1, max_length=120)


class DeviceUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    name: str | None = Field(default=None, min_length=1, max_length=120)
    is_active: bool | None = None


class SessionCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    device_id: UUID
