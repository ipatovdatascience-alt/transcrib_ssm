import typing

from fastapi import APIRouter
from pydantic import BaseModel

health_router = APIRouter(tags=["System"])


@typing.final
class HealthResponse(BaseModel):
    """Статус сервиса."""

    status: typing.Literal["ok"] = "ok"


@health_router.get("/health")
async def check_health() -> HealthResponse:
    return HealthResponse()
