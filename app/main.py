import collections.abc
import contextlib
import logging
import os

from fastapi import FastAPI

from app.models import load_llm
from app.routers import check_router, health_router

app_logger = logging.getLogger("uvicorn.error")


@contextlib.asynccontextmanager
async def run_lifespan(fastapi_app: FastAPI) -> collections.abc.AsyncIterator[None]:
    fastapi_app.state.llm_client = load_llm()

    server_port = os.getenv("DEV_PORT", "8787")
    app_logger.info("Server: http://localhost:%s", server_port)
    app_logger.info("Docs:   http://localhost:%s/docs", server_port)

    yield


application = FastAPI(
    title="Red Flag Detector API",
    version="1.0.0",
    description="API для детекции red flags в диалогах.",
    lifespan=run_lifespan,
)

application.include_router(health_router)
application.include_router(check_router)
