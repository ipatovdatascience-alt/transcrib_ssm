# ruff: noqa: RUF001, RUF002
"""Файл для тестирования с eval сервисом, желательно не трогать."""

import time
import typing

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from app.models import process_risk_detection

check_router = APIRouter(tags=["Dialogue Check"])


@typing.final
class DialogueMessage(BaseModel):
    role: str = Field(description="Роль отправителя сообщения (user, support, assistant)")
    content: str = Field(description="Содержимое сообщения")


def format_dialogue(messages: list[DialogueMessage]) -> str:
    """Форматирует историю сообщений диалога в один текстовый блок."""
    return "\n".join(f"{one_message.role}: {one_message.content}" for one_message in messages)


@typing.final
class DialogueCheckRequest(BaseModel):
    session_id: str = Field(description="Идентификатор пользовательской сессии")
    messages: list[DialogueMessage] = Field(description="Список сообщений в диалоге")


@typing.final
class RedFlagItem(BaseModel):
    category: str = Field(description="Категория обнаруженного риска")


@typing.final
class DialogueCheckResponse(BaseModel):
    session_id: str = Field(description="Идентификатор сессии")
    predicted_red_flags: list[RedFlagItem] = Field(
        description="Список предсказанных нарушений (сравнивается eval-сервисом с expected_red_flags)",
    )
    processing_time_ms: int = Field(description="Время обработки сессии в миллисекундах")


@check_router.post("/check")
def check_dialogue(
    http_request: Request,
    request_body: DialogueCheckRequest,
) -> DialogueCheckResponse:
    start_time = time.perf_counter()

    raw_text = format_dialogue(request_body.messages)

    response = process_risk_detection(http_request.app.state.llm_client, raw_text)
    predicted_red_flags = [RedFlagItem(category=response["category"])] if response else []

    processing_time_ms = int((time.perf_counter() - start_time) * 1000)

    return DialogueCheckResponse(
        session_id=request_body.session_id,
        predicted_red_flags=predicted_red_flags,
        processing_time_ms=processing_time_ms,
    )
