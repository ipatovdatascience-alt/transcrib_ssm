# ruff: noqa: RUF001, RUF002
"""Файл для тестирования с eval сервисом, желательно не трогать."""

import time
import typing

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

check_router = APIRouter(tags=["Dialogue Check"])


@typing.final
class DialogueMessage(BaseModel):
    role: str = Field(description="Роль отправителя сообщения (user, support, assistant)")
    content: str = Field(description="Содержимое сообщения")


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

    messages = [
        {"role": one_message.role, "content": one_message.content} for one_message in request_body.messages
    ]

    category = http_request.app.state.red_flag_model.check_dialogue(messages)
    predicted_red_flags = [RedFlagItem(category=category)] if category else []

    processing_time_ms = int((time.perf_counter() - start_time) * 1000)

    return DialogueCheckResponse(
        session_id=request_body.session_id,
        predicted_red_flags=predicted_red_flags,
        processing_time_ms=processing_time_ms,
    )
