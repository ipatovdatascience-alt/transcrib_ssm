# ruff: noqa: RUF002, RUF003, PLR2004
"""Контрактные тесты пайплайна /check.

Проверяют соответствие ответа схеме evaluator'а
(CheckResponse + RedFlagItem) и жёстким лимитам: ≤200 флагов, category ≤4096 символов.
"""

import typing

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from app.main import application


@pytest.fixture(scope="module")
def client():
    with TestClient(application) as test_client:
        yield test_client


def assert_check_response(response_data: dict[str, typing.Any], expected_session_id: str) -> None:
    assert response_data["session_id"] == expected_session_id

    red_flags = response_data["predicted_red_flags"]
    assert isinstance(red_flags, list)
    assert len(red_flags) <= 200

    for one_flag in red_flags:
        assert "category" in one_flag
        assert isinstance(one_flag["category"], str)
        assert len(one_flag["category"]) <= 4096

    processing_time_value = response_data["processing_time_ms"]
    assert isinstance(processing_time_value, int)
    assert processing_time_value >= 0


def test_health(client) -> None:
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"status": "ok"}


def test_check_contract(client) -> None:
    # Базовая проверка контракта /check — без assert'ов на содержимое predicted_red_flags
    response = client.post(
        "/check",
        json={
            "session_id": "session_smoke",
            "messages": [{"role": "user", "content": "Здравствуйте."}],
        },
    )
    assert response.status_code == status.HTTP_200_OK
    assert_check_response(response.json(), "session_smoke")


def test_check_validation_missing_messages(client) -> None:
    response = client.post("/check", json={"session_id": "x"})
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


def test_check_validation_missing_session_id(client) -> None:
    response = client.post("/check", json={"messages": []})
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


def test_check_validation_invalid_message_shape(client) -> None:
    # У сообщения отсутствует обязательное поле content
    response = client.post(
        "/check",
        json={"session_id": "x", "messages": [{"role": "user"}]},
    )
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
