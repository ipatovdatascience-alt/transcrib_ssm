# ruff: noqa: RUF002, RUF003
"""LLM-клиент и детектор red flags на основе few-shot классификации."""

from __future__ import annotations

import json
import os
import re
import typing

import httpx

from app.prompt import CATEGORIES, build_prompt

_JSON_CATEGORY_RE = re.compile(r'"category"\s*:\s*"([a-z_]+)"')

OPENROUTER_MODEL = "google/gemini-2.5-flash"

# Бюджет evaluator'а — 5000 мс/пример. Держим запас на сериализацию/сеть.
_REQUEST_TIMEOUT_S = 4.5


@typing.final
class LLMClient:
    """chat-completions via OpenRouter."""

    def __init__(self) -> None:
        self.api_key = os.getenv("OPENROUTER_API_KEY", "")

    def request_completion(self, prompt_text: str, *, json_mode: bool = True) -> str | None:
        if not self.api_key:
            return None

        request_payload: dict[str, typing.Any] = {
            "model": OPENROUTER_MODEL,
            "messages": [{"role": "user", "content": prompt_text}],
        }
        if json_mode:
            request_payload["response_format"] = {"type": "json_object"}

        try:
            response = httpx.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=request_payload,
                timeout=_REQUEST_TIMEOUT_S,
            )
            return str(response.json()["choices"][0]["message"]["content"])
        except Exception:  # noqa: BLE001
            return None


def _parse_category(raw_content: str | None) -> str | None:
    """Извлекает категорию из ответа LLM, устойчиво к рассуждениям вокруг JSON.

    Промпт просит рассуждение + финальный JSON, поэтому ответ — не чистый JSON.
    Стратегия: (1) попытаться распарсить как чистый JSON; (2) найти последний
    `"category": "..."` в тексте; (3) fallback — последнее упоминание имени класса.
    Возвращает валидную категорию (включая "clean") либо None.
    """
    if not raw_content:
        return None

    stripped_text = raw_content.strip()
    try:
        decoded_json = json.loads(stripped_text)
    except (json.JSONDecodeError, TypeError):
        decoded_json = None
    if isinstance(decoded_json, dict):
        direct_value = decoded_json.get("category")
        if isinstance(direct_value, str) and direct_value in CATEGORIES:
            return direct_value

    for one_match in reversed(_JSON_CATEGORY_RE.findall(stripped_text)):
        if one_match in CATEGORIES:
            return str(one_match)

    lowered_text = stripped_text.lower()
    best_category: str | None = None
    best_position = -1
    for one_category in CATEGORIES:
        found_position = lowered_text.rfind(one_category)
        if found_position > best_position:
            best_position, best_category = found_position, one_category
    return best_category


def process_risk_detection(
    llm_client: LLMClient,
    messages: str,
) -> dict[str, typing.Any] | None:
    """Классифицирует диалог через LLM few-shot.

    `messages` — диалог, уже отформатированный в текст с ролевыми префиксами.
    Возвращает {"category": <класс>} для нарушения либо None для clean/ошибки/сбоя
    (None превращается в пустой predicted_red_flags — это и есть метка clean).
    """
    category = _parse_category(
        llm_client.request_completion(build_prompt(messages), json_mode=False),
    )
    if category is None or category == "clean":
        return None
    return {"category": category}


def load_llm() -> LLMClient:
    """Создаёт LLM-клиент при старте приложения."""
    return LLMClient()
