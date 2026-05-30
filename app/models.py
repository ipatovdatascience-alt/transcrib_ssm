# ruff: noqa: RUF002, RUF003
"""Детектор red flags: голосование TF-IDF (высокий precision) + LLM (высокий recall) + арбитр.

Обе модели смотрят на диалог всегда. TF-IDF + LinearSVC (линия релиза v1.0.1, обучается на
train.json при старте) флагует редко, но почти наверняка (precision ≈91.7%); LLM через
OpenRouter (см. app/prompt.py) добирает recall. Арбитраж по голосам (см. process_risk_detection):
любой одиночный флаг проходит (OR -> recall), а при конфликте КЛАССОВ зовётся третий
LLM-вызов-судья, который выбирает один из двух классов и не может уйти в clean.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import typing

import httpx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from app.prompt import CATEGORIES, build_judge_prompt, build_prompt

_JSON_CATEGORY_RE = re.compile(r'"category"\s*:\s*"([a-z_]+)"')

OPENROUTER_MODEL = "google/gemini-2.5-flash"

CLEAN_LABEL = "clean"

_TRAIN_PATH = pathlib.Path(__file__).resolve().parent.parent / "train.json"

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


def _format_dialogue(messages: list[dict[str, str]]) -> str:
    """Склеивает реплики в один текст с ролевыми префиксами + усиленный user-блок.

    Нарушения почти всегда в user-репликах, но роли support/chatbot дают контекст,
    поэтому сохраняем все реплики и дополнительно дублируем конкатенацию user-реплик.
    """
    dialogue_lines = [f"{one_message.get('role', '')}: {one_message.get('content', '')}" for one_message in messages]
    user_messages = " ".join(
        one_message.get("content", "") for one_message in messages if one_message.get("role") == "user"
    )
    return "\n".join(dialogue_lines) + "\n[USER] " + user_messages


def _build_pipeline() -> Pipeline:
    # word 1-2 + char_wb 3-5: линия релиза v1.0.1. char n-граммы устойчивее к OOD-перефразировкам.
    word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)
    return Pipeline(
        [
            ("features", FeatureUnion([("word", word_vec), ("char", char_vec)])),
            ("clf", LinearSVC(class_weight="balanced")),
        ],
    )


@typing.final
class RedFlagModel:
    """Локальный TF-IDF + LinearSVC fallback на случай недоступности LLM."""

    def __init__(self, pipeline: Pipeline) -> None:
        self._pipeline = pipeline

    def check_dialogue(self, messages: list[dict[str, str]]) -> str | None:
        """Возвращает категорию нарушения или None для чистого диалога."""
        predicted_label = str(self._pipeline.predict([_format_dialogue(messages)])[0])
        return None if predicted_label == CLEAN_LABEL else predicted_label


def _extract_training_data() -> tuple[list[str], list[str]]:
    train_records = json.loads(_TRAIN_PATH.read_text(encoding="utf-8"))
    dialogue_texts: list[str] = []
    dialogue_labels: list[str] = []
    for one_record in train_records:
        dialogue_texts.append(_format_dialogue(one_record["messages"]))
        record_flags = one_record["expected_red_flags"]
        dialogue_labels.append(record_flags[0]["category"] if record_flags else CLEAN_LABEL)
    return dialogue_texts, dialogue_labels


def load_model() -> RedFlagModel:
    """Обучает TF-IDF fallback на train.json. Вызывается один раз при старте приложения."""
    dialogue_texts, dialogue_labels = _extract_training_data()
    model_pipeline = _build_pipeline()
    model_pipeline.fit(dialogue_texts, dialogue_labels)
    return RedFlagModel(model_pipeline)


def _get_llm_vote(llm_client: LLMClient, messages: str) -> str | None:
    """LLM-классификатор: категория нарушения, либо None для clean/недоступности."""
    category = _parse_category(llm_client.request_completion(build_prompt(messages), json_mode=False))
    return None if category == CLEAN_LABEL else category


def _resolve_conflict(llm_client: LLMClient, messages: str, tfidf_label: str, llm_label: str) -> str:
    """LLM-арбитр при конфликте классов. Fallback на TF-IDF (P≈91.7%), НЕ в clean."""
    judge_verdict = _parse_category(
        llm_client.request_completion(build_judge_prompt(messages, tfidf_label, llm_label), json_mode=False),
    )
    return judge_verdict if judge_verdict in {tfidf_label, llm_label} else tfidf_label


def process_risk_detection(
    llm_client: LLMClient,
    messages: str,
    fallback_model: RedFlagModel | None = None,
    raw_messages: list[dict[str, str]] | None = None,
) -> dict[str, typing.Any] | None:
    """Классифицирует диалог: голосование TF-IDF (высокий precision) + LLM (высокий recall) + арбитр.

    `messages` — диалог, уже отформатированный в текст с ролевыми префиксами. Обе модели
    смотрят на диалог ВСЕГДА; решение принимает арбитраж по их голосам:
      - обе clean -> clean (None);
      - флагует ровно одна -> берём её класс (OR-голос: максимум recall);
      - обе флагуют ОДИН класс -> он же;
      - обе флагуют РАЗНЫЕ классы -> зовём LLM-судью (3-й вызов, только здесь),
        он выбирает один из двух; судья не может уйти в clean -> recall не теряется.
    Возвращает {"category": <класс>} для нарушения либо None для чистого диалога.
    """
    tfidf_category = (
        fallback_model.check_dialogue(raw_messages) if fallback_model is not None and raw_messages is not None else None
    )
    llm_category = _get_llm_vote(llm_client, messages)

    if tfidf_category is None and llm_category is None:
        return None
    if tfidf_category is None:
        return {"category": llm_category}
    if llm_category is None or llm_category == tfidf_category:
        return {"category": tfidf_category}
    return {"category": _resolve_conflict(llm_client, messages, tfidf_category, llm_category)}


def load_llm() -> LLMClient:
    """Создаёт LLM-клиент при старте приложения."""
    return LLMClient()
