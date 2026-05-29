# ruff: noqa: RUF002
"""Рабочая ML-модель детектора red flags (TF-IDF + LinearSVC).

Обучается один раз на train.json при старте приложения. Решение принимается
на уровне всего диалога. Класс `clean` означает отсутствие нарушений и
кодируется пустым списком флагов в ответе /check.

Метрика хакатона — macro-F1 по 7 классам, поэтому редкие классы важны так же,
как частые: используем class_weight='balanced'. Тест OOD, поэтому к словным
n-граммам добавляем символьные (char_wb 3-5), которые устойчивее к новым
формулировкам и русской морфологии.
"""

from __future__ import annotations

import json
import pathlib
import typing

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

CLEAN_LABEL = "clean"

_TRAIN_PATH = pathlib.Path(__file__).resolve().parent.parent / "train.json"


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
    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        sublinear_tf=True,
    )
    return Pipeline(
        [
            ("features", FeatureUnion([("word", word_vec), ("char", char_vec)])),
            ("clf", LinearSVC(class_weight="balanced")),
        ],
    )


@typing.final
class RedFlagModel:
    """Обёртка над обученным пайплайном TF-IDF + LinearSVC."""

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
    """Обучает модель на train.json. Вызывается один раз при старте приложения."""
    dialogue_texts, dialogue_labels = _extract_training_data()
    model_pipeline = _build_pipeline()
    model_pipeline.fit(dialogue_texts, dialogue_labels)
    return RedFlagModel(model_pipeline)
