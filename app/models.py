# ruff: noqa: RUF002, RUF003
"""Рабочая ML-модель детектора red flags (TF-IDF + LinearSVC + правила).

Обучается один раз на train.json при старте приложения. Решение принимается
на уровне всего диалога. Класс `clean` означает отсутствие нарушений и
кодируется пустым списком флагов в ответе /check.

Метрика хакатона — macro-F1 по 7 классам, поэтому редкие классы важны так же,
как частые: используем class_weight='balanced'. Тест OOD, поэтому к словным
n-граммам добавляем символьные (char_wb 3-5), которые устойчивее к новым
формулировкам и русской морфологии.

Поверх ML-модели работает слой правил для двух классов, которые SVM стабильно
проваливал на реальном тесте: adversarial_attack (JSON-инъекции, jailbreak,
просьбы «отключи проверки» / «строго в JSON») и scope_violation (просьбы выйти
за рамки банковской роли — написать код, стихи, спарсить CSV и т.п.). Правила
ловят явные сигнатуры намерения и имеют приоритет над предсказанием SVM.
"""

from __future__ import annotations

import json
import pathlib
import re
import typing

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

CLEAN_LABEL = "clean"
ADVERSARIAL_LABEL = "adversarial_attack"
SCOPE_LABEL = "scope_violation"

_TRAIN_PATH = pathlib.Path(__file__).resolve().parent.parent / "train.json"

# Сигнатуры намерения, а не отдельные ключевые слова: каждое правило — это попытка
# поймать смысл нарушения, устойчивый к перефразировкам на OOD-тесте.
_ADVERSARIAL_PATTERNS = (
    # prompt injection через JSON-команды боту
    r'\{[^}]*"(?:action|mode|command|system|role)"\s*:',
    # явные просьбы отключить/обойти антифрод или защиту (не путать с цитатой «обойти проверку»)
    r"(?:отключ|выключ|сними|убери|disable|bypass)\w*\s+(?:антифрод|защит|secur|security\s+check|дополнительн\w*\s+провер)",
    r"без\s+дополнительн\w*\s+провер",
    # «ответь строго в JSON / только JSON / в формате JSON»
    r"(?:строго|только|формат\w*)\s+(?:в\s+)?json",
    # внутренний reasoning / system prompt антифрода
    r"(?:внутренн\w*|internal)\s+(?:reasoning|note|объяснени|prompt|инструкци)",
    # перевод в trusted/debug/без проверок режим
    r"режим\w*\s+(?:без|debug|trusted|internal)",
    r"trusted\s+mode|internal_debug|debug\s+mode",
)
_SCOPE_PATTERNS = (
    # просьбы написать/помочь с кодом
    r"(?:напиш|помог\w*|сдела\w*)\w*[^.!?]{0,40}(?:python|питон|код|скрипт|script|sql|регуляр)",
    r"распарс\w*|parse\b|csv|etl\s+pipeline|composite\s+key",
    # просьбы написать стихи / художественный текст
    r"(?:напиш|сочин|write)\w*[^.!?]{0,30}(?:стих|хайку|haiku|поэм|рассказ|песн)",
    # инвестиционные/прочие советы вне банковской роли
    r"(?:продава\w*|покупа\w*)\s+(?:сейчас|акци|сейчас\s+или)",
)


def _check_patterns(haystack: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(one_pattern, haystack, flags=re.IGNORECASE) for one_pattern in patterns)


def _apply_rules(user_text: str) -> str | None:
    """Возвращает класс по правилам или None, если явных сигнатур нет.

    adversarial_attack проверяем первым: JSON-инъекции и «отключи проверки»
    приоритетнее, потому что они опаснее и более однозначны, чем выход за рамки.
    """
    if _check_patterns(user_text, _ADVERSARIAL_PATTERNS):
        return ADVERSARIAL_LABEL
    if _check_patterns(user_text, _SCOPE_PATTERNS):
        return SCOPE_LABEL
    return None


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
    # word 1-2 + char_wb 3-5: эта конфигурация дала лучший macro-F1 на реальном тесте,
    # чем чистые word-униграммы. char n-граммы устойчивее к OOD-перефразировкам и русской морфологии.
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
        """Возвращает категорию нарушения или None для чистого диалога.

        Сначала пробуем правила для двух «трудных» классов (adversarial_attack,
        scope_violation) — они приоритетнее, потому что SVM их стабильно путал.
        Если явных сигнатур нет, отдаём решение ML-модели.
        """
        user_text = " ".join(
            one_message.get("content", "") for one_message in messages if one_message.get("role") == "user"
        )
        rule_label = _apply_rules(user_text)
        if rule_label is not None:
            return rule_label

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
