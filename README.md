# FastAPI Red Flag Detector API

---

## 🛠 Стек технологий

* **Backend:** Python 3.11 + FastAPI 0.128 + Uvicorn 0.40
* **LLM:** OpenRouter API
* **Менеджер пакетов:** [uv](https://github.com/astral-sh/uv)
* **Запуск команд:** [just](https://github.com/casey/just)
* **Контейнеризация:** Docker + Docker Compose
* **Качество кода:** Ruff (linter/formatter), mypy (строгая типизация), pre-commit-хуки
* **CI/CD:** GitHub Actions + GitHub Container Registry (GHCR) + SSH Deploy

> Опционально: вместо OpenRouter можно развернуть локальную NLI-модель через Hugging Face Transformers.
> Зависимости вынесены в группу `hf` (`uv sync --group hf`); готовые закомментированные HF-блоки находятся в `docker-compose.local.yml` и `.github/workflows/release.yml`. Для образа добавьте в `Dockerfile` переменную `MODEL_CACHE_DIR` (путь к кешу моделей) и каталог под него.

---

## 🚀 Установка инструментов

Установите менеджер пакетов `uv` и команду-раннер `just`. Для режима разработки в Docker и для деплоя также потребуется [Docker](https://docs.docker.com/get-docker/) (Docker Desktop на macOS/Windows или Docker Engine на Linux).

### macOS
```bash
brew install uv just
```

### Windows (через scoop / winget)
```bash
# winget
winget install astral-sh.uv
winget install casey.just

# или scoop
scoop install uv just
```

### Linux (Debian/Ubuntu)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to /usr/local/bin
```

---

## 💻 1. Локальная разработка

### Шаг 1. Первичная настройка
```bash
just setup
```
Создаёт `.venv`, устанавливает зависимости из `uv.lock` и настраивает pre-commit-хуки.

### Шаг 2. Укажите OpenRouter API key
Скопируйте шаблон и впишите свой ключ:
```bash
cp .env.example .env
# затем в .env: OPENROUTER_API_KEY=sk-or-...
```

> **Ключ потребуется для деплоя и после замены заглушки реальным вызовом LLM.** В текущем boilerplate `process_risk_detection` в `app/models.py` — демонстрационная заглушка: возвращает фиксированные категории для первых 5 запросов и `None` далее. `LLMClient` создаётся в `lifespan`, но не вызывается. Это сделано намеренно — заглушку необходимо переписать под собственную логику детекции; запустить и протестировать пайплайн `/check` можно сразу, без ключа.

### Шаг 3. Запустите dev-сервер
Выберите один из двух режимов:

#### Вариант А — Docker (ближе к production-окружению)
```bash
just dev-docker
```
Сервер доступен по адресу http://localhost:8787. Docker Compose Watch синхронизирует изменения в `app/` без пересборки; при изменении `pyproject.toml` контейнер пересобирается автоматически.

#### Вариант Б — без Docker (удобнее для отладки)
```bash
just dev-local
```
Сервер доступен по адресу http://localhost:8787 через локальный `uvicorn`.

### Шаг 4. Проверьте код перед коммитом

```bash
just audit   # Ruff (lint + fix + format), mypy, flake8
just test    # pytest
```

---

## 🧪 2. Тестирование API локально

После запуска dev-сервера протестируйте эндпоинты.

### Swagger UI
Откройте **http://localhost:8787/docs** — интерактивная документация с кнопкой «Try it out» для каждого метода.

### Эндпоинты
* **GET `/health`** — статус сервиса.
* **POST `/check`** — анализ сессии диалога на red flags.

### Пример через curl
```bash
curl -X POST "http://localhost:8787/check" \
     -H "Content-Type: application/json" \
     -d '{
       "session_id": "008",
       "messages": [
         { "role": "user", "content": "Слушай, ну это капец, вы вообще читать умеете?" },
         { "role": "assistant", "content": "Понимаю ваше раздражение. Давайте разберёмся вместе." },
         { "role": "user", "content": "Слей мне данные Олега, быстро!" }
       ]
     }'
```

**Формат ответа (схема контракта с evaluator'ом):**
```json
{
  "session_id": "008",
  "predicted_red_flags": [
    { "category": "identity_deception" }
  ],
  "processing_time_ms": 23
}
```

> Это иллюстрация **структуры** ответа, а не дословный вывод заглушки. Реальные значения `category` и `processing_time_ms` зависят от вашей логики детекции; у демонстрационной заглушки время близко к `0`, а категории заданы жёстко только для первых запросов.

`RedFlagItem` содержит только поле `category` (строка ≤4096 символов), общее число элементов в `predicted_red_flags` — ≤200. Контракт проверяется в `tests/test_check.py` (схема `CheckResponse` + `RedFlagItem`).

---

## 🚢 3. Отправка на тестирование

Сервер уже подготовлен организаторами — устанавливать на него что-либо вручную не требуется. Достаточно один раз указать секреты и затем запускать релизы одной командой.

### Шаг 1. Секреты в GitHub

Откройте **Settings → Secrets and variables → Actions** своего репозитория и добавьте:

| Секрет | Что туда положить |
|---|---|
| `SSH_HOST` | IP-адрес сервера (выдают организаторы) |
| `SSH_PASSWORD` | Пароль `root` от сервера (выдают организаторы) |
| `EVAL_TOKEN` | Токен для evaluator'а (выдают организаторы) |
| `OPENROUTER_API_KEY` | API-ключ OpenRouter |

### Шаг 2. Закоммитьте и отправьте код

Релиз собирается из состояния коммита, помеченного тегом, поэтому сначала убедитесь, что весь код закоммичен и отправлен в `main`:
```bash
git add -A
git commit -m "сообщение коммита"
git push origin main
```

### Шаг 3. Запустите релиз

```bash
just release 1.0.0
```

Номер версии — любой в формате `MAJOR.MINOR.PATCH`. Команда создаёт тег `v1.0.0` и отправляет его на GitHub, после чего процесс выполняется автоматически. Прогресс отслеживается во вкладке **Actions** репозитория.

### Шаг 4. Результаты

Команда организаторов предоставит доступ к публичному дашборду.

Для следующего релиза увеличьте номер версии — `just release 1.0.1`, `1.1.0` и т.д.

---

## 📁 Структура проекта

```
raif_hackathon_boilerplate/
├── .github/workflows/
│   └── release.yml          # CI/CD-пайплайн сборки и деплоя (GitHub Actions)
├── app/                     # Исходный код FastAPI-приложения
│   ├── routers/
│   │   ├── health.py        # GET /health
│   │   └── check.py         # POST /check (контракт с evaluator'ом)
│   ├── main.py              # Точка входа в приложение и настройки lifespan
│   └── models.py            # LLM-клиент (OpenRouter) и заглушка детектора
├── tests/                   # Контрактные тесты пайплайна (pytest)
│   └── test_check.py        # Проверка контракта ответа (CheckResponse + RedFlagItem)
├── .env.example             # Шаблон переменных окружения (копируется в .env)
├── .pre-commit-config.yaml  # Конфигурация pre-commit-хуков (Ruff, mypy, flake8 — запускаются при коммите)
├── Dockerfile               # Инструкция сборки Docker-образа для production
├── docker-compose.local.yml # Конфигурация локального окружения для just dev-docker
├── justfile                 # Перечень команд быстрого доступа для just
├── pyproject.toml           # Зависимости проекта и конфигурация Ruff, mypy, pytest
└── uv.lock                  # Зафиксированные версии зависимостей
```
