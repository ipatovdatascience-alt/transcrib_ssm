# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

FastAPI service that detects "red flags" in dialogue sessions. This is a hackathon
boilerplate: the detection logic in `app/models.py` (`process_risk_detection`) is a
**stub** meant to be replaced. It returns hardcoded categories for the first 5 calls
and `None` afterward. The `LLMClient` is constructed at startup but never actually
invoked — wiring it into `process_risk_detection` is the intended work.

## Commands

All commands run through `just` (which wraps `uv`):

- `just setup` — create `.venv`, install deps from `uv.lock`, install pre-commit hooks
- `just dev-local` — run uvicorn on http://localhost:8787 with `--reload` (best for debugging)
- `just dev-docker` — run via Docker Compose Watch on http://localhost:8787 (closer to prod)
- `just audit` — `ruff check --fix` + `ruff format` + `mypy` + `flake8` (run before committing)
- `just test` — run pytest
- `just release X.Y.Z` — tag `vX.Y.Z` and push, triggering the CI/CD release+deploy pipeline

Run a single test: `uv run pytest tests/test_check.py::test_check_contract`

Interactive API docs (Swagger UI) at http://localhost:8787/docs when the server runs.

## Architecture

Three-layer request flow for the core endpoint:

1. **`app/main.py`** — creates the `FastAPI` app (`application`), registers routers, and
   in the `lifespan` handler stores a single `LLMClient` on `app.state.llm_client`.
2. **`app/routers/check.py`** — `POST /check`. Defines the request/response Pydantic
   contract, flattens `messages` into one text block via `format_dialogue`, pulls the
   shared client off `request.app.state`, calls `process_risk_detection`, and times the call.
3. **`app/models.py`** — `LLMClient` (OpenRouter chat-completions over `httpx`) plus the
   `process_risk_detection` stub. `load_llm()` is the factory used at startup.

`app/routers/health.py` exposes `GET /health` (returns `{"status": "ok"}`).

### The response contract is fixed

`tests/test_check.py` enforces the `/check` response schema that an external evaluator
relies on: `predicted_red_flags` is a list of `{"category": str}` (≤200 items, each
category ≤4096 chars) plus `session_id` and integer `processing_time_ms`. The file
header says "желательно не трогать" (preferably do not touch). Change detection logic
in `app/models.py`, not the contract in `check.py`.

## Conventions

- Python 3.11. `uv` manages all deps — never `pip install`; add deps via `pyproject.toml` + `uv sync`.
- Ruff runs with `select = ["ALL"]`; mypy runs in `strict` mode with the pydantic plugin.
  Per-file ignores live in `pyproject.toml` `[tool.ruff.lint.per-file-ignores]`.
- `app/**/__init__.py` re-exports routers and ignores F401 — register new routers in
  `app/routers/__init__.py` and include them in `app/main.py`.
- Comments and docstrings in this codebase are in Russian; match the existing style.

## Optional Hugging Face mode

Instead of OpenRouter, a local NLI model can be used. Deps are in the `hf` group
(`uv sync --group hf`, not installed by default). Commented-out HF blocks exist in
`docker-compose.local.yml` and `.github/workflows/release.yml`.

## Deployment

`just release X.Y.Z` pushes a tag, triggering `.github/workflows/release.yml`: builds and
pushes a Docker image to GHCR, SSH-deploys to the organizers' server (port 8787), then
notifies an evaluator endpoint. Requires GitHub secrets: `SSH_HOST`, `SSH_PASSWORD`,
`EVAL_TOKEN`, `OPENROUTER_API_KEY`. The container listens on port 8000 internally
(mapped to 8787 on the host).
