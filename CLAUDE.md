# CLAUDE.md

Guidelines for working on this repo: X-HEC MLOps project, predicting abalone age (rings) from physical measurements. Dataset: [Kaggle](https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset).

## Goal

Turn a simple ML model into a production-ready system:
- **Automated ML pipeline**: Prefect training workflows, scheduled retraining, reproducible model and encoder generation, separate modules for training and inference.
- **Automated deployment**: Prefect deployment for regular retraining.
- **Prediction API**: REST API (FastAPI), Pydantic input validation, Docker containerization.
- **Production-ready code**: clean, documented, tested, formatted, proper error handling.

## Environment

- Python 3.10 or 3.11 (`requires-python = ">=3.10,<3.12"`).
- Package manager: **uv**. Never use bare `pip install`.
  - `uv sync` creates `.venv` and installs app + dev dependencies.
  - `uv add <package>==<version>` adds an app dependency; `uv add --dev <package>` adds a dev dependency. Commit `pyproject.toml` and `uv.lock` together.
  - Run tools with `uv run <cmd>`.
- App dependencies live in `[project].dependencies`, dev tooling in `[dependency-groups].dev`.

## Code quality

- Lint/format with **ruff** (config in `pyproject.toml`); pre-commit runs it automatically. Install hooks once: `uv run pre-commit install`.
- Before committing: `uv run pre-commit run --all-files`.
- CI (`.github/workflows/ci.yaml`) runs on push and PR (Python 3.10 and 3.11): `uv sync --frozen`, `ruff check`, `ruff format --check`. Keep it green.
- Evaluation criteria: code quality, formatting, functionality, documentation/reproducibility, collaboration.

### Coding conventions

- Type hints on all function signatures; docstrings on all modules, classes and public functions.
- Clear, descriptive names (snake_case functions/variables, PascalCase classes, UPPER_CASE constants).
- Small single-purpose functions; keep training and inference in separate modules.
- Validate API inputs/outputs with Pydantic models; raise explicit errors, no bare `except`.
- No hard-coded paths or secrets; no secrets in git (`.env`, `secrets/` are ignored).
- Remove all `TODO` comments, dead code and unused files before a PR is merged (including the `PR_i.md` instruction file of that branch).
- Make training reproducible (fixed seeds, saved model + encoders).

## Git workflow

- One numbered branch = one PR with specific tasks, done in order (`0/environment_setup`, then 1, 2, ...). Read the branch's `PR_i.md` first.
- For every branch except the first, integrate previous work: `git pull origin main`.
- Open **one** PR per branch, targeting `main` of **our fork**, not the original `artefactory` repo.
- Commit messages: conventional style (`feat:`, `fix:`, `chore:`, `docs:`).
- Do not commit `.venv`, `*.egg-info`, data files or model artifacts unless explicitly required.

## Documentation

- README must contain: team member names and GitHub usernames, setup instructions, and step-by-step instructions to run everything (training, Prefect deployment, API, Docker).
- Update the README whenever a new way of running something is added.

## Final deliverables checklist

- [ ] Prefect workflows for model training
- [ ] Separate modules for training and inference
- [ ] Reproducible model and encoder generation
- [ ] Prefect deployment for regular retraining
- [ ] Working REST API for predictions with Pydantic validation
- [ ] Docker containerization
- [ ] README updated with team info and clear setup/run instructions
- [ ] All TODOs removed from code
