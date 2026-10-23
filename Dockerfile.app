FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

# Install the dependencies first, so this layer is cached when only the code changes
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# The model must have been trained beforehand (src/web_service/local_objects/model.pkl)
COPY src ./src
COPY bin ./bin
RUN uv sync --frozen --no-dev

# API on 8001, Prefect UI on 4201
EXPOSE 8001 4201

CMD ["bash", "bin/run_services.sh"]
