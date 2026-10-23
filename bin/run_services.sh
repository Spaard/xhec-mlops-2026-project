#!/bin/bash
# Start the services of the container: the Prefect server (UI on port 4201) in the
# background, then the prediction API (port 8001) in the foreground. `exec` makes
# uvicorn the main process of the container, so it receives Ctrl+C and `docker stop`.
set -e

prefect server start --host 0.0.0.0 --port 4201 &
exec uvicorn web_service.main:app --host 0.0.0.0 --port 8001
