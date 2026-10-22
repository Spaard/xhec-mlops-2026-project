#!/bin/bash
# Start the services of the container: the Prefect server (UI on port 4201) in the
# background, then the prediction API (port 8001) in the foreground.
set -e

prefect server start --host 0.0.0.0 --port 4201 &
uvicorn web_service.main:app --host 0.0.0.0 --port 8001
