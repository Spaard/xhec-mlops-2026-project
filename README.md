<div align="center">

# MLOps Project: Abalone Age Prediction

[![Python Version](https://img.shields.io/badge/python-3.10%20or%203.11-blue.svg)]()
[![Linting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-informational?logo=pre-commit&logoColor=white)](https://github.com/artefactory/xhec-mlops-project-student/blob/main/.pre-commit-config.yaml)
</div>

## 🎯 Project Overview

This project predicts the age of abalones (a type of sea snail) from physical measurements, instead of the traditional time-consuming method of counting shell rings under a microscope. It turns a machine learning model into a production-ready system:

- **Training pipeline**: a Prefect flow that cleans the data, trains the model and saves it, with a deployment that retrains it every week.
- **Prediction API**: a FastAPI service with Pydantic input validation, packaged in a Docker container.

## 📊 About the Dataset

Traditionally, determining an abalone's age requires:
1. Cutting the shell through the cone
2. Staining it
3. Counting rings under a microscope (very time-consuming!)

The model uses easier-to-obtain physical measurements (shell weight, diameter, etc.) to predict the number of rings; the age in years is the number of rings + 1.5.

📥 **Download**: get the dataset from the [Kaggle page](https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset) (see [Running the Notebooks](#-running-the-notebooks) for a command line download).

## 👥 Team

| Name | Email | GitHub |
|------|-------|--------|
| Gaspard Baumelou | gaspard.baumelou@gmail.com | Spaard |
| Leo Burtin | leo.burtin@hec.edu | leottawa |
| Ayush Singh | ayush.singh@hec.edu | ayu5hsingh |
| Ethan Ben Hamou | ethan.ben-hamou@polytechnique.edu | ethan-benhamou |

## 🗂️ Project Structure

```
├── bin/run_services.sh       # starts the Prefect server and the API (used by Docker)
├── data/                     # the dataset (git-ignored)
├── Dockerfile.app            # Docker image of the API
├── notebooks/                # EDA and modelling notebooks
├── src/
│   ├── modelling/            # training pipeline (Prefect flow and deployment)
│   └── web_service/          # prediction API (FastAPI)
│       └── local_objects/    # the trained model, model.pkl (git-ignored)
└── tests/                    # unit tests (pytest)
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or 3.11
- [uv](https://docs.astral.sh/uv/)
- [Docker](https://docs.docker.com/get-docker/), to run the API in a container

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Spaard/xhec-mlops-2026-project.git
   cd xhec-mlops-2026-project
   ```

2. **Set up the development environment**:
   ```bash
   # Create the virtual environment and install app + dev dependencies
   uv sync

   # Activate it (optional, `uv run <cmd>` works without)
   source .venv/bin/activate   # on Windows: .venv\Scripts\activate

   # Install the pre-commit hooks (ruff lint + format, file checks, uv-lock)
   uv run pre-commit install
   ```

3. **Check the code locally** (the same checks run in CI on every push / PR):
   ```bash
   uv run pre-commit run --all-files
   uv run ruff check .
   uv run ruff format --check .
   uv run pytest
   ```

## 📓 Running the Notebooks

1. **Download the dataset** into the git-ignored `data/` folder. Kaggle's download link works without an account:
   ```bash
   curl -L -o data/abalone.zip https://www.kaggle.com/api/v1/datasets/download/rodolfomendes/abalone-dataset
   unzip data/abalone.zip -d data && rm data/abalone.zip
   ```

2. **Open the notebooks** with the project's environment (`.venv`, created by `uv sync`) as the kernel, and run them from top to bottom:
   - [`notebooks/eda.ipynb`](notebooks/eda.ipynb): exploratory data analysis of the dataset
   - [`notebooks/modelling.ipynb`](notebooks/modelling.ipynb): model selection, final model and its analysis (takes about 5 minutes)

3. **Browse the experiments** logged by the modelling notebook in the MLflow UI, then open http://127.0.0.1:5001 (port 5000 is often taken by AirPlay on macOS):
   ```bash
   cd notebooks
   uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
   ```
   MLflow stores its data in `notebooks/mlflow.db` and `notebooks/mlruns/`, which are git-ignored.

## 🏋️ Training the Model

The notebook code is also available as Python modules in [`src/modelling/`](src/modelling/):

| Module | Role |
|---|---|
| `preprocessing.py` | loading, cleaning, train/test split and feature engineering |
| `training.py` | the final stacking ensemble, its training and evaluation |
| `predicting.py` | loading the pickled model and predicting |
| `utils.py` | pickling helpers |
| `main.py` | training flow (Prefect): load, clean, split, train, evaluate and save the model |
| `deployment.py` | Prefect deployment that retrains the model every week |

Train the final model (about 20 seconds, the dataset must be in `data/`, see above):

```bash
uv run python src/modelling/main.py data/abalone.csv
```

It prints the test metrics and pickles the whole pipeline (feature engineering, encoder and model) to `src/web_service/local_objects/model.pkl`, which is git-ignored. The dataset path is optional and defaults to `data/abalone.csv`.

### 🔁 Tracking and Scheduling the Training with Prefect

The training is a Prefect flow (`train-model`), and each of its steps is a Prefect task. Without a Prefect server, Prefect starts a temporary one for each run, so the command above works on its own. To keep the history of the runs and retrain the model automatically:

1. **Start the Prefect server** in its own terminal, and open the Prefect UI at http://127.0.0.1:4200:
   ```bash
   uv run prefect server start
   ```

2. **Point Prefect to this server** (once, in another terminal):
   ```bash
   uv run prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
   ```

3. **Train the model**: the flow run and its tasks appear in the **Runs** page of the Prefect UI, with their logs (including the test metrics).
   ```bash
   uv run python src/modelling/main.py
   ```

4. **Start the weekly retraining** in its own terminal, and keep it running: it creates the `weekly-retraining` deployment, which retrains the model every Monday at 03:00 UTC. To retrain it right away, click **Run** on the deployment in the **Deployments** page of the Prefect UI, or run `uv run prefect deployment run 'train-model/weekly-retraining'`.
   ```bash
   uv run python src/modelling/deployment.py
   ```

## 🌐 Prediction API

The API ([`src/web_service/`](src/web_service/)) loads the trained model when it starts and exposes two routes:

| Route | Description |
|---|---|
| `GET /` | health check |
| `POST /predict` | predicts the number of rings and the age of one or several abalones (up to 1,000 per request) |

The inputs are validated with Pydantic: besides the types, the API rejects (HTTP 422) the measurements that no real abalone can have, with the same rules as the cleaning of the training data (a non-positive measurement, a `diameter` larger than the `length`, a part heavier than the whole abalone, or a `height` / `length` ratio outside [0.05, 0.6]). If no model has been trained, `/predict` returns HTTP 503.

**The model must be trained first** (see *Training the Model* above), so that `src/web_service/local_objects/model.pkl` exists.

### 🐳 Running the API in Docker

1. **Build the image** (the trained model is copied into it, so train the model before building):
   ```bash
   docker build -f Dockerfile.app -t abalone-api .
   ```

2. **Run the container**. It starts the API on port 8001 and the Prefect server on port 4201, published on ports 8000 and 4200 of the host:
   ```bash
   docker run --rm -p 0.0.0.0:8000:8001 -p 0.0.0.0:4200:4201 abalone-api
   ```

3. **Use the API**: open the interactive documentation at http://localhost:8000/docs (click **Try it out** on `POST /predict`), or send a request:
   ```bash
   curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"abalones": [{"sex": "M", "length": 0.455, "diameter": 0.365, "height": 0.095, "whole_weight": 0.514, "shucked_weight": 0.2245, "viscera_weight": 0.101, "shell_weight": 0.15}]}'
   ```
   The response contains one prediction per abalone, in the order of the request:
   ```json
   {"predictions": [{"rings": 8.83, "age": 10.33}]}
   ```
   `sex` is `M` (male), `F` (female) or `I` (infant), and the measurements are in the units of the Kaggle dataset. The Prefect UI of the container is available at http://localhost:4200.

### Running the API without Docker

```bash
uv run uvicorn web_service.main:app --port 8001
```

The API is then available at http://localhost:8001 (documentation at http://localhost:8001/docs).

## 💡 Development Tips

### Managing Dependencies

Use uv to manage dependencies: `uv add <package>==<version>` adds an app dependency, and `uv add --dev <package>` a development one. Commit `pyproject.toml` and `uv.lock` together.

### Code Quality
- The pre-commit hooks automatically lint and format the code with ruff.
- The CI ([`.github/workflows/ci.yaml`](.github/workflows/ci.yaml)) runs ruff and the tests on Python 3.10 and 3.11 on every push and pull request.
