<div align="center">

# MLOps Project: Abalone Age Prediction

[![Python Version](https://img.shields.io/badge/python-3.10%20or%203.11-blue.svg)]()
[![Linting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-informational?logo=pre-commit&logoColor=white)](https://github.com/artefactory/xhec-mlops-project-student/blob/main/.pre-commit-config.yaml)
</div>

## 🎯 Project Overview

Welcome to your MLOps project! In this hands-on project, you'll build a complete machine learning system to predict the age of abalone (a type of sea snail) using physical measurements instead of the traditional time-consuming method of counting shell rings under a microscope.

**Your Mission**: Transform a simple ML model into a production-ready system with automated training, deployment, and prediction capabilities.

## 📊 About the Dataset

Traditionally, determining an abalone's age requires:
1. Cutting the shell through the cone
2. Staining it
3. Counting rings under a microscope (very time-consuming!)

**Your Goal**: Use easier-to-obtain physical measurements (shell weight, diameter, etc.) to predict the age automatically.

📥 **Download**: Get the dataset from the [Kaggle page](https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset)


## 🚀 Quick Start

### Prerequisites
- GitHub account
- [Kaggle account](https://www.kaggle.com/account/login?phase=startRegisterTab&returnUrl=%2F) (for dataset download)
- Python 3.10 or 3.11

### Setup Steps

1. **Fork this repository**
   - ⚠️ **Important**: Uncheck "Copy the `main` branch only" to get all project branches

2. **Add your team members** as admins to your forked repository

3. **Set up your development environment** (requires [uv](https://docs.astral.sh/uv/)):
   ```bash
   # Create the virtual environment and install app + dev dependencies
   uv sync

   # Activate it (optional, `uv run <cmd>` works without)
   source .venv/bin/activate   # on Windows: .venv\Scripts\activate

   # Install the pre-commit hooks (ruff lint + format, file checks, uv-lock)
   uv run pre-commit install
   ```

4. **Check your code locally** (the same checks run in CI on every push / PR):
   ```bash
   uv run pre-commit run --all-files
   uv run ruff check .
   uv run ruff format --check .
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

It prints the test metrics and pickles the whole pipeline (feature engineering, encoder and model) to `src/web_service/local_objects/model.pkl`, which is git-ignored. The dataset path is optional and defaults to `data/abalone.csv`. Run the tests with `uv run pytest`.

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

## 👥 Team

| Name | Email | GitHub |
|------|-------|--------|
| Gaspard Baumelou | gaspard.baumelou@gmail.com | Spaard |
| Leo Burtin | leo.burtin@hec.edu | leottawa |
| Ayush Singh | ayush.singh@hec.edu | ayu5hsingh |
| Ethan Ben Hamou | ethan.ben-hamou@polytechnique.edu | ethan-benhamou |

## 📋 What You'll Build

By the end of this project, you'll have created:

### 🤖 **Automated ML Pipeline**
- Training workflows using Prefect
- Automatic model retraining on schedule
- Reproducible model and data processing

### 🌐 **Prediction API**
- REST API for real-time predictions
- Input validation with Pydantic
- Docker containerization

### 📊 **Production-Ready Code**
- Clean, well-documented code
- Automated testing and formatting
- Proper error handling

## 📝 How to Work on This Project

### The Branch-by-Branch Approach

This project is organized into numbered branches, each representing a step in building your MLOps system. Think of it like a guided tutorial where each branch teaches you something new!

**Here's how it works**:

1. **Each branch = One pull request** with specific tasks
2. **Follow the numbers** (branch_0, branch_1, etc.) in order
3. **Read the PR instructions** (PR_0.md, PR_1.md, etc.) before starting
4. **Complete all TODOs** in that branch's code
5. **Create a pull request** when done
6. **Merge and move to the next branch**

### Step-by-Step Workflow

For each numbered branch:

```bash
# Switch to the branch
git checkout branch_number_i

# Get latest changes (except for branch_0)
git pull origin main
# Note: A VIM window might open - just type ":wq" to close it

# Push your branch
git push
```

Then:
1. 📖 Read the PR_i.md file carefully
2. 💻 Complete all the TODOs in the code
3. 🔧 Test your changes
4. 📤 Open **ONE** pull request to your main branch
5. ✅ Merge the pull request
6. 🔄 Move to the next branch

> **💡 Pro Tip**: Always integrate your previous work when starting a new branch (except branch_0)!

### 🔍 Understanding Pull Requests

Pull Requests (PRs) are how you propose and review changes before merging them into your main codebase. They're essential for team collaboration!

**Important**: When creating a PR, make sure you're merging into YOUR forked repository, not the original:

❌ **Wrong** (merging to original repo):
![PR Wrong](assets/PR_wrong.png)

✅ **Correct** (merging to your fork):
![PR Right](assets/PR_right.png)

## 💡 Development Tips

### Managing Dependencies

Use uv to manage dependencies. Install or update packages with:

```bash
uv add <package>==<version>
```

Then sync the environment and regenerate the dependency files:

```bash
uv sync
```

### Code Quality
- The pre-commit hooks will automatically format your code
- Remove all TODOs and unused code before final submission
- Use clear variable names and add docstrings

## 📊 Evaluation Criteria

Your project will be evaluated on:

### 🔍 **Code Quality**
- Clean, readable code structure
- Proper naming conventions
- Good use of docstrings and type hints

### 🎨 **Code Formatting**
- Consistent style (automated with pre-commit)
- Professional presentation

### ⚙️ **Functionality**
- Code runs without errors
- All requirements implemented correctly

### 📖 **Documentation & Reproducibility**
- Clear README with setup instructions
- Team member names and GitHub usernames
- Step-by-step instructions to run everything

### 🤝 **Collaboration**
- Effective use of Pull Requests
- Good teamwork and communication

---

## 🎯 Final Deliverables Checklist

When you're done, your repository should contain:

✅ **Automated Training Pipeline**
- [ ] Prefect workflows for model training
- [ ] Separate modules for training and inference
- [ ] Reproducible model and encoder generation

✅ **Automated Deployment**
- [ ] Prefect deployment for regular retraining

✅ **Production API**
- [ ] Working REST API for predictions
- [ ] Pydantic input validation
- [ ] Docker containerization

✅ **Professional Documentation**
- [ ] Updated README with team info
- [ ] Clear setup and run instructions
- [ ] All TODOs removed from code

---

**Ready to start? Head to branch_0 and read PR_0.md for your first task! 🚀**
