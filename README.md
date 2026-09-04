# 🚀 MLOps ML Pipeline Practice

An end-to-end **Machine Learning Operations (MLOps) practice project** that demonstrates how to transform a traditional machine learning workflow into a **reproducible, version-controlled, and modular ML pipeline**.

This project uses **DVC (Data Version Control)** to manage data and pipeline stages, **DVCLive** for experiment tracking, **Scikit-learn** for model development, and **Git/GitHub** for source-code version control.

The project uses an **SMS Spam Classification** problem to demonstrate the complete ML workflow from data ingestion to model building.

---

## 📌 Project Overview

In a traditional machine learning project, data processing, feature engineering, model training, and experimentation are often performed manually in notebooks or scripts.

This project demonstrates how those steps can be organized into a reproducible MLOps pipeline:

```text
Raw Dataset
     │
     ▼
Data Ingestion
     │
     ▼
Data Preprocessing
     │
     ▼
Feature Engineering
     │
     ▼
Model Building
     │
     ▼
Model Evaluation
     │
     ▼
Experiment Tracking
     │
     ▼
Versioned ML Pipeline
```

The pipeline is orchestrated using **DVC**, allowing each stage to be reproduced whenever its dependencies or parameters change.

---

## 🎯 Objectives

The main objectives of this project are:

* Build a modular machine learning pipeline
* Separate ML workflow into independent stages
* Version datasets and ML artifacts using DVC
* Track experiments and metrics using DVCLive
* Manage model hyperparameters through YAML configuration
* Make experiments reproducible
* Track source code using Git and GitHub
* Understand the fundamentals of production-oriented ML workflows
* Practice MLOps concepts using a real classification problem

---

## 🧠 Machine Learning Problem

The project uses an **SMS Spam Classification** dataset.

The objective is to classify a message into one of two categories:

* **Ham** — legitimate/non-spam message
* **Spam** — unwanted or spam message

This is a **binary text classification** problem.

The machine learning workflow includes:

1. Loading the raw dataset
2. Cleaning and preprocessing the text
3. Transforming text into numerical features
4. Training a Random Forest classifier
5. Evaluating the trained model
6. Tracking experiments and pipeline outputs

---

## 🛠️ Technology Stack

| Technology       | Purpose                                    |
| ---------------- | ------------------------------------------ |
| **Python**       | Programming language                       |
| **Pandas**       | Data manipulation and preprocessing        |
| **NumPy**        | Numerical operations                       |
| **Scikit-learn** | Machine learning and TF-IDF/modeling       |
| **DVC**          | Data versioning and pipeline orchestration |
| **DVCLive**      | Experiment and metric tracking             |
| **PyYAML**       | Configuration management                   |
| **Git**          | Source-code version control                |
| **GitHub**       | Remote code repository                     |

### Cloud Storage

The DVC remote can be configured with cloud object storage for storing versioned datasets and model artifacts.

For example:

```text
DVC
 │
 └── Azure Blob Storage
       │
       └── Versioned ML artifacts
```

---

# 📂 Project Structure

```text
mlops-ml-pipeline-practice/
│
├── .dvc/
│   └── config
│
├── .dvcignore
├── .gitignore
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── dvclive/
│   └── experiment metrics and plots
│
├── experiments/
│   └── experiment-related files
│
├── logs/
│   └── pipeline execution logs
│
├── src/
│   ├── data-ingestion.py
│   ├── pre-processing.py
│   ├── feature-engineering.py
│   ├── model-building.py
│   └── model-evaluation.py
│
├── dvc.yaml
├── dvc.lock
├── params.yaml
├── projectflow.txt
├── spam.csv
└── README.md
```

---

# 🔄 Pipeline Stages

The ML workflow is divided into multiple stages using DVC.

## 1. Data Ingestion

The data ingestion stage is responsible for obtaining the raw dataset and preparing it for downstream processing.

```text
Raw Dataset
     ↓
Data Ingestion
     ↓
data/raw/
```

This stage ensures that the dataset is available in a consistent location.

---

## 2. Data Preprocessing

The preprocessing stage cleans the raw data and prepares it for feature engineering.

Typical preprocessing operations include:

* Removing unnecessary columns
* Handling missing values
* Removing duplicate records
* Standardizing data
* Preparing text data for feature extraction

```text
data/raw/
     ↓
Preprocessing
     ↓
data/interim/
```

---

## 3. Feature Engineering

The cleaned text is converted into numerical features using **TF-IDF (Term Frequency–Inverse Document Frequency)**.

TF-IDF represents the importance of words in a document relative to the complete collection of documents.

```text
Cleaned Text
     ↓
TF-IDF Vectorization
     ↓
Numerical Features
     ↓
data/processed/
```

The number of features is controlled through `params.yaml`.

Example:

```yaml
feature_engineering:
  max_features: 50
```

---

## 4. Model Building

The processed features are passed to a **Random Forest Classifier**.

The model configuration is maintained separately in `params.yaml`.

Example:

```yaml
model_building:
  n_estimators: 25
  random_state: 2
```

This separates model configuration from the source code and makes experimentation easier.

The trained model is saved as:

```text
models/model.pkl
```

---

## 5. Model Evaluation

The trained model is evaluated using the test dataset.

Typical classification metrics include:

* Accuracy
* Precision
* Recall
* F1-score

The evaluation results can be tracked using DVCLive.

---

# ⚙️ Configuration Management

Instead of hardcoding parameters inside Python scripts, model and pipeline parameters are maintained in:

```text
params.yaml
```

Example:

```yaml
data_ingestion:
  test_size: 0.2

feature_engineering:
  max_features: 50

model_building:
  n_estimators: 25
  random_state: 2
```

This approach makes experimentation easier because parameters can be modified without changing the underlying Python code.

---

# 🔁 DVC Pipeline

The pipeline is defined using:

```text
dvc.yaml
```

DVC tracks:

* Pipeline stages
* Dependencies
* Outputs
* Parameters
* Commands

The generated:

```text
dvc.lock
```

records the exact state of the pipeline dependencies and outputs, helping reproduce experiments consistently.

---

## ▶️ Run the Complete Pipeline

After installing the dependencies, execute:

```bash
dvc repro
```

DVC determines which pipeline stages need to be executed based on changes to:

* Source code
* Input data
* Parameters
* Dependencies

If nothing relevant has changed, DVC avoids unnecessarily rerunning unchanged stages.

---

# 📊 Experiment Tracking with DVCLive

**DVCLive** is used to track experiment metrics and outputs.

Experiment information is stored under:

```text
dvclive/
```

This makes it easier to compare different experiments and understand how changes to parameters affect model performance.

For example, you can modify:

```yaml
model_building:
  n_estimators: 25
```

to:

```yaml
model_building:
  n_estimators: 50
```

and reproduce the pipeline:

```bash
dvc repro
```

The resulting experiment can then be compared with previous runs.

---

# ☁️ DVC Remote Storage

DVC separates **code versioning** from **data/artifact versioning**.

Git stores lightweight metadata and source code, while DVC can store large datasets and model artifacts in remote object storage.

Conceptually:

```text
                 GitHub
                    │
                    │ Git
                    ▼
          ┌──────────────────┐
          │ Source Code      │
          │ dvc.yaml         │
          │ dvc.lock         │
          │ params.yaml      │
          └──────────────────┘
                    │
                    │ DVC
                    ▼
          ┌──────────────────┐
          │ Cloud Storage    │
          │                  │
          │ Dataset versions │
          │ Model artifacts  │
          │ Pipeline outputs │
          └──────────────────┘
```

For Azure Blob Storage, DVC can be configured using an Azure remote.

Example:

```bash
pip install "dvc[azure]"
```

Then authenticate with Azure CLI:

```bash
az login
```

Configure the DVC remote:

```bash
dvc remote add -d dvcstore azure://dvcstore
```

Finally, upload DVC-tracked artifacts:

```bash
dvc push
```

> **Note:** Cloud credentials, access keys, SAS tokens, and other secrets should never be committed to GitHub.

---

# 🧪 Reproducing an Experiment

A typical experiment workflow is:

```bash
# Modify parameters
# params.yaml

# Reproduce pipeline
dvc repro

# Check experiment results
dvc exp show
```

For example:

```yaml
model_building:
  n_estimators: 25
  random_state: 2
```

can be changed to:

```yaml
model_building:
  n_estimators: 50
  random_state: 2
```

Then:

```bash
dvc repro
```

DVC identifies the affected stages and executes them again.

---

# 🔧 Installation

## 1. Clone the repository

```bash
git clone https://github.com/Shashikala-11/mlops-ml-pipeline-practice.git
```

```bash
cd mlops-ml-pipeline-practice
```

## 2. Create a virtual environment

### Windows

```powershell
python -m venv myenv
```

Activate it:

```powershell
.\myenv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv myenv
source myenv/bin/activate
```

---

## 3. Install dependencies

Install the required Python packages:

```bash
pip install pandas numpy scikit-learn pyyaml dvc dvclive
```

For Azure DVC remote support:

```bash
pip install "dvc[azure]"
```

---

# ▶️ Running Individual Pipeline Stages

The individual scripts can also be executed directly.

### Data ingestion

```bash
python src/data-ingestion.py
```

### Preprocessing

```bash
python src/pre-processing.py
```

### Feature engineering

```bash
python src/feature-engineering.py
```

### Model building

```bash
python src/model-building.py
```

### Model evaluation

```bash
python src/model-evaluation.py
```

However, for reproducible execution, the recommended approach is:

```bash
dvc repro
```

---

# 📝 Logging

Each pipeline component uses Python's `logging` module to record execution information.

Logs are maintained under:

```text
logs/
```

Logging helps identify:

* Pipeline execution status
* Data loading problems
* Model training errors
* Output generation
* Exceptions and debugging information

Example:

```text
2026-09-04 - model_building - DEBUG -
Model training completed successfully
```

---

# 📦 Model Artifact

After successful model training, the trained Random Forest model is serialized using Python's `pickle` module.

Example output:

```text
models/model.pkl
```

The model can later be loaded for inference without retraining it.

---

# 🔬 MLOps Concepts Demonstrated

This project provides hands-on practice with several important MLOps concepts:

### Data Versioning

Using DVC to track datasets and large ML artifacts.

### Pipeline Automation

Using `dvc.yaml` to define reproducible pipeline stages.

### Reproducibility

Using `dvc.lock` and parameter files to reproduce experiments.

### Configuration Management

Using `params.yaml` instead of hardcoding hyperparameters.

### Experiment Tracking

Using DVCLive to record experiment metrics and outputs.

### Logging

Using Python logging for debugging and pipeline monitoring.

### Source Control

Using Git and GitHub to version ML source code.

### Remote Artifact Storage

Using cloud object storage such as Azure Blob Storage as a DVC remote.

---

# 🏗️ Overall Architecture

```text
                       ┌─────────────────┐
                       │    GitHub       │
                       │                 │
                       │ Source Code     │
                       │ dvc.yaml        │
                       │ dvc.lock        │
                       │ params.yaml     │
                       └────────┬────────┘
                                │
                                │
                                ▼
                     ┌───────────────────┐
                     │       DVC         │
                     │ Pipeline Manager  │
                     └─────────┬─────────┘
                               │
                ┌──────────────┼──────────────┐
                │              │              │
                ▼              ▼              ▼
          Data Ingestion  Preprocessing  Feature Engineering
                │              │              │
                └──────────────┼──────────────┘
                               │
                               ▼
                        Model Building
                               │
                               ▼
                        Model Evaluation
                               │
                               ▼
                    ┌─────────────────────┐
                    │      DVCLive        │
                    │ Experiment Tracking │
                    └─────────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Azure Blob        │
                    │   Storage           │
                    │                     │
                    │ Data + Artifacts    │
                    └─────────────────────┘
```

---

# 🎓 Learning Outcomes

By completing this project, the following MLOps concepts are practiced:

* Understanding the difference between ML development and MLOps
* Designing modular ML pipelines
* Using DVC for data and pipeline versioning
* Understanding pipeline dependencies
* Managing hyperparameters externally
* Reproducing ML experiments
* Tracking experiment metrics
* Managing ML artifacts
* Using cloud storage with DVC
* Integrating Git with ML workflows
* Writing production-oriented Python scripts
* Using logging for ML pipeline debugging

---

# 🚀 Future Improvements

The project can be extended into a more production-oriented MLOps system by adding:

* [ ] MLflow experiment tracking
* [ ] Model registry
* [ ] Automated testing with Pytest
* [ ] GitHub Actions CI/CD
* [ ] Docker containerization
* [ ] FastAPI model serving
* [ ] Streamlit frontend
* [ ] Model monitoring
* [ ] Data drift detection
* [ ] Automated model retraining
* [ ] Azure-based model deployment
* [ ] Infrastructure as Code using Terraform

---

# 📚 Key MLOps Workflow

The overall development cycle can be summarized as:

```text
Develop
   ↓
Version Code
   ↓
Version Data
   ↓
Define Pipeline
   ↓
Run Experiment
   ↓
Track Metrics
   ↓
Evaluate Model
   ↓
Store Artifacts
   ↓
Reproduce
   ↓
Automate
   ↓
Deploy
   ↓
Monitor
```

This project focuses primarily on the **data versioning, pipeline orchestration, reproducibility, experiment tracking, and artifact management** stages of this lifecycle.

---

# 👩‍💻 Author

**Shashikala Gupta**

GitHub:
https://github.com/Shashikala-11

Repository:
https://github.com/Shashikala-11/mlops-ml-pipeline-practice

---

## ⭐ Project Purpose

This repository is created as a hands-on learning project to understand and practice **MLOps principles by converting a machine learning workflow into a reproducible pipeline**.

The emphasis is not only on building a machine learning model, but on understanding how data, code, parameters, experiments, models, and pipeline stages can be managed systematically.

---

## 📄 License

This project is intended for educational and learning purposes.
