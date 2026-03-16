# Cervical Cancer Classification — End-to-End MLOps Project

An end-to-end Machine Learning project that trains a deep learning model, explains its predictions visually (Grad-CAM), tracks experiments (MLflow), versions data (DVC), serves everything via a REST API (FastAPI), and wraps it all in a conversational AI interface (LLaMA 3 / OpenAI) for non-technical stakeholders.

---

## Project Overview

This project demonstrates a production-grade MLOps pipeline applied to a real-world medical imaging task — classifying cervical cancer cell images into 5 categories using transfer learning (ResNet18).

| Component | Technology |
|---|---|
| Model Training | PyTorch + ResNet18 (Transfer Learning) |
| Experiment Tracking | MLflow |
| Data Versioning | DVC |
| Explainability | Grad-CAM (pytorch-grad-cam) |
| Backend API | FastAPI |
| Chatbot | LangChain + LLaMA 3 (Ollama) / OpenAI |
| Frontend UI | Streamlit |
| Containerization | Docker + Docker Compose |

---

## Features

- **5-Class Classification** — Dyskeratotic, Koilocytotic, Metaplastic, Parabasal, Superficial-Intermediate
- **Grad-CAM Explainability** — Visual heatmaps showing where the model focused in the image
- **MLflow Dashboard** — Track, compare, and reproduce training experiments
- **DVC Data Versioning** — Dataset changes are tracked just like code
- **AI Chatbot** — Non-technical stakeholders can ask questions like "Why did the model predict this?", "How reliable is it?", "What does the heatmap mean?"
- **LLM Toggle** — Switch between OpenAI API and local LLaMA 3 (via Ollama) with a single environment variable

---

## Project Structure

```
cervical_cancer_mlops/
|
|-- ml/
|   |-- dataset.py      # Data loading & augmentation
|   |-- model.py        # ResNet18 fine-tuned architecture
|   |-- train.py        # Training loop + MLflow logging
|   |-- explain.py      # Grad-CAM heatmap generation
|
|-- backend/
|   |-- main.py         # FastAPI: /predict, /explain, /chat endpoints
|
|-- frontend/
|   |-- app.py          # Streamlit UI
|
|-- data/               # DVC-tracked dataset
|-- Dockerfile.backend
|-- Dockerfile.frontend
|-- docker-compose.yml
|-- requirements*.txt
```

---

## Quick Start

### 1. Train the Model

```bash
# Activate virtual environment (Windows)
.\venv\Scripts\Activate.ps1

# Run training — logs to MLflow automatically
cd ml
python train.py

# View MLflow experiment dashboard
mlflow ui
# Open http://localhost:5000
```

### 2. Launch the App (Docker)

```bash
docker-compose up --build
```

- Streamlit UI  → http://localhost:8501
- FastAPI Docs  → http://localhost:8000/docs

---

## Switch LLM Provider

Change one line in `docker-compose.yml`:

```yaml
environment:
  - LLM_PROVIDER=ollama    # Local LLaMA 3 (free, private)
  # - LLM_PROVIDER=openai  # OpenAI API (requires key)
```

---

## MLflow Metrics Tracked

- Training and Validation Loss per epoch
- Accuracy, Precision, Recall, F1-Score
- Model artifacts (.pth weights)
- Hyperparameters (epochs, batch size, learning rate)

---

## How It Works

1. User uploads a cervical cancer cell image
2. ResNet18 model predicts the cell type and confidence score
3. Grad-CAM generates a heatmap showing where the model focused
4. The AI chatbot (LLaMA 3 or GPT) answers questions with full model context injected

---

## Tech Stack

Python · PyTorch · MLflow · DVC · FastAPI · Streamlit · LangChain · Docker · Grad-CAM · ResNet18
