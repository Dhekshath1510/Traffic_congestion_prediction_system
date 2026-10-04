# 🚦 TrafficPulse — AI-Powered Traffic Congestion & Speed Prediction

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.12+" />
  <img src="https://img.shields.io/badge/FastAPI-0.116-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/React-19-61DAFB?style=for-the-badge&logo=react&logoColor=black" alt="React 19" />
  <img src="https://img.shields.io/badge/Vite-8-646CFF?style=for-the-badge&logo=vite&logoColor=white" alt="Vite 8" />
  <img src="https://img.shields.io/badge/PyTorch-2.13+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/XGBoost-3.0-FF6600?style=for-the-badge&logo=xgboost&logoColor=white" alt="XGBoost" />
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker" />
  <img src="https://img.shields.io/badge/License-Research_Open-green?style=for-the-badge" alt="License" />
</p>

<p align="center">
  <strong>An enterprise-grade, explainable intelligent transportation system (ITS)</strong><br />
  Combines <strong>GRU neural sequence forecasting</strong>, <strong>XGBoost multiclass congestion classification</strong>, and <strong>TreeSHAP feature attributions</strong> across a network of <strong>207 highway traffic sensors</strong> — packaged with a high-performance <strong>FastAPI backend</strong> and a responsive <strong>React 19 dashboard</strong> with Dark / Light theme switching.
</p>

<p align="center">
  <a href="#-quick-start">🚀 Quick Start</a> •
  <a href="#-web-dashboard-showcase">🖥 UI Showcase</a> •
  <a href="#-system-architecture">🏗 Architecture</a> •
  <a href="#-ml-pipeline--research">🧠 ML & Research</a> •
  <a href="#-api-documentation">📡 API Reference</a> •
  <a href="#-troubleshooting--faq">❓ FAQ</a>
</p>

---

## 📸 Web Dashboard Showcase

TrafficPulse includes a responsive, glassmorphic single-page web dashboard built with **React 19**, **Vite**, **Recharts**, and **Lucide Icons**. It includes full support for **Dark Mode**, **Light Mode**, and **System OS theme sync**.

| 🌙 Dark Mode (Analytics & Network State) | ☀️ Light Mode (Real-Time Sensor Overview) |
|:---:|:---:|
| <img src="./docs/assets/dashboard_dark.png" alt="Dark Mode Dashboard" width="100%"/> | <img src="./docs/assets/dashboard_light.png" alt="Light Mode Dashboard" width="100%"/> |

| 🎯 Multi-Sensor Prediction & SHAP Waterfalls | 💓 Live System & Microservice Health |
|:---:|:---:|
| <img src="./docs/assets/predict_page.png" alt="Prediction Interface" width="100%"/> | <img src="./docs/assets/health_page.png" alt="System Health Page" width="100%"/> |

---

## ✨ Key Features

- **🌐 Comprehensive 207-Sensor Network Coverage**: Real-time multi-variate speed forecasts across 207 sensor checkpoints concurrently.
- **⚡ Dual-Stage ML Inference Pipeline**:
  - **Stage 1 (Forecasting)**: GRU (Gated Recurrent Unit) network processing 12 historical time steps (60 minutes) to predict the next 5-minute velocity.
  - **Stage 2 (Classification)**: XGBoost gradient-boosted decision tree classifying congestion severity into `LOW`, `MODERATE`, or `SEVERE`.
- **🔍 Explainable AI (TreeSHAP)**: Every single sensor prediction includes mathematical SHAP attribution values showing *exactly why* congestion was classified (e.g., low-speed fraction, temporal rush-hour cycle, speed variance).
- **🎨 Modern Web UI**:
  - Live API status badge and sensor health indicators.
  - Interactive distribution charts (Pie charts for severity, Area graphs for speeds).
  - Searchable and paginated 207-sensor result table with expandable SHAP contribution bars.
  - 1-click **"Load Sample 12×207 Sequence"** button for immediate testing.
- **🛡 Production-Ready Backend**:
  - FastAPI with strict Pydantic v2 schemas and validation.
  - Docker Compose orchestration with PostgreSQL 16 and Redis 7.
  - Structured, asynchronous logging with Loguru.
  - Resilience fallback: system stays functional even if external DB / Redis caches are offline.

---

## 🏗 System Architecture

```mermaid
flowchart TD
    subgraph Client["🖥 Frontend (React 19 + Vite)"]
        UI["React Dashboard / Predict / Health"]
        Theme["Theme Context (Light / Dark / System)"]
        Axios["Axios API Client (:5173)"]
    end

    subgraph Gateway["⚡ Backend (FastAPI :8000)"]
        Router["API v1 Router"]
        HealthAPI["/api/v1/health/*"]
        PredAPI["/api/v1/predictions"]
        PredService["Prediction Service"]
    end

    subgraph ML["🧠 Machine Learning Engine"]
        GRU["GRU Speed Forecaster (PyTorch)\n[12 x 207] Matrix -> Next 5-Min Speeds"]
        FeatEng["Feature Builder\n[25 Engineered Features / Sensor]"]
        XGB["XGBoost Classifier\n[LOW, MODERATE, SEVERE]"]
        SHAP["TreeSHAP Explainer\nTop-5 Feature Contributions"]
    end

    subgraph Infra["🗄 Storage & Cache"]
        PG[("PostgreSQL 16 (:5433)")]
        Redis[("Redis Cache (:6379)")]
    end

    UI --> Axios
    Axios --> Router
    Router --> HealthAPI
    Router --> PredAPI
    HealthAPI --> PG
    HealthAPI --> Redis
    PredAPI --> PredService
    PredService --> GRU
    GRU --> FeatEng
    FeatEng --> XGB
    XGB --> SHAP
    SHAP --> PredService
    PredService --> PredAPI
```

---

## 🚀 Quick Start

You can run TrafficPulse either via **Docker Compose** (recommended for full stack) or as **Local Standalone Services** (ideal for rapid development).

### Option A: Docker Compose (One-Command Setup)

Ensure Docker Desktop is installed and running, then execute from the repository root:

```bash
# Clone the repository
git clone https://github.com/your-username/traffic-congestion-and-prediction.git
cd traffic-congestion-and-prediction

# Start PostgreSQL, Redis, and FastAPI backend
docker compose up -d

# Check running services
docker compose ps
```

The backend is now live at:
- **API Docs (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **API Health**: [http://localhost:8000/api/v1/health/ready](http://localhost:8000/api/v1/health/ready)

Now start the frontend in a separate terminal:
```bash
cd frontend
npm install
npm run dev
```
Open **[http://localhost:5173](http://localhost:5173)** in your browser! 🎉

---

### Option B: Local Development Setup

#### 1. Backend Setup (FastAPI & PyTorch)

```bash
# 1. Navigate to backend directory
cd backend

# 2. Set up Python 3.12 environment (using uv or venv)
# Using uv (recommended):
uv sync
.venv\Scripts\activate      # On Windows
# source .venv/bin/activate # On Linux/macOS

# Or using standard python venv:
python -m venv .venv
.venv\Scripts\activate
pip install -e .

# 3. Configure environment
cp .env.example .env

# 4. Start the FastAPI server
uvicorn app.main:app --port 8000 --reload
```

#### 2. Frontend Setup (React 19 + Vite)

```bash
# In a new terminal window:
cd frontend

# Install Node dependencies
npm install

# Start Vite dev server
npm run dev
```

> [!TIP]
> On Windows PowerShell, you can also launch the full development suite using our helper script:
> ```powershell
> .\scripts\dev.ps1
> ```

---

## 📡 API Documentation

### 1. Health & Liveness Probes

#### `GET /api/v1/health/live`
Confirms the FastAPI application process is up and healthy.
```bash
curl -X GET http://localhost:8000/api/v1/health/live
```
```json
{
  "status": "healthy",
  "service": "Traffic Congestion Prediction API",
  "version": "0.1.0"
}
```

#### `GET /api/v1/health/ready`
Deep readiness probe evaluating PostgreSQL and Redis connectivity.
```bash
curl -X GET http://localhost:8000/api/v1/health/ready
```

---

### 2. Multi-Sensor Speed & Congestion Prediction

#### `POST /api/v1/predictions`

Performs end-to-end inference across all 207 sensors: GRU speed forecasting $\to$ feature synthesis $\to$ XGBoost classification $\to$ TreeSHAP explainability.

**Quick Test using the bundled sample payload:**
```bash
curl -X POST http://localhost:8000/api/v1/predictions \
  -H "Content-Type: application/json" \
  -d @prediction_request.json
```

**Or with PowerShell:**
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/predictions" `
  -Method Post `
  -InFile "prediction_request.json" `
  -ContentType "application/json"
```

#### Request Structure
```json
{
  "sequence": [
    [64.2, 58.1, 61.3, "... 207 sensor values ..."],
    "... 12 time steps total (12 x 207 matrix) ..."
  ],
  "timestamp": "2026-10-04T12:00:00Z"
}
```

#### Response Structure (Per Sensor)
```json
{
  "timestamp": "2026-10-04T12:00:00Z",
  "forecast_horizon": "next 5 minutes",
  "sensors": [
    {
      "sensor_index": 0,
      "predicted_speed": 62.45,
      "congestion_index": 0.037,
      "congestion_class": "LOW",
      "probabilities": {
        "LOW": 0.892,
        "MODERATE": 0.078,
        "SEVERE": 0.030
      },
      "explanation": {
        "predicted_class": "LOW",
        "predicted_class_id": 0,
        "top_features": [
          { "feature": "low_speed_fraction", "contribution": 0.3241 },
          { "feature": "sensor_index", "contribution": -0.1523 },
          { "feature": "high_speed_fraction", "contribution": 0.1102 },
          { "feature": "hour", "contribution": -0.0871 },
          { "feature": "current_speed_mean", "contribution": 0.0654 }
        ]
      }
    }
  ]
}
```

---

## 🧠 ML Pipeline & Research Results

### 1. Mathematical Formulation

#### Relative Congestion Index (CI)
To account for varying sensor geometries and free-flow capacities, congestion is calculated dynamically against each sensor's historical 85th-percentile speed ($v_{\text{reference}}$):

$$\text{CI} = \max\left(0, \min\left(1, 1 - \frac{v_{\text{predicted}}}{v_{\text{reference}}}\right)\right)$$

#### Congestion Severity Categorization
- 🟢 **LOW**: $\text{CI} < 0.10$ (Traffic moving at $\ge 90\%$ free-flow speed)
- 🟡 **MODERATE**: $0.10 \le \text{CI} < 0.30$ (Traffic moving at $70\% - 90\%$ capacity)
- 🔴 **SEVERE**: $\text{CI} \ge 0.30$ (Significant delays, traffic moving below $70\%$ capacity)

---

### 2. Experimental Benchmark Results

Evaluated on the benchmark highway sensor dataset consisting of **207 loop detectors** sampled continuously at 5-minute intervals:

| Property | Experimental Value | Notes |
|:---|:---:|:---|
| **Sensors Monitored** | `207` | Full highway network spatial topology |
| **Temporal Resolution** | `5 minutes` | Granular time-step window |
| **Sequence Length** | `12 steps` | 60 minutes of rolling historical context |
| **Dataset Split** | `23,976 / 5,126 / 5,128` | Train / Validation / Test sets |
| **Forecasting Horizon** | `+5 minutes` | Real-time immediate traffic dispatch horizon |

#### Speed Forecasting Model Comparison (Test Set)
| Model Architecture | MAE (mph) | RMSE (mph) | sMAPE | Strengths |
|:---|:---:|:---:|:---:|:---|
| **Persistence (Baseline)** | 3.8795 | 9.5040 | 11.02% | Naive baseline |
| **Lightweight XGBoost** | 4.6272 | 9.2205 | 32.77% | Fast tabular baseline |
| **GRU Neural Network (Ours)** | **3.9040** | **9.4304** | **33.01%** | **Best balance for continuous time series** |

<p align="center">
  <img src="./results/figures/gru_actual_vs_predicted.png" alt="GRU Actual vs Predicted Speed" width="70%"/>
  <br /><em>Figure 1: GRU Model Actual vs Predicted Speed on unseen test sequence.</em>
</p>

#### Congestion Classification Performance (Test Set)
| Metric | Benchmark Result |
|:---|:---:|
| **Overall Accuracy** | **65.33%** |
| **Weighted F1-Score** | **0.6706** |
| **Macro Recall** | **0.6653** |
| **High-Confidence Accuracy ($\ge 0.80$)** | **95.49%** |
| **Expected Calibration Error (ECE)** | **0.0485** *(Well-calibrated probabilities)* |

<p align="center">
  <img src="./results/figures/congestion_confusion_matrix.png" alt="Confusion Matrix" width="48%"/>
  <img src="./results/figures/calibration_reliability.png" alt="Calibration Curve" width="48%"/>
  <br /><em>Figure 2: (Left) Congestion Classification Confusion Matrix. (Right) Reliability and Calibration Diagram.</em>
</p>

---

### 3. Explainable AI (XAI) with TreeSHAP

TrafficPulse does not operate as a black box. For every prediction, **TreeSHAP** computes exact Shapley feature attributions across 25 engineered features.

<p align="center">
  <img src="./results/figures/shap_global_feature_importance.png" alt="SHAP Global Feature Importance" width="75%"/>
  <br /><em>Figure 3: Global TreeSHAP Feature Attributions showing primary drivers of congestion predictions.</em>
</p>

#### Top Predictors Identified by SHAP:
1. `low_speed_fraction`: Proportion of recent sensor readings falling into the lower quartile.
2. `sensor_index`: Road segment-specific capacity and bottleneck tendency.
3. `high_speed_fraction`: Indicator of active free-flow conditions.
4. `hour` & `hour_cos`: Diurnal commute cycles and rush-hour temporal patterns.
5. `current_speed_mean`: Moving average of speed over the last 15 minutes.

---

## 📁 Repository Structure

```text
traffic-congestion-and-prediction/
│
├── compose.yml                          # Docker Compose multi-container stack
├── prediction_request.json              # Sample 12x207 API request payload
├── prediction_response.json             # Sample API response with explanations
├── README.md                            # Comprehensive project guide
│
├── docs/                                # Documentation & screenshots
│   └── assets/                          # UI screenshots and banners
│
├── backend/                             # High-performance FastAPI backend
│   ├── pyproject.toml                   # uv / pip dependencies & metadata
│   ├── Dockerfile                       # Production container build
│   ├── .env.example                     # Environment configuration template
│   ├── app/
│   │   ├── main.py                      # FastAPI application entry point
│   │   ├── api/v1/                      # Endpoints (/health, /predictions)
│   │   ├── core/                        # Pydantic Settings, exceptions, Loguru
│   │   ├── db/                          # PostgreSQL (SQLAlchemy) & Redis clients
│   │   ├── schemas/                     # Request, Response, Health schemas
│   │   ├── services/                    # Prediction orchestration pipeline
│   │   └── ml/
│   │       ├── classification/          # XGBoost classifier & feature engineering
│   │       ├── explainability/          # TreeSHAP explainer module
│   │       ├── prediction/              # GRU forecasting service
│   │       └── runtime/                 # Model registry & weights loader
│   └── tests/                           # Pytest test suite (health, config, logging)
│
├── frontend/                            # React 19 + Vite dashboard
│   ├── package.json                     # Frontend dependencies
│   ├── vite.config.js                   # Vite configuration & proxy settings
│   └── src/
│       ├── App.jsx                      # Router & layout container
│       ├── contexts/ThemeContext.jsx    # Dark / Light / System theme provider
│       ├── components/Navbar/           # Navigation & theme selector
│       ├── pages/
│       │   ├── Dashboard/               # Overview, quick stats & navigation
│       │   ├── Predict/                 # Interactive sequence tester & charts
│       │   └── Health/                  # Real-time infrastructure status
│       └── services/api.js              # Axios backend connection client
│
├── results/                             # Model artifacts & experiment logs
│   ├── figures/                         # 40+ generated research & evaluation plots
│   └── models/                          # Serialized GRU & XGBoost models
│
└── scripts/                             # Utility & pipeline execution scripts
    ├── dev.ps1                          # PowerShell one-click dev runner
    ├── prepare_dataset.py               # Raw traffic sequence data prep
    ├── train_gru.py                     # GRU model training script
    └── train_congestion_classifier.py   # XGBoost training script
```

---

## ⚙ Environment Variables

Configure backend behavior by creating a `.env` file in the `backend/` directory:

| Environment Variable | Default Value | Description |
|:---|:---|:---|
| `APP_NAME` | `Traffic Congestion Prediction API` | Service display name |
| `ENVIRONMENT` | `development` | `development`, `staging`, or `production` |
| `DEBUG` | `false` | Enables verbose debug outputs |
| `DATABASE_URL` | `postgresql+psycopg://traffic_user:traffic_pass@localhost:5433/traffic_db` | PostgreSQL connection string |
| `REDIS_URL` | `redis://localhost:6379` | Redis cache instance connection URL |
| `LOG_LEVEL` | `INFO` | Output log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

---

## 🧪 Testing & Code Quality

### Backend Automated Tests
```bash
cd backend

# Execute all pytest suites
pytest

# Execute with line-by-line coverage analysis
pytest --cov=app --cov-report=term-missing

# Run isolated health check tests
pytest tests/test_health.py -v
```

### Static Analysis & Formatting
```bash
# Code linting with Ruff
ruff check app/

# Code formatting with Black
black app/

# Static type verification
mypy app/
```

### Frontend Build Verification
```bash
cd frontend
npm run build
```

---

## ❓ Troubleshooting & FAQ

<details>
<summary><strong>Q: What if PostgreSQL or Redis is not running locally?</strong></summary>
The backend is built with fault tolerance. If PostgreSQL or Redis is unavailable, the <code>/health/ready</code> endpoint will report degraded status with details, but the core ML prediction endpoint (<code>/api/v1/predictions</code>) will still process inferences without interruption.
</details>

<details>
<summary><strong>Q: What shape must the input data have?</strong></summary>
The prediction endpoint strictly expects a 2D float array with shape <code>[12, 207]</code>: exactly 12 time steps (5-minute intervals spanning 1 hour) across all 207 highway sensors. Click the <em>"Load Sample"</em> button on the Predict web page or use <code>prediction_request.json</code> for a pre-validated test matrix.
</details>

<details>
<summary><strong>Q: Port 8000 or 5173 is already in use. How do I change ports?</strong></summary>
<ul>
  <li><strong>Backend</strong>: Run <code>uvicorn app.main:app --port 8001 --reload</code>. Update <code>frontend/src/services/api.js</code> with the new port.</li>
  <li><strong>Frontend</strong>: Run <code>npm run dev -- --port 3000</code>.</li>
</ul>
</details>

<details>
<summary><strong>Q: Does model inference require a dedicated GPU?</strong></summary>
No. The GRU forecaster and XGBoost classifier are optimized for both CPU and CUDA inference. Real-time batch inference for all 207 sensors executes in <strong>&lt; 50ms</strong> on modern CPUs.
</details>

---

## 📄 License & Attribution

This project is developed for research in **Explainable AI for Intelligent Transportation Systems**. Model artifacts and code are open for academic and developer use.

<p align="center">
  <strong>TrafficPulse</strong> — Bridging Deep Learning & Explainable Real-Time Traffic Operations<br />
  Built with ❤️ using FastAPI, PyTorch, XGBoost, and React 19.
</p>
