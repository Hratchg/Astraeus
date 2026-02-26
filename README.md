# Astraeus

Research-grade financial forecasting dashboard powered by Temporal Fusion Transformers.

Generates probabilistic multi-horizon stock predictions (1, 5, 10, 20 trading days) with model explainability visualizations. Built for a professional portfolio -- not a tutorial project.

## Architecture

- **Backend:** Python + PyTorch Forecasting + FastAPI
- **Frontend:** Next.js 15 + Shadcn UI + Tailwind + Lightweight Charts
- **ML Model:** Temporal Fusion Transformer (TFT) with quantile regression
- **Deploy:** Vercel (frontend) + Railway (API) -- hybrid pre-computed predictions

## Quick Start

### Prerequisites

- Python 3.10-3.12
- Node.js 20+
- Docker (optional)

### Backend

```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows
pip install -r backend/requirements.txt

# Run with sample data
uvicorn backend.api.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Visit http://localhost:3000

### Run the ML Pipeline

```bash
# Quick test (2 tickers, fast training)
python -m backend.pipeline.run_pipeline --tickers AAPL,NVDA --fast

# Full pipeline (all 50 tickers)
python -m backend.pipeline.run_pipeline
```

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /health` | Health check |
| `GET /tickers?q=` | List/search tickers |
| `GET /predictions/{symbol}?horizon=5` | Predictions with confidence intervals |
| `GET /explainability/{symbol}?horizon=5` | Feature importance + temporal attention |

## Project Structure

```
astraeus/
├── backend/
│   ├── api/          # FastAPI serving layer
│   ├── ml/           # Model code (TFT, features, explainability)
│   ├── pipeline/     # Offline data -> train -> predict -> export
│   ├── predictions/  # Pre-computed prediction store
│   └── tests/
├── frontend/
│   ├── app/          # Next.js App Router
│   ├── components/   # Shadcn + chart components
│   └── lib/          # API client, types
└── docs/plans/       # Design and implementation docs
```

## Security

- No API keys or credentials are committed to this repository
- All secrets are managed via `.env` (see `.env.example`)
