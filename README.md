# Astraeus

Research-grade financial forecasting dashboard powered by Temporal Fusion Transformers.

Generates probabilistic multi-horizon stock predictions (1, 5, 10, 20 trading days) with model explainability visualizations. Built for a professional portfolio -- not a tutorial project.

## Architecture

- **Backend:** Python + PyTorch Forecasting + FastAPI (21 tests passing)
- **Frontend:** Next.js 15 + Shadcn UI + Tailwind + Lightweight Charts v5 + Recharts
- **ML Model:** Temporal Fusion Transformer (TFT) with quantile regression (80% & 95% CI)
- **Deploy:** Vercel (frontend) + Railway (API) -- hybrid pre-computed predictions

```
┌─────────────────────────────────────────────────┐
│              OFFLINE PIPELINE                     │
│  yfinance + FRED → Feature Eng → TFT Training   │
│  → Predictions + Explainability → JSON Store     │
└────────────────────┬────────────────────────────┘
                     │
          ┌──────────▼───────────┐
          │   FastAPI (api/)     │
          │   /predictions/{tkr} │
          │   /explainability    │
          │   /tickers           │
          └──────────┬───────────┘
                     │
          ┌──────────▼───────────┐
          │   Next.js Frontend   │
          │   Candlestick chart  │
          │   Confidence bands   │
          │   Explainability tab │
          └──────────────────────┘
```

## Features

- **Candlestick chart** with 80% and 95% confidence interval bands (upper area + lower boundary lines)
- **Multi-horizon toggle** (1D / 5D / 10D / 20D forecast windows)
- **Model explainability** — feature importance bar chart + temporal attention heatmap
- **Ticker search** with sector grouping, active-link highlighting, and error/empty states
- **Fintech dark theme** with monospace numbers and green/red gain/loss colors
- **Input validation** — regex-guarded ticker symbols, horizon validation
- **Decoupled data loading** — prediction and explainability fetches are independent
- **Full pipeline** — fetch OHLCV data, engineer 20+ technical indicators, train TFT, generate predictions + explainability, export to JSON

## Quick Start

### Prerequisites

- Python 3.10-3.12
- Node.js 20+
- Docker (optional)

### Backend

```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows Git Bash
pip install -r backend/requirements.txt

# Run with sample data (5 tickers included)
uvicorn backend.api.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Visit http://localhost:3000

### Docker Compose

```bash
docker compose up
```

Backend at http://localhost:8000, frontend at http://localhost:3000.

### Run the ML Pipeline

```bash
# Quick test (2 tickers, fast training)
python -m backend.pipeline.run_pipeline --tickers AAPL NVDA --fast

# Full pipeline (all 50 tickers, ~30min with GPU)
python -m backend.pipeline.run_pipeline
```

The pipeline now generates both predictions and explainability data for every ticker/horizon combination.

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /health` | Health check |
| `GET /tickers?q=` | List/search tickers (50 in universe) |
| `GET /predictions/{symbol}?horizon=5` | Predictions with 80% & 95% confidence intervals |
| `GET /explainability/{symbol}?horizon=5` | Feature importance + temporal attention weights |

**Validation:** Ticker symbols must be 1-6 uppercase letters. Horizon must be 1, 5, 10, or 20.

## Project Structure

```
astraeus/
├── backend/
│   ├── api/          # FastAPI serving layer (routes, schemas, dependencies)
│   ├── ml/           # Model code (TFT, features, dataset, explainability)
│   ├── pipeline/     # Offline: fetch → features → train → predict → export
│   ├── predictions/  # Pre-computed prediction store (sample/ committed)
│   └── tests/        # 21 tests (API, features, model, pipeline, explain)
├── frontend/
│   ├── app/          # Next.js App Router (layout, pages)
│   ├── components/   # ClientShell, TickerSearch, charts, Shadcn UI
│   └── lib/          # API client, TypeScript types
├── docker-compose.yml
├── .env.example
└── docs/plans/       # Design and implementation docs
```

## Technical Indicators

The feature engineering pipeline computes 20+ indicators via pandas-ta:

| Category | Indicators |
|---|---|
| Trend | RSI(14), MACD(12,26,9), EMA(20), EMA(50), SMA(20) |
| Volatility | Bollinger Bands(20,2), ATR(14) |
| Momentum | Stochastic(14,3), Williams %R(14), ROC(10) |
| Volume | OBV, VWAP |
| Lags | 1/5/10/20-day return lags |

## Security

- No API keys or credentials are committed to this repository
- All secrets are managed via `.env` (see `.env.example`)
- CORS configured for read-only access (`GET` methods only)
- Ticker symbol input sanitized via regex (`^[A-Z]{1,6}$`)
- JSON file loading handles corrupt files gracefully

## Status

**Phase 1: Complete** — All 16 implementation tasks done, 21 tests passing, polished and reviewed.

**Phase 2 (planned):** GNN relational layer, backtesting view, daily auto-refresh pipeline, macro indicators panel.
