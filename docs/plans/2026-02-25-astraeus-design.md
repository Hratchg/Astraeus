# Astraeus — Design Document

**Date:** 2026-02-25
**Status:** APPROVED
**Author:** King Hratch + Claude

---

## Overview

Astraeus is a research-grade financial forecasting and intelligence dashboard. It uses a Temporal Fusion Transformer (TFT) to generate probabilistic multi-horizon stock predictions, served through a professional fintech-dark dashboard with model explainability visualizations.

**Portfolio goal:** Demonstrate mastery of modern deep learning, production architecture, and professional frontend engineering.

---

## Decisions Summary

| Decision | Choice |
|---|---|
| Stock scope | Sector-level, 50-100 tickers (Tech + Energy) |
| Data sources | yfinance (OHLCV) + FRED API (macro) |
| Model — Phase 1 | Temporal Fusion Transformer (PyTorch Forecasting) |
| Model — Phase 2 | GNN relational layer (GAT) on top of TFT |
| Forecast horizons | Multi-horizon: 1, 5, 10, 20 trading days |
| Output type | Probabilistic — 80% and 95% confidence intervals |
| Deploy strategy | Hybrid: Vercel (frontend) + Railway (lightweight API) |
| Pipeline/API split | Pipeline runs offline, API serves pre-computed predictions |
| Frontend | Next.js 15 + Shadcn UI + Tailwind + Lightweight Charts |
| Backend | Python 3.10+ + PyTorch Forecasting + FastAPI |

---

## MVP Features (Phase 1)

1. Candlestick chart with prediction confidence bands
2. Model explainability tab (attention maps + feature importance)
3. Ticker search & selection
4. Multi-horizon toggle (1/5/10/20 days)

## Deferred (Phase 2+)

- Macro indicators panel
- Backtesting / historical accuracy view
- Daily auto-refresh pipeline
- GNN relationship graph visualization

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    OFFLINE PIPELINE                      │
│                                                         │
│  ┌──────────┐   ┌──────────────┐   ┌────────────────┐  │
│  │ yfinance │──▶│ Feature Eng  │──▶│  TFT Training  │  │
│  │ FRED API │   │ (pandas-ta)  │   │  (PyTorch      │  │
│  └──────────┘   └──────────────┘   │  Forecasting)  │  │
│                                     └───────┬────────┘  │
│                                             │           │
│                              ┌──────────────▼────────┐  │
│                              │  Prediction + SHAP /  │  │
│                              │  Attention Export     │  │
│                              └──────────┬───────────┘  │
│                                         │              │
│                              ┌──────────▼───────────┐  │
│                              │  predictions/ store   │  │
│                              │  (JSON + Parquet)     │  │
│                              └──────────┬───────────┘  │
└─────────────────────────────────────────┼───────────────┘
                                          │
                              ┌───────────▼────────────┐
                              │   FastAPI (api/)       │
                              │   - /predictions/{tkr} │
                              │   - /explainability    │
                              │   - /tickers           │
                              │   Deployed: Railway    │
                              └───────────┬────────────┘
                                          │
                              ┌───────────▼────────────┐
                              │   Next.js Frontend     │
                              │   Deployed: Vercel     │
                              │   - Candlestick chart  │
                              │   - Confidence bands   │
                              │   - Explainability tab │
                              │   - Horizon toggle     │
                              └────────────────────────┘
```

The pipeline and API are decoupled. The pipeline writes predictions to a JSON store. The API reads from that store. No GPU needed at serving time.

---

## Directory Structure

```
astraeus/
├── backend/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI app, CORS, lifespan
│   │   ├── routes/
│   │   │   ├── predictions.py   # GET /predictions/{ticker}
│   │   │   ├── explainability.py# GET /explainability/{ticker}
│   │   │   └── tickers.py       # GET /tickers
│   │   ├── schemas.py           # Pydantic response models
│   │   └── dependencies.py      # Prediction store loader
│   │
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── model.py             # TFT model config & wrapper
│   │   ├── features.py          # Feature engineering (pandas-ta, macro)
│   │   ├── dataset.py           # TimeSeriesDataSet construction
│   │   └── explain.py           # Attention extraction, SHAP computation
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── fetch_data.py        # yfinance + FRED data download
│   │   ├── build_features.py    # Raw -> features
│   │   ├── train.py             # Train TFT with walk-forward validation
│   │   ├── predict.py           # Generate multi-horizon predictions
│   │   ├── export.py            # Write predictions + explanations to store
│   │   └── run_pipeline.py      # Orchestrator: fetch -> build -> train -> predict -> export
│   │
│   ├── tests/
│   │   ├── test_api.py
│   │   ├── test_features.py
│   │   ├── test_pipeline.py
│   │   └── test_model.py
│   │
│   ├── predictions/             # Output store (gitignored)
│   ├── predictions/sample/      # Sample data for frontend dev (committed)
│   ├── models/                  # Saved checkpoints (gitignored)
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx             # Dashboard landing
│   │   └── ticker/
│   │       └── [symbol]/
│   │           └── page.tsx     # Per-ticker detail view
│   ├── components/
│   │   ├── ui/                  # Shadcn components
│   │   ├── charts/
│   │   │   ├── CandlestickChart.tsx
│   │   │   ├── ConfidenceBands.tsx
│   │   │   └── AttentionHeatmap.tsx
│   │   ├── TickerSearch.tsx
│   │   └── HorizonToggle.tsx
│   ├── lib/
│   │   ├── api.ts               # API client
│   │   └── types.ts             # TypeScript interfaces
│   ├── tailwind.config.ts
│   ├── next.config.ts
│   └── package.json
│
├── docker-compose.yml
├── .env.example
├── .gitignore
├── README.md
└── docs/
    └── plans/
```

---

## API Contract

### GET /tickers?q={search}

```json
{
  "tickers": [
    {"symbol": "AAPL", "name": "Apple Inc.", "sector": "Technology", "confidence_level": "high"}
  ]
}
```

`confidence_level: "low"` flags cold-start tickers with < 6 months history.

### GET /predictions/{symbol}?horizon=5

```json
{
  "symbol": "AAPL",
  "horizon_days": 5,
  "generated_at": "2026-02-25T00:00:00Z",
  "model_version": "tft-v1",
  "historical": [
    {"date": "2026-02-20", "open": 182.1, "high": 184.5, "low": 181.2, "close": 183.7, "volume": 54000000}
  ],
  "predictions": [
    {"date": "2026-02-26", "median": 185.2, "lower_80": 183.1, "upper_80": 187.3, "lower_95": 181.5, "upper_95": 189.0}
  ]
}
```

Returns ~60 days historical OHLCV + forward predictions with 80% and 95% confidence intervals. `horizon` accepts 1, 5, 10, or 20.

### GET /explainability/{symbol}?horizon=5

```json
{
  "symbol": "AAPL",
  "horizon_days": 5,
  "feature_importance": [
    {"feature": "RSI_14", "importance": 0.23}
  ],
  "temporal_attention": [
    {"date": "2026-02-20", "weight": 0.08}
  ]
}
```

---

## Frontend UX

```
┌──────────┬─────────────────────────────────────────┐
│          │  AAPL - Apple Inc.          [1D][5D]... │
│  Search  │─────────────────────────────────────────│
│  ┌────┐  │                                         │
│  │    │  │  ┌─────────────────────────────────┐    │
│  └────┘  │  │                                 │    │
│          │  │    Candlestick + Confidence      │    │
│  AAPL    │  │    Bands Chart                   │    │
│  MSFT    │  │    (Lightweight Charts)          │    │
│  NVDA    │  │                                 │    │
│  GOOGL   │  └─────────────────────────────────┘    │
│  META    │                                         │
│  ...     │  [Chart] [Explainability]  <- tab toggle│
│          │─────────────────────────────────────────│
│  Sectors │  When "Explainability" tab active:      │
│  ├ Tech  │  ┌────────────────┬────────────────┐    │
│  ├ Energy│  │ Feature Import.│ Temporal Attn  │    │
│  └ Health│  │ (bar chart)    │ (heatmap)      │    │
│          │  └────────────────┴────────────────┘    │
└──────────┴─────────────────────────────────────────┘
```

- **Fintech dark theme:** `bg-zinc-950`, `text-zinc-100`, green/red for gains/losses
- **Candlestick chart:** Lightweight Charts with confidence band overlays
- **Horizon toggle:** Pill buttons (1D / 5D / 10D / 20D) above chart
- **Explainability tab:** Feature importance bar chart (Recharts) + temporal attention heatmap
- **Cold start:** Amber badge + tooltip on low-confidence tickers
- **Typography:** Monospace for numbers (tabular-nums), system sans for labels

---

## Data Leakage Prevention

- Strict temporal train/validation/test splits — no shuffling
- Features computed only from data available at prediction time
- Walk-forward validation: retrain on expanding window, predict next period
- All splits logged with timestamps for auditability
- Dedicated test asserting no future data in training features

## Cold Start Strategy

Tickers with < 6 months of history use a sector-average model with wider confidence intervals and a `confidence_level: "low"` flag surfaced in the UI.

---

## Security & Secrets

- `.env` for all keys — **gitignored, never committed**
- `.env.example` with placeholders only
- `predictions/`, `models/`, `*.parquet`, `*.pkl` in `.gitignore`
- No API keys, tokens, or credentials in any committed file
- Frontend `NEXT_PUBLIC_` vars only for non-sensitive values (API base URL)

---

## Error Handling

- API: structured errors `{"error": "ticker_not_found", "detail": "..."}`
- 404 unknown tickers, 422 invalid horizon values
- Frontend: graceful empty states, never blank screens
- Pipeline: logs to `pipeline/logs/`, atomic writes to prediction store

## Testing Strategy

- **Unit tests:** Feature engineering shapes, model wrapper, API route schemas
- **Integration test:** Fetch 1 ticker -> features -> train (tiny window) -> predict -> verify JSON
- **Frontend:** Component tests with mock data (Vitest + Testing Library)
- **Data leakage test:** Assert no future data in training features

## Dev Workflow

- `docker compose up` for local development
- `python -m backend.pipeline.run_pipeline --tickers AAPL,MSFT --fast` for quick dev runs
- Sample predictions in `predictions/sample/` for frontend development
- All changes update this design doc and `README.md`
- Git: `main` branch, feature branches, conventional commits (`feat:`, `fix:`, `docs:`)
