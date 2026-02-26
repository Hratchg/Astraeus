"""FastAPI application."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.routes import tickers, predictions, explainability

app = FastAPI(
    title="Astraeus",
    description="Research-grade financial forecasting API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tickers.router)
app.include_router(predictions.router)
app.include_router(explainability.router)


@app.get("/health")
def health():
    return {"status": "ok"}
