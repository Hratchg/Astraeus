import re
from fastapi import APIRouter, Path, Query, HTTPException
from backend.api.schemas import ExplainabilityResponse, ErrorResponse
from backend.api.dependencies import load_explainability

router = APIRouter()

VALID_HORIZONS = {1, 5, 10, 20}
_SYMBOL_RE = re.compile(r"^[A-Z]{1,6}$")


@router.get(
    "/explainability/{symbol}",
    response_model=ExplainabilityResponse,
    responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
def get_explainability(
    symbol: str = Path(description="Ticker symbol"),
    horizon: int = Query(default=5, description="Forecast horizon in trading days"),
):
    symbol = symbol.upper()
    if not _SYMBOL_RE.match(symbol):
        raise HTTPException(
            status_code=422,
            detail={"error": "invalid_symbol", "detail": "Symbol must be 1-6 uppercase letters"},
        )

    if horizon not in VALID_HORIZONS:
        raise HTTPException(
            status_code=422,
            detail={"error": "invalid_horizon", "detail": f"Horizon must be one of {sorted(VALID_HORIZONS)}"},
        )

    data = load_explainability(symbol, horizon)
    if data is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "ticker_not_found", "detail": f"No explainability data for {symbol}"},
        )

    return data
