from fastapi import APIRouter, Query
from backend.api.schemas import TickerListResponse, TickerInfo
from backend.pipeline.fetch_data import TICKER_UNIVERSE

router = APIRouter()


@router.get("/tickers", response_model=TickerListResponse)
def list_tickers(q: str = Query(default=None, description="Search query")):
    tickers = []
    for t in TICKER_UNIVERSE:
        info = TickerInfo(
            symbol=t["symbol"],
            name=t["name"],
            sector=t["sector"],
            confidence_level="high",
        )
        if q:
            query = q.lower()
            if query in t["symbol"].lower() or query in t["name"].lower():
                tickers.append(info)
        else:
            tickers.append(info)
    return TickerListResponse(tickers=tickers)
