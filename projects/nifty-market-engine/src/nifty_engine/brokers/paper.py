from __future__ import annotations

from uuid import uuid4

from .base import OrderRequest, OrderResult


class PaperBroker:
    def place_market_order(self, request: OrderRequest) -> OrderResult:
        return OrderResult(
            broker_order_id=f"PAPER-{uuid4().hex[:12]}",
            status="FILLED_SIMULATED",
            raw={
                "trading_symbol": request.trading_symbol,
                "quantity": request.quantity,
                "side": request.side,
                "order_reference_id": request.order_reference_id,
            },
        )
